#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from train import SyntheticCIFAR10, build_model, set_seed


def build_member_dataset(num_samples: int = 500, transform=None):
    dataset = SyntheticCIFAR10(num_samples=num_samples, transform=transform)
    return dataset


def build_nonmember_dataset(num_samples: int = 500, transform=None, dataset_name: str = "cifar10", allow_download: bool = False):
    dataset_path = "./data"

    if dataset_name == "svhn":
        svhn_exists = Path(dataset_path, "test_32x32.mat").exists()
        if not svhn_exists and allow_download:
            dataset = datasets.SVHN(root=dataset_path, split="test", download=True, transform=transform)
        elif not svhn_exists:
            raise FileNotFoundError(
                "SVHN test data is not available locally. Download it first or pass --allow-download."
            )
        else:
            dataset = datasets.SVHN(root=dataset_path, split="test", download=False, transform=transform)
    else:
        cifar_exists = Path(dataset_path, "cifar-10-batches-py").exists()
        if not cifar_exists and allow_download:
            dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
        elif not cifar_exists:
            raise FileNotFoundError(
                "CIFAR-10 test data is not available locally. Download it first or pass --allow-download."
            )
        else:
            dataset = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform)

    if num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))
    return dataset


def extract_logits(model, loader, device):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def train_shadow_model(train_loader, device, model_name: str = "mobilenet_v2", epochs=5, lr=1e-3):
    model = build_model(model_name=model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    return model


def build_attack_features(logits, labels):
    confidences = F.softmax(logits, dim=1)
    top_conf, _ = confidences.max(dim=1)
    entropy = -(confidences * torch.log(confidences + 1e-12)).sum(dim=1)
    return torch.stack([top_conf, entropy, logits.max(dim=1).values], dim=1)


def train_attack_classifier(features_in, labels_in, features_out, labels_out):
    X = torch.cat([features_in, features_out], dim=0)
    y = torch.cat([labels_in, labels_out], dim=0)
    model = nn.Sequential(
        nn.Linear(X.size(1), 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for _ in range(20):
        for x, target in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

    return model


def main():
    parser = argparse.ArgumentParser(description="Membership inference attack against a victim model.")
    parser.add_argument("--victim-path", type=str, default="model_cifar10.pth", help="Path to victim checkpoint.")
    parser.add_argument("--model", choices=["mobilenet_v2", "resnet18", "cnn"], default="mobilenet_v2", help="Model architecture to train and attack.")
    parser.add_argument("--epochs", type=int, default=5, help="Shadow-model training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Attack training rate.")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to use as non-members.")
    parser.add_argument("--dataset", choices=["cifar10", "svhn"], default="cifar10", help="Non-member dataset to use for the attack.")
    parser.add_argument("--use-real-cifar-test", action="store_true", help="Deprecated alias for --dataset cifar10 with a real CIFAR-10 test set.")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the dataset when the test split is missing.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_name = args.dataset
    if args.use_real_cifar_test:
        dataset_name = "cifar10"

    member_data = build_member_dataset(num_samples=args.samples, transform=transform)
    nonmember_data = build_nonmember_dataset(
        num_samples=args.samples,
        transform=transform,
        dataset_name=dataset_name,
        allow_download=args.allow_download,
    )
    member_loader = DataLoader(member_data, batch_size=args.batch_size, shuffle=False)
    nonmember_loader = DataLoader(nonmember_data, batch_size=args.batch_size, shuffle=False)

    victim = build_model(model_name=args.model).to(device)
    victim.load_state_dict(torch.load(args.victim_path, map_location=device))
    victim.eval()

    member_logits, member_labels = extract_logits(victim, member_loader, device)
    nonmember_logits, nonmember_labels = extract_logits(victim, nonmember_loader, device)

    shadow_train = build_member_dataset(num_samples=args.samples, transform=transform)
    shadow_loader = DataLoader(shadow_train, batch_size=args.batch_size, shuffle=True)
    shadow_model = train_shadow_model(shadow_loader, device, model_name=args.model, epochs=args.epochs, lr=args.learning_rate)

    shadow_member_logits, _ = extract_logits(shadow_model, member_loader, device)
    shadow_nonmember_logits, _ = extract_logits(shadow_model, nonmember_loader, device)

    in_features = build_attack_features(shadow_member_logits, member_labels)
    out_features = build_attack_features(shadow_nonmember_logits, nonmember_labels)
    attack_labels_in = torch.ones(len(in_features), dtype=torch.long)
    attack_labels_out = torch.zeros(len(out_features), dtype=torch.long)

    attack_model = train_attack_classifier(in_features, attack_labels_in, out_features, attack_labels_out)

    victim_member_features = build_attack_features(member_logits, member_labels)
    victim_nonmember_features = build_attack_features(nonmember_logits, nonmember_labels)

    attack_member_preds = attack_model(victim_member_features).argmax(dim=1)
    attack_nonmember_preds = attack_model(victim_nonmember_features).argmax(dim=1)

    member_acc = (attack_member_preds == 1).float().mean().item()
    nonmember_acc = (attack_nonmember_preds == 0).float().mean().item()
    overall = 0.5 * (member_acc + nonmember_acc)

    print(f"Membership inference accuracy on members: {member_acc:.4f}")
    print(f"Membership inference accuracy on non-members: {nonmember_acc:.4f}")
    print(f"Overall attack accuracy: {overall:.4f}")


if __name__ == "__main__":
    main()
