#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import build_model, normalize_model_name, set_seed


def load_victim(path: str, device: torch.device, model_name: str = "mobilenet_v2"):
    model = build_model(model_name=model_name)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_query_loader(batch_size: int = 64, num_samples: int = 2000, use_svhn: bool = True):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if use_svhn:
        dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        if num_samples < len(dataset):
            dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))
    else:
        dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
        if num_samples < len(dataset):
            dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def evaluate_agreement(victim, surrogate, loader, device):
    victim.eval()
    surrogate.eval()
    agreement = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            victim_logits = victim(images)
            surrogate_logits = surrogate(images)

            victim_pred = victim_logits.argmax(dim=1)
            surrogate_pred = surrogate_logits.argmax(dim=1)
            agreement += (victim_pred == surrogate_pred).sum().item()
            total += labels.size(0)

    return agreement / max(total, 1)


def distill(victim, surrogate, loader, optimizer, device, temperature: float = 5.0):
    victim.eval()
    surrogate.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    total_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            victim_logits = victim(images)
            soft_targets = F.softmax(victim_logits / temperature, dim=1)

        surrogate_logits = surrogate(images)
        surrogate_probs = F.log_softmax(surrogate_logits / temperature, dim=1)
        loss = criterion(surrogate_probs, soft_targets) * temperature * temperature

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Simulate a model stealing attack using knowledge distillation.")
    parser.add_argument("--victim-path", type=str, default="model_cifar10.pth", help="Path to the victim model checkpoint.")
    parser.add_argument("--victim-model", type=str, default="mobilenet_v2", help="Victim model architecture (e.g. mobilenet_v2, resnet18, cnn, resnet-18).")
    parser.add_argument("--surrogate-model", type=str, default="mobilenet_v2", help="Surrogate model architecture (e.g. mobilenet_v2, resnet18, cnn, resnet-18).")
    parser.add_argument("--epochs", type=int, default=8, help="Number of distillation epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Query batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Surrogate optimizer learning rate.")
    parser.add_argument("--temperature", type=float, default=5.0, help="Softmax temperature for distillation.")
    parser.add_argument("--num-query-samples", type=int, default=2000, help="Number of unlabeled query samples to use.")
    parser.add_argument("--dataset", choices=["svhn", "cifar10"], default="svhn", help="Query dataset for model stealing.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    victim_model_name = normalize_model_name(args.victim_model)
    surrogate_model_name = normalize_model_name(args.surrogate_model)
    victim = load_victim(args.victim_path, device, model_name=victim_model_name)
    surrogate = build_model(model_name=surrogate_model_name).to(device)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=args.learning_rate)
    query_loader = build_query_loader(batch_size=args.batch_size, num_samples=args.num_query_samples, use_svhn=(args.dataset == "svhn"))

    for epoch in range(args.epochs):
        loss = distill(victim, surrogate, query_loader, optimizer, device, temperature=args.temperature)
        agreement = evaluate_agreement(victim, surrogate, query_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | distillation_loss={loss:.4f} | agreement={agreement:.4f}")

    attack_path = Path(f"surrogate_{surrogate_model_name}_cifar10.pth")
    torch.save(surrogate.state_dict(), attack_path)
    print(f"Saved surrogate model to {attack_path.resolve()}")

    final_agreement = evaluate_agreement(victim, surrogate, query_loader, device)
    print(f"Final agreement with victim: {final_agreement:.4f}")


if __name__ == "__main__":
    main()
