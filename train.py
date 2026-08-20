#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, resnet18


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class SyntheticCIFAR10(Dataset):
    def __init__(self, num_samples: int = 5000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.class_colors = [
            (255, 70, 70), (70, 170, 255), (70, 255, 120), (255, 180, 70),
            (190, 90, 255), (255, 110, 180), (110, 255, 220), (255, 220, 110),
            (100, 120, 255), (255, 120, 90),
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = idx % 10
        color = np.array(self.class_colors[label], dtype=np.uint8)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = color

        phase = (idx * 13 + label * 17) % 19
        for channel in range(3):
            offset = ((idx + 1) * (channel + 3) + label * 5) % 21
            image[:, :, channel] = np.clip(
                image[:, :, channel].astype(np.int16) + offset - 10,
                0,
                255,
            ).astype(np.uint8)

        stripe_width = 5 + (label % 4)
        stripe_positions = np.arange(32)
        if label % 2 == 0:
            stripe = ((stripe_positions[:, None] + phase) % (2 * stripe_width)) < stripe_width
            image[:, :, 0] = np.clip(image[:, :, 0].astype(np.int16) + (stripe * 110).astype(np.int16), 0, 255).astype(np.uint8)
        else:
            stripe = ((stripe_positions[None, :] + phase) % (2 * stripe_width)) < stripe_width
            image[:, :, 1] = np.clip(image[:, :, 1].astype(np.int16) + (stripe * 110).astype(np.int16), 0, 255).astype(np.uint8)

        image = Image.fromarray(image, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_name(model_name: str) -> str:
    name = model_name.strip().lower().replace("-", "_")
    aliases = {
        "mobilenet": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "resnet": "resnet18",
        "resnet_18": "resnet18",
        "resnet18": "resnet18",
        "cnn": "cnn",
        "simple_cnn": "cnn",
        "simplecnn": "cnn",
    }
    return aliases.get(name, name)


def build_model(model_name: str = "mobilenet_v2", num_classes: int = 10) -> nn.Module:
    normalized_name = normalize_model_name(model_name)

    if normalized_name in {"mobilenet", "mobilenet_v2"}:
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        return model

    if normalized_name in {"resnet", "resnet18", "resnet_18"}:
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if normalized_name in {"cnn", "simple_cnn"}:
        return SimpleCNN(num_classes=num_classes)

    available = ["mobilenet_v2", "resnet18", "cnn"]
    raise ValueError(f"Unsupported model '{model_name}'. Choose from: {available}")


def build_loaders(batch_size: int = 64, synthetic: bool = False):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if synthetic or not Path("./data/cifar-10-batches-py").exists():
        train_dataset = SyntheticCIFAR10(num_samples=5000, transform=train_transform)
        test_dataset = SyntheticCIFAR10(num_samples=1000, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return loss, acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier with a configurable backbone.")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Model backbone to train (e.g. mobilenet_v2, resnet18, cnn, resnet-18).")
    parser.add_argument("--epochs", '-e', type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", '-b', type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", '-lr', type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--output", '-o', type=str, default="model_cifar10.pth", help="Output model file path.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic CIFAR-like data instead of downloading the dataset.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = normalize_model_name(args.model)
    model = build_model(model_name=model_name)
    model.to(device)

    train_loader, test_loader = build_loaders(batch_size=args.batch_size, synthetic=args.synthetic)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss, valid_acc = evaluate(model, test_loader, device)
        scheduler.step()
        best_accuracy = max(best_accuracy, valid_acc)
        print(f"Epoch {epoch + 1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.5f} | train_loss={train_loss:.4f} | val_loss={valid_loss:.4f} | val_acc={valid_acc:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path.resolve()}")

    final_loss, final_acc = evaluate(model, test_loader, device)
    print(f"Final test loss: {final_loss:.4f}")
    print(f"Final test accuracy: {final_acc:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
