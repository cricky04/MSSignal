# MobileNet CIFAR-10 Boilerplate

This project provides a minimal PyTorch image-classification boilerplate using MobileNetV2 on the CIFAR-10 dataset.

## Features

- CIFAR-10 dataset loading with torchvision
- MobileNetV2 backbone
- Custom classifier head for 10 classes
- Training loop, validation loop, and model saving
- CLI options for epochs, batch size, and learning rate

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 5 --batch-size 64 --learning-rate 1e-3
python train.py --model cnn --epochs 5 --batch-size 64 --learning-rate 1e-3
python train.py --model resnet18 --epochs 5 --batch-size 64 --learning-rate 1e-3
```

Supported backbones include `mobilenet_v2`, `resnet18` (also `resnet-18`), and `cnn`.
The script saves a model to `model_cifar10.pth` by default unless you pass `--output`.

## Notes

This is a lightweight boilerplate intended for quick experimentation or prototyping before fine-tuning or adding data augmentation and transfer learning.
