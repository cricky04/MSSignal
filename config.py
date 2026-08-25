#!/usr/bin/env python3
"""YAML configuration and hyperparameter logging system."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLogger:
    """Save and load hyperparameters to/from YAML files."""

    @staticmethod
    def create_checkpoint_dir(base_name: str = "checkpoint", checkpoints_root: str | Path = "checkpoints") -> Path:
        """
        Create a timestamped checkpoint directory.

        Args:
            base_name: Base name for the checkpoint
            checkpoints_root: Root directory for checkpoints

        Returns:
            Path to the created checkpoint directory
        """
        checkpoints_root = Path(checkpoints_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = checkpoints_root / f"{base_name}_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @staticmethod
    def save_config(config: Dict[str, Any], checkpoint_dir: str | Path, filename: str = "config.yaml") -> Path:
        """
        Save hyperparameters to a YAML file in the checkpoint directory.

        Args:
            config: Dictionary of hyperparameters/config values
            checkpoint_dir: Directory where checkpoint is saved
            filename: Name of the config file

        Returns:
            Path to the saved YAML config file
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = checkpoint_dir / filename

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return yaml_path

    @staticmethod
    def load_config(yaml_path: str | Path) -> Dict[str, Any]:
        """
        Load hyperparameters from a YAML file.

        Args:
            yaml_path: Path to the YAML config file

        Returns:
            Dictionary of hyperparameters
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config or {}

    @staticmethod
    def save_training_config(args, checkpoint_dir: str | Path) -> Path:
        """Save training hyperparameters to YAML."""
        config = {
            "training": {
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "synthetic": args.synthetic,
            },
            "data": {
                "dataset": "CIFAR-10",
                "num_classes": 10,
            },
        }
        return ConfigLogger.save_config(config, checkpoint_dir, "training_config.yaml")

    @staticmethod
    def save_stealing_config(args, checkpoint_dir: str | Path) -> Path:
        """Save model stealing attack hyperparameters to YAML."""
        config = {
            "attack": {
                "name": "knowledge_distillation",
                "victim_model": args.victim_model,
                "surrogate_model": args.surrogate_model,
                "victim_checkpoint": args.victim_path,
            },
            "distillation": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "optimizer": "AdamW",
                "loss_function": "KLDivLoss",
            },
            "query": {
                "num_samples": args.num_query_samples,
                "dataset": args.dataset,
                "use_svhn": args.dataset == "svhn",
            },
        }
        return ConfigLogger.save_config(config, checkpoint_dir, "stealing_config.yaml")


