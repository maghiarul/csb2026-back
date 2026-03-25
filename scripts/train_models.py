from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b3,
    resnet50,
    vit_b_16,
)
from tqdm import tqdm

from app.services.plant_dataset import collect_image_samples


NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = ["resnet50", "efficientnet_b3", "vit_b_16"]


@dataclass
class TrainingResult:
    model_name: str
    best_val_accuracy: float
    best_val_top5_accuracy: float
    epochs: int
    class_count: int
    sample_count: int
    checkpoint_path: str


class PlantImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train medicinal plant classifiers")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("models") / "medicinal plants",
        help="Path to medicinal plant folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "trained",
        help="Directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help="Models to train",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_learning_rate(model_name: str, raw_learning_rate: float | None) -> float:
    if raw_learning_rate is not None:
        return raw_learning_rate
    if model_name == "resnet50":
        return 3e-4
    if model_name == "efficientnet_b3":
        return 2e-4
    return 1e-4


def build_model(model_name: str, num_classes: int, use_pretrained: bool) -> tuple[nn.Module, int]:
    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, 224

    if model_name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model, 300

    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model, 224

    raise ValueError(f"Unsupported model: {model_name}")


def build_transforms(input_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )
    return train_transform, valid_transform


def build_indexed_samples(dataset_root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    raw_samples = collect_image_samples(dataset_root)
    if not raw_samples:
        raise ValueError(f"No images found in {dataset_root}")

    class_names = sorted({class_name for _, class_name in raw_samples})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    indexed_samples = [(image_path, class_to_index[class_name]) for image_path, class_name in raw_samples]
    return indexed_samples, class_names


def split_train_validation(
    samples: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    labels = [label for _, label in samples]
    unique_labels = set(labels)

    use_stratify = True
    for label in unique_labels:
        if labels.count(label) < 2:
            use_stratify = False
            break

    if use_stratify:
        train_samples, valid_samples = train_test_split(
            samples,
            test_size=val_split,
            random_state=seed,
            stratify=labels,
        )
    else:
        train_samples, valid_samples = train_test_split(
            samples,
            test_size=val_split,
            random_state=seed,
            stratify=None,
        )

    return train_samples, valid_samples


def evaluate(model: nn.Module, data_loader: DataLoader[tuple[Tensor, int]], device: torch.device) -> float:
    model.eval()
    total = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            topk = min(5, logits.shape[1])
            _, predictions = logits.topk(topk, dim=1)
            top1_predictions = predictions[:, 0]
            total += labels.size(0)
            correct_top1 += int((top1_predictions == labels).sum().item())
            correct_top5 += int(predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item())

    if total == 0:
        return 0.0, 0.0

    return correct_top1 / total, correct_top5 / total


def build_balanced_sampler(samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    label_counts = Counter(label for _, label in samples)
    weights = [1.0 / label_counts[label] for _, label in samples]
    weight_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weight_tensor, num_samples=len(samples), replacement=True)


def make_cosine_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_model(
    model_name: str,
    samples: list[tuple[Path, int]],
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingResult:
    model, input_size = build_model(model_name, len(class_names), use_pretrained=not args.no_pretrained)
    train_transform, valid_transform = build_transforms(input_size)

    train_samples, valid_samples = split_train_validation(samples, val_split=args.val_split, seed=args.seed)
    use_balanced_sampler = not args.no_balanced_sampler

    train_dataset = PlantImageDataset(train_samples, transform=train_transform)
    valid_dataset = PlantImageDataset(valid_samples, transform=valid_transform)

    train_sampler = build_balanced_sampler(train_samples) if use_balanced_sampler else None
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = model.to(device)
    learning_rate = resolve_learning_rate(model_name, args.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(total_steps - 1, max(0, args.warmup_epochs * steps_per_epoch))
    scheduler = make_cosine_with_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=pin_memory)

    best_accuracy = 0.0
    best_top5_accuracy = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{model_name}.pt"
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=pin_memory):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1

            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_accuracy, val_top5_accuracy = evaluate(model, valid_loader, device)

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_top5_accuracy = val_top5_accuracy
            torch.save(
                {
                    "model_name": model_name,
                    "class_names": class_names,
                    "input_size": input_size,
                    "state_dict": model.state_dict(),
                    "val_accuracy": best_accuracy,
                    "val_top5_accuracy": best_top5_accuracy,
                },
                checkpoint_path,
            )

        print(
            f"{model_name} epoch {epoch + 1}/{args.epochs} "
            f"val_top1={val_accuracy:.4f} val_top5={val_top5_accuracy:.4f}"
        )

    return TrainingResult(
        model_name=model_name,
        best_val_accuracy=best_accuracy,
        best_val_top5_accuracy=best_top5_accuracy,
        epochs=args.epochs,
        class_count=len(class_names),
        sample_count=len(samples),
        checkpoint_path=str(checkpoint_path),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    samples, class_names = build_indexed_samples(args.dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[TrainingResult] = []
    for model_name in args.models:
        result = train_one_model(
            model_name=model_name,
            samples=samples,
            class_names=class_names,
            args=args,
            device=device,
        )
        results.append(result)
        print(f"{model_name}: best_val_accuracy={result.best_val_accuracy:.4f}")

    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
