from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_CLASSES = [
    "Calendula officinalis",
    "Hypericum perforatum",
    "Alliaria petiolata",
    "Lamium amplexicaule",
    "Lamium album",
    "Mercurialis annua",
    "Mercurialis perennis",
    "Trifolium pratense",
    "Trifolium repens",
    "Trifolium fragiferum",
    "Lactuca serriola",
    "Lactuca muralis",
    "Papaver rhoeas",
    "Papaver dubium",
    "Papaver argemone",
    "Cirsium arvense",
    "Cirsium vulgare",
    "Dryopteris filix-mas",
    "Osmunda regalis",
    "Daucus carota",
    "Aegopodium podagraria",
]
SUPPORTED_MODELS = ["efficientnet_b3", "resnet50", "vit_b_16"]


@dataclass
class Sample:
    image_path: Path
    class_name: str
    split: str | None


@dataclass
class ModelRunResult:
    model_name: str
    accuracy: float
    recall_macro: float
    f1_macro: float
    confusion_matrix_path: str
    checkpoint_path: str


class PlantDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[Sample], class_to_index: dict[str, int], transform: transforms.Compose):
        self.samples = samples
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[sample.class_name]
        return tensor, label


def normalize_name(name: str) -> str:
    collapsed = " ".join(name.replace("_", " ").replace("-", " ").strip().split())
    return collapsed.casefold()


def normalize_binomial_name(name: str) -> str:
    text = " ".join(name.replace("_", " ").strip().split())
    tokens = [token for token in re.split(r"\s+", text) if token]
    cleaned_tokens: list[str] = []
    for token in tokens:
        cleaned = token.strip(".,;:()[]{}\"'")
        if cleaned:
            cleaned_tokens.append(cleaned)
    if len(cleaned_tokens) >= 2:
        return f"{cleaned_tokens[0]} {cleaned_tokens[1]}".casefold()
    return " ".join(cleaned_tokens).casefold()


def normalize_split(name: str) -> str | None:
    key = name.strip().lower()
    if key in {"train", "training"}:
        return "train"
    if key in {"val", "valid", "validation"}:
        return "val"
    if key in {"test", "testing"}:
        return "test"
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate EfficientNet-B3, ResNet50 and ViT-B/16 on 20 target PlantNet-300K classes"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models") / "plantnet20_trained")
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS, choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--allow-partial-classes", action="store_true")
    return parser.parse_args()


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
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def discover_split_folders(dataset_root: Path) -> dict[str, Path]:
    split_paths: dict[str, Path] = {}
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        normalized = normalize_split(child.name)
        if normalized is not None:
            split_paths[normalized] = child
    return split_paths


def list_close_species_names(species_id_to_name: dict[str, str], target_name: str, limit: int = 5) -> list[str]:
    genus = target_name.split()[0].casefold()
    candidates = [name for name in species_id_to_name.values() if name.casefold().startswith(genus + " ")]
    return sorted(candidates)[:limit]


def load_plantnet_species_map(dataset_root: Path) -> dict[str, str]:
    species_map_path = dataset_root / "plantnet300K_species_id_2_name.json"
    if not species_map_path.exists():
        return {}
    species_map_raw = json.loads(species_map_path.read_text(encoding="utf-8"))
    if not isinstance(species_map_raw, dict):
        return {}
    return {str(species_id): str(species_name) for species_id, species_name in species_map_raw.items()}


def collect_samples_from_plantnet_images(
    dataset_root: Path,
    target_classes: list[str],
    allow_partial_classes: bool,
) -> list[Sample]:
    images_root = dataset_root / "images"
    if not images_root.exists() or not images_root.is_dir():
        return []

    species_id_to_name = load_plantnet_species_map(dataset_root)
    if not species_id_to_name:
        return []

    target_binomial_to_label = {normalize_binomial_name(class_name): class_name for class_name in target_classes}
    accepted_species_ids: dict[str, str] = {}
    for species_id, full_species_name in species_id_to_name.items():
        binomial = normalize_binomial_name(full_species_name)
        target_label = target_binomial_to_label.get(binomial)
        if target_label is not None:
            accepted_species_ids[species_id] = target_label

    discovered_classes = sorted(set(accepted_species_ids.values()))
    missing_classes = sorted(set(target_classes) - set(discovered_classes))
    if missing_classes:
        lines = []
        for missing in missing_classes:
            close_matches = list_close_species_names(species_id_to_name, missing)
            if close_matches:
                lines.append(f"- {missing}: close matches in dataset: {', '.join(close_matches)}")
            else:
                lines.append(f"- {missing}: no same-genus matches found")
        details = "\n".join(lines)
        if not allow_partial_classes:
            raise ValueError(
                "PlantNet-300K is missing some required target classes.\n"
                f"Missing ({len(missing_classes)}): {', '.join(missing_classes)}\n"
                f"Hints:\n{details}"
            )
        print(
            "Warning: continuing with partial class set because --allow-partial-classes was provided.\n"
            f"Missing ({len(missing_classes)}): {', '.join(missing_classes)}\n"
            f"Hints:\n{details}"
        )

    collected: list[Sample] = []
    for split_name in ["train", "val", "test"]:
        split_dir = images_root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            target_label = accepted_species_ids.get(species_dir.name)
            if target_label is None:
                continue
            for image_path in species_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    collected.append(Sample(image_path=image_path, class_name=target_label, split=split_name))
    return collected


def collect_samples_from_class_folders(root: Path, split: str | None) -> list[Sample]:
    samples: list[Sample] = []
    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        for image_path in class_folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(image_path=image_path, class_name=class_name, split=split))
    return samples


def load_filtered_samples(
    dataset_root: Path,
    target_classes: list[str],
    allow_partial_classes: bool,
) -> list[Sample]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    collected = collect_samples_from_plantnet_images(
        dataset_root=dataset_root,
        target_classes=target_classes,
        allow_partial_classes=allow_partial_classes,
    )

    if not collected:
        split_dirs = discover_split_folders(dataset_root)
        if split_dirs:
            for split_name in ["train", "val", "test"]:
                split_dir = split_dirs.get(split_name)
                if split_dir is not None:
                    collected.extend(collect_samples_from_class_folders(split_dir, split=split_name))
        else:
            collected.extend(collect_samples_from_class_folders(dataset_root, split=None))

    target_map = {normalize_name(class_name): class_name for class_name in target_classes}
    filtered: list[Sample] = []
    for sample in collected:
        normalized = normalize_name(sample.class_name)
        if normalized in target_map:
            filtered.append(
                Sample(
                    image_path=sample.image_path,
                    class_name=target_map[normalized],
                    split=sample.split,
                )
            )

    if not filtered:
        raise ValueError("No matching samples found for the requested 20 classes")

    discovered_classes = sorted({sample.class_name for sample in filtered})
    missing_classes = sorted(set(target_classes) - set(discovered_classes))
    if missing_classes:
        if allow_partial_classes:
            print(
                "Warning: continuing with partial class set because --allow-partial-classes was provided.\n"
                f"Missing ({len(missing_classes)}): {', '.join(missing_classes)}"
            )
            return filtered
        missing = ", ".join(missing_classes)
        raise ValueError(f"Dataset does not contain all required classes. Missing: {missing}")

    return filtered


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    has_original_split = any(sample.split is not None for sample in samples)
    if has_original_split:
        train_samples = [sample for sample in samples if sample.split == "train"]
        val_samples = [sample for sample in samples if sample.split == "val"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        if train_samples and val_samples and test_samples:
            return train_samples, val_samples, test_samples

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    labels = [sample.class_name for sample in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [sample.class_name for sample in temp_samples]
    val_portion_from_temp = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - val_portion_from_temp),
        random_state=seed,
        stratify=temp_labels,
    )
    return train_samples, val_samples, test_samples


def maybe_cap_samples(samples: list[Sample], cap: int | None, seed: int) -> list[Sample]:
    if cap is None or cap <= 0 or len(samples) <= cap:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    selected = indices[:cap]
    return [samples[index] for index in selected]


def evaluate_predictions(y_true: list[int], y_pred: list[int], num_classes: int) -> tuple[float, float, float, np.ndarray]:
    accuracy = float(accuracy_score(y_true, y_pred))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return accuracy, recall_macro, f1_macro, cm


def predict(model: nn.Module, loader: DataLoader[tuple[Tensor, int]], device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    return y_true, y_pred


def train_single_model(
    model_name: str,
    class_names: list[str],
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> ModelRunResult:
    model, input_size = build_model(model_name, len(class_names), use_pretrained=not args.no_pretrained)
    train_tf, eval_tf = build_transforms(input_size)

    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model_name == "resnet50":
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        elif model_name == "efficientnet_b3":
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        elif model_name == "vit_b_16":
            for parameter in model.heads.parameters():
                parameter.requires_grad = True

    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    train_dataset = PlantDataset(train_samples, class_to_index=class_to_index, transform=train_tf)
    val_dataset = PlantDataset(val_samples, class_to_index=class_to_index, transform=eval_tf)
    test_dataset = PlantDataset(test_samples, class_to_index=class_to_index, transform=eval_tf)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = max(1, args.grad_accum_steps)

    best_val_f1 = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch + 1}/{args.epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        y_val_true, y_val_pred = predict(model, val_loader, device)
        _, _, val_f1, _ = evaluate_predictions(y_val_true, y_val_pred, num_classes=len(class_names))
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())

        print(f"{model_name} epoch {epoch + 1}/{args.epochs} val_f1_macro={val_f1:.4f}")

    model.load_state_dict(best_state_dict)

    y_test_true, y_test_pred = predict(model, test_loader, device)
    accuracy, recall_macro, f1_macro, cm = evaluate_predictions(y_test_true, y_test_pred, num_classes=len(class_names))

    model_out_dir = output_dir / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_out_dir / f"{model_name}.pth"
    torch.save(
        {
            "model_name": model_name,
            "class_names": class_names,
            "input_size": input_size,
            "state_dict": model.state_dict(),
            "metrics": {
                "accuracy": accuracy,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            },
        },
        checkpoint_path,
    )

    cm_path = model_out_dir / "confusion_matrix.csv"
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    print(f"{model_name} TEST accuracy={accuracy:.4f} recall_macro={recall_macro:.4f} f1_macro={f1_macro:.4f}")
    print(f"{model_name} confusion_matrix:\n{cm}")

    return ModelRunResult(
        model_name=model_name,
        accuracy=accuracy,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        confusion_matrix_path=str(cm_path),
        checkpoint_path=str(checkpoint_path),
    )


def export_best_model(
    best_result: ModelRunResult,
    class_names: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(best_result.checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    input_size = int(checkpoint["input_size"])
    model, _ = build_model(model_name, len(class_names), use_pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    best_pth_path = output_dir / "best_model.pth"
    shutil.copyfile(best_result.checkpoint_path, best_pth_path)

    if not args.no_onnx:
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
        best_onnx_path = output_dir / "best_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            best_onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )

    best_meta_path = output_dir / "best_model_metadata.json"
    best_meta_path.write_text(
        json.dumps(
            {
                "best_model_name": best_result.model_name,
                "metrics": {
                    "accuracy": best_result.accuracy,
                    "recall_macro": best_result.recall_macro,
                    "f1_macro": best_result.f1_macro,
                },
                "class_names": class_names,
                "best_checkpoint": str(best_pth_path),
                "onnx_exported": not args.no_onnx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_filtered_samples(
        dataset_root=dataset_root,
        target_classes=TARGET_CLASSES,
        allow_partial_classes=args.allow_partial_classes,
    )
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_samples = maybe_cap_samples(train_samples, args.max_train_samples, args.seed)
    val_samples = maybe_cap_samples(val_samples, args.max_val_samples, args.seed + 1)
    test_samples = maybe_cap_samples(test_samples, args.max_test_samples, args.seed + 2)

    class_names = sorted({sample.class_name for sample in samples})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset root: {dataset_root}")
    print(f"Total filtered images: {len(samples)}")
    print(f"Train images: {len(train_samples)} | Val images: {len(val_samples)} | Test images: {len(test_samples)}")
    print(f"Device: {device}")

    results: list[ModelRunResult] = []
    for model_name in args.models:
        result = train_single_model(
            model_name=model_name,
            class_names=class_names,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        results.append(result)

    best_result = max(results, key=lambda item: item.f1_macro)

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")

    export_best_model(best_result=best_result, class_names=class_names, args=args, output_dir=output_dir, device=device)

    print("Model comparison finished.")
    for item in results:
        print(
            f"{item.model_name}: accuracy={item.accuracy:.4f} recall_macro={item.recall_macro:.4f} f1_macro={item.f1_macro:.4f}"
        )
    print(f"Best model by F1: {best_result.model_name} (F1={best_result.f1_macro:.4f})")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
