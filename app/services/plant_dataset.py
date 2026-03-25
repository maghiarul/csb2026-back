from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_plant_name(raw_name: str) -> str:
    return " ".join(raw_name.strip().split())


def list_plant_class_names(dataset_root: Path) -> list[str]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []

    class_names: list[str] = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue

        has_images = any(
            file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
            for file_path in folder.rglob("*")
        )
        if has_images:
            class_names.append(normalize_plant_name(folder.name))

    return class_names


def collect_image_samples(dataset_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue

        class_name = normalize_plant_name(folder.name)
        for image_path in sorted(folder.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_name))

    return samples
