import io
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


class LocalPlantIdentifier:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.backend = ""
        self.model: Any = None
        self.transform: Any = None
        self.class_names: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))

        if self.model_path.suffix.lower() in {".pt", ".pth"}:
            self._load_torch_checkpoint()
            return

        self._load_pickle_model()

    def _load_pickle_model(self) -> None:
        try:
            with self.model_path.open("rb") as model_file:
                model = pickle.load(model_file)
        except Exception as exc:
            raise ValueError(f"Failed to load model: {exc}") from exc

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must implement predict_proba")

        self.backend = "pickle"
        self.model = model

    def _load_torch_checkpoint(self) -> None:
        try:
            import torch
            from torchvision import transforms
            from torchvision.models import efficientnet_b3, resnet50, vit_b_16
        except Exception as exc:
            raise ValueError(f"Missing deep learning dependencies: {exc}") from exc

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
        except Exception as exc:
            raise ValueError(f"Failed to load model: {exc}") from exc

        model_name = checkpoint.get("model_name")
        class_names = checkpoint.get("class_names")
        input_size = int(checkpoint.get("input_size", 224))
        state_dict = checkpoint.get("state_dict")

        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Torch checkpoint is missing model_name")

        if not isinstance(class_names, list) or not class_names:
            raise ValueError("Torch checkpoint is missing class_names")

        num_classes = len(class_names)

        if model_name == "resnet50":
            model = resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "efficientnet_b3":
            model = efficientnet_b3(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "vit_b_16":
            model = vit_b_16(weights=None)
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

        if not isinstance(state_dict, dict):
            raise ValueError("Torch checkpoint is missing state_dict")

        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise ValueError(f"Failed to load state_dict: {exc}") from exc

        model.eval()

        self.backend = "torch"
        self.model = model
        self.class_names = [str(item) for item in class_names]
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.15)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _extract_features(self, image_bytes: bytes) -> list[float]:
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:
            raise ValueError(f"Missing local ML dependencies: {exc}") from exc

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
        except Exception as exc:
            raise ValueError(f"Invalid image input: {exc}") from exc

        image_array = np.asarray(image, dtype=np.float32)
        feature_parts: list[np.ndarray] = []
        for channel in range(3):
            histogram, _ = np.histogram(image_array[:, :, channel], bins=16, range=(0, 255), density=True)
            feature_parts.append(histogram.astype(np.float32))

        features = np.concatenate(feature_parts, axis=0)
        return features.tolist()

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if self.backend == "torch":
            return self._predict_torch(image_bytes)

        try:
            import numpy as np
        except Exception as exc:
            raise ValueError(f"Missing local ML dependencies: {exc}") from exc

        features = self._extract_features(image_bytes)

        try:
            probabilities = self.model.predict_proba([features])[0]
        except Exception as exc:
            raise ValueError(f"Model inference failed: {exc}") from exc

        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        predicted_class = None
        model_classes = getattr(self.model, "classes_", None)
        if model_classes is not None and len(model_classes) > best_index:
            predicted_class = model_classes[best_index]

        plant_id: int | None = None
        plant_name: str | None = None

        if isinstance(predicted_class, (int, np.integer)):
            plant_id = int(predicted_class)
        elif isinstance(predicted_class, str):
            plant_name = predicted_class.strip() or None

        return {
            "plant_id": plant_id,
            "plant_name": plant_name,
            "confidence": confidence,
        }

    def _predict_torch(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
        except Exception as exc:
            raise ValueError(f"Missing deep learning dependencies: {exc}") from exc

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image input: {exc}") from exc

        if self.transform is None:
            raise ValueError("Torch preprocessing transform is not available")

        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        topk = min(5, probabilities.shape[0])
        top_values, top_indices = torch.topk(probabilities, k=topk)

        best_index = int(top_indices[0].item())
        confidence = float(top_values[0].item())
        plant_name = None

        if 0 <= best_index < len(self.class_names):
            plant_name = self.class_names[best_index]

        candidates: list[dict[str, float | str | None]] = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            candidate_name = None
            if 0 <= int(index) < len(self.class_names):
                candidate_name = self.class_names[int(index)]
            candidates.append(
                {
                    "plant_name": candidate_name,
                    "confidence": float(value),
                }
            )

        return {
            "plant_id": None,
            "plant_name": plant_name,
            "confidence": confidence,
            "candidates": candidates,
        }


@lru_cache
def get_local_identifier() -> LocalPlantIdentifier:
    settings = get_settings()
    return LocalPlantIdentifier(settings.ml_model_path)
