import io
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


class LocalPlantIdentifier:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))

        try:
            with self.model_path.open("rb") as model_file:
                model = pickle.load(model_file)
        except Exception as exc:
            raise ValueError(f"Failed to load model: {exc}") from exc

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must implement predict_proba")

        return model

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

    def predict(self, image_bytes: bytes) -> dict[str, int | float | str | None]:
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


@lru_cache
def get_local_identifier() -> LocalPlantIdentifier:
    settings = get_settings()
    return LocalPlantIdentifier(settings.ml_model_path)
