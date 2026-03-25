import difflib
import re
import unicodedata

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from supabase import Client

from app.config import Settings, get_settings
from app.schemas.plant import PlantSelection
from app.services.local_identifier import get_local_identifier
from app.services.supabase import get_supabase_service_client


class IdentifyResponse(BaseModel):
    plant_id: int | None
    plant_name: str | None
    confidence: float
    top_candidates: list[PlantSelection] | None = None
    fallback_plants: list[PlantSelection] | None = None


router = APIRouter(prefix="/identify", tags=["identify"])


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = ascii_only.lower()
    lowered = re.sub(r"[-_/]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _name_keys(value: str) -> set[str]:
    normalized = _normalize_name(value)
    condensed = normalized.replace(" ", "")
    keys = {normalized}
    if condensed:
        keys.add(condensed)
    return keys


def _resolve_plant_from_name(predicted_name: str, service_client: Client) -> tuple[int | None, str | None]:
    by_ro = (
        service_client.table("plants")
        .select("id, name_ro")
        .eq("name_ro", predicted_name)
        .limit(1)
        .execute()
        .data
        or []
    )
    if by_ro:
        return by_ro[0]["id"], by_ro[0]["name_ro"]

    by_latin = (
        service_client.table("plants")
        .select("id, name_ro")
        .eq("name_latin", predicted_name)
        .limit(1)
        .execute()
        .data
        or []
    )
    if by_latin:
        return by_latin[0]["id"], by_latin[0]["name_ro"]

    all_plants = (
        service_client.table("plants")
        .select("id, name_ro, name_latin")
        .execute()
        .data
        or []
    )

    predicted_keys = _name_keys(predicted_name)
    normalized_to_plant: dict[str, tuple[int, str]] = {}
    for row in all_plants:
        plant_id = int(row["id"])
        name_ro = str(row["name_ro"])
        name_latin = str(row["name_latin"])
        for key in _name_keys(name_ro) | _name_keys(name_latin):
            normalized_to_plant.setdefault(key, (plant_id, name_ro))

    for key in predicted_keys:
        candidate = normalized_to_plant.get(key)
        if candidate is not None:
            return candidate

    all_candidate_keys = list(normalized_to_plant.keys())
    best_match: str | None = None
    for key in predicted_keys:
        close = difflib.get_close_matches(key, all_candidate_keys, n=1, cutoff=0.86)
        if close:
            best_match = close[0]
            break

    if best_match is None:
        return None, predicted_name

    matched_id, matched_name = normalized_to_plant[best_match]
    return matched_id, matched_name


@router.post("", response_model=IdentifyResponse)
async def identify_plant(
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service_client: Client = Depends(get_supabase_service_client),
) -> IdentifyResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file")

    try:
        identifier = get_local_identifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local model file not found. Train and export model first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local model is invalid: {exc}",
        ) from exc

    try:
        payload = identifier.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local model is invalid: {exc}",
        ) from exc

    raw_plant_id = payload.get("plant_id")
    plant_id = int(raw_plant_id) if raw_plant_id is not None else None
    plant_name = payload.get("plant_name")
    confidence = float(payload.get("confidence", 0.0))
    raw_candidates = payload.get("candidates")

    if plant_name is None and plant_id is not None:
        plant_rows = service_client.table("plants").select("name_ro").eq("id", plant_id).limit(1).execute().data or []
        if plant_rows:
            plant_name = plant_rows[0]["name_ro"]

    if plant_name is not None and plant_id is None:
        plant_id, plant_name = _resolve_plant_from_name(plant_name, service_client)

    top_candidates: list[PlantSelection] = []
    if isinstance(raw_candidates, list):
        seen_plant_ids: set[int] = set()
        if plant_id is not None:
            seen_plant_ids.add(plant_id)

        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            candidate_name = item.get("plant_name")
            if not isinstance(candidate_name, str) or not candidate_name.strip():
                continue
            candidate_id, _ = _resolve_plant_from_name(candidate_name, service_client)
            if candidate_id is None:
                continue
            if candidate_id in seen_plant_ids:
                continue

            candidate_rows = (
                service_client.table("plants")
                .select("id, name_ro, name_latin")
                .eq("id", candidate_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            if not candidate_rows:
                continue

            top_candidates.append(PlantSelection.model_validate(candidate_rows[0]))
            seen_plant_ids.add(candidate_id)

            if len(top_candidates) >= 5:
                break

    fallback_plants: list[PlantSelection] | None = None
    if confidence < settings.ml_min_confidence:
        fallback_rows = (
            service_client.table("plants")
            .select("id, name_ro, name_latin")
            .order("name_ro")
            .limit(10)
            .execute()
            .data
            or []
        )
        fallback_plants = [PlantSelection.model_validate(row) for row in fallback_rows]

    return IdentifyResponse(
        plant_id=plant_id,
        plant_name=plant_name,
        confidence=confidence,
        top_candidates=top_candidates or None,
        fallback_plants=fallback_plants,
    )
