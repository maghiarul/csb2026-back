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
    fallback_plants: list[PlantSelection] | None = None


router = APIRouter(prefix="/identify", tags=["identify"])


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

    if plant_name is None and plant_id is not None:
        plant_rows = service_client.table("plants").select("name_ro").eq("id", plant_id).limit(1).execute().data or []
        if plant_rows:
            plant_name = plant_rows[0]["name_ro"]

    if plant_name is not None and plant_id is None:
        by_ro = service_client.table("plants").select("id, name_ro").eq("name_ro", plant_name).limit(1).execute().data or []
        if by_ro:
            plant_id = by_ro[0]["id"]
            plant_name = by_ro[0]["name_ro"]
        else:
            by_latin = service_client.table("plants").select("id, name_ro").eq("name_latin", plant_name).limit(1).execute().data or []
            if by_latin:
                plant_id = by_latin[0]["id"]
                plant_name = by_latin[0]["name_ro"]

    fallback_plants: list[PlantSelection] | None = None
    if confidence < settings.ml_min_confidence:
        fallback_rows = service_client.table("plants").select("id, name_ro, name_latin").order("name_ro").execute().data or []
        fallback_plants = [PlantSelection.model_validate(row) for row in fallback_rows]

    return IdentifyResponse(
        plant_id=plant_id,
        plant_name=plant_name,
        confidence=confidence,
        fallback_plants=fallback_plants,
    )
