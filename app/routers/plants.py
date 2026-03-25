from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.schemas.plant import PlantDetail, PlantListItem
from app.services.supabase import get_supabase_service_client


router = APIRouter(prefix="/plants", tags=["plants"])


@router.get("", response_model=list[PlantListItem])
def list_plants(service_client: Client = Depends(get_supabase_service_client)) -> list[PlantListItem]:
    response = service_client.table("plants").select("id, name_ro, name_latin, image_url").order("name_ro").execute()
    return response.data or []


@router.get("/{plant_id}", response_model=PlantDetail)
def get_plant(plant_id: int, service_client: Client = Depends(get_supabase_service_client)) -> PlantDetail:
    response = (
        service_client.table("plants")
        .select(
            "id, name_ro, name_latin, usable_parts, health_benefits, contraindications, description, image_url, created_at"
        )
        .eq("id", plant_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found")

    return PlantDetail.model_validate(rows[0])
