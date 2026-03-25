from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from supabase import Client

from app.dependencies import CurrentUser, get_current_user
from app.schemas.plant import PlantSelection
from app.schemas.poi import POICreateRequest, POIDetail, POIImageOut, POIListItem
from app.services.storage import generate_signed_image_url, upload_poi_image
from app.services.supabase import get_supabase_service_client


router = APIRouter(prefix="/poi", tags=["poi"])


def _extract_error_message(exc: Exception, fallback: str) -> str:
    message = str(exc).strip()
    return message if message else fallback


@router.get("", response_model=list[POIListItem])
def list_poi(
    plant_id: int | None = Query(default=None),
    lat: float | None = Query(default=None, ge=-90, le=90),
    lng: float | None = Query(default=None, ge=-180, le=180),
    radius_km: float | None = Query(default=None, gt=0, le=50),
    service_client: Client = Depends(get_supabase_service_client),
) -> list[POIListItem]:
    spatial_values = [lat, lng, radius_km]
    if any(value is not None for value in spatial_values) and not all(value is not None for value in spatial_values):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="lat, lng and radius_km must be provided together",
        )

    try:
        response = service_client.rpc(
            "get_approved_poi",
            {
                "p_plant_id": plant_id,
                "p_lat": lat,
                "p_lng": lng,
                "p_radius_km": radius_km,
            },
        ).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch POIs: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    result: list[POIListItem] = []
    for row in response.data or []:
        image_url = None
        if row.get("image_path"):
            image_url = generate_signed_image_url(row["image_path"], service_client)

        result.append(
            POIListItem(
                id=row["id"],
                user_id=row["user_id"],
                plant_id=row["plant_id"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                comment=row.get("comment"),
                created_at=row["created_at"],
                distance_km=row.get("distance_km"),
                image_url=image_url,
            )
        )

    return result


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_poi(
    plant_id: int = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    comment: str | None = Form(default=None),
    image: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
    service_client: Client = Depends(get_supabase_service_client),
) -> dict[str, object]:
    payload = POICreateRequest(
        plant_id=plant_id,
        latitude=latitude,
        longitude=longitude,
        comment=comment,
    )

    storage_path, signed_url = await upload_poi_image(image, current_user.id, service_client)

    try:
        response = service_client.rpc(
            "create_poi_with_image",
            {
                "p_user_id": current_user.id,
                "p_plant_id": payload.plant_id,
                "p_lat": payload.latitude,
                "p_lng": payload.longitude,
                "p_comment": payload.comment,
                "p_image_url": storage_path,
            },
        ).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to create POI entry: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create POI")

    row = rows[0]
    return {
        "id": row["poi_id"],
        "user_id": row["user_id"],
        "plant_id": row["plant_id"],
        "comment": row.get("comment"),
        "created_at": row["created_at"],
        "image": {
            "id": row["image_id"],
            "status": row["image_status"],
            "image_url": signed_url,
        },
    }


@router.get("/{poi_id}", response_model=POIDetail)
def get_poi_detail(poi_id: int, service_client: Client = Depends(get_supabase_service_client)) -> POIDetail:
    try:
        poi_response = (
            service_client.table("points_of_interest")
            .select("id, user_id, plant_id, comment, created_at")
            .eq("id", poi_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch POI detail: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    detail_rows = poi_response.data or []

    if not detail_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="POI not found")

    detail = detail_rows[0]

    try:
        plant_response = (
            service_client.table("plants")
            .select("id, name_ro, name_latin")
            .eq("id", detail["plant_id"])
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load plant details: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    plant_rows = plant_response.data or []
    if not plant_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found for this POI")

    try:
        coords_response = service_client.rpc("get_poi_detail", {"p_poi_id": poi_id}).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load POI coordinates: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    coords_rows = coords_response.data or []
    if not coords_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="POI coordinates not found")

    coords = coords_rows[0]

    try:
        images_response = (
            service_client.table("poi_images")
            .select("id, poi_id, user_id, image_url, status, created_at")
            .eq("poi_id", poi_id)
            .eq("status", "approved")
            .order("created_at")
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch POI images: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    images: list[POIImageOut] = []
    for image in images_response.data or []:
        images.append(
            POIImageOut(
                id=image["id"],
                poi_id=image["poi_id"],
                user_id=image["user_id"],
                image_url=generate_signed_image_url(image["image_url"], service_client),
                status=image["status"],
                created_at=image["created_at"],
            )
        )

    return POIDetail(
        id=detail["id"],
        user_id=detail["user_id"],
        plant_id=detail["plant_id"],
        latitude=coords["latitude"],
        longitude=coords["longitude"],
        comment=detail.get("comment"),
        created_at=detail["created_at"],
        plant=PlantSelection(
            id=plant_rows[0]["id"],
            name_ro=plant_rows[0]["name_ro"],
            name_latin=plant_rows[0]["name_latin"],
        ),
        images=images,
    )
