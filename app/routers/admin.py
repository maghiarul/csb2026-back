from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.dependencies import CurrentUser, require_admin
from app.schemas.poi import ModerationUpdateRequest, POIImageOut
from app.schemas.user import UpdateUserRoleRequest, UserOut
from app.services.storage import generate_signed_image_url
from app.services.supabase import get_supabase_service_client


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserOut])
def list_users(
    _: CurrentUser = Depends(require_admin),
    service_client: Client = Depends(get_supabase_service_client),
) -> list[UserOut]:
    response = service_client.table("profiles").select("id, role, username, created_at").order("created_at", desc=True).execute()
    return [UserOut.model_validate({**row, "email": None}) for row in (response.data or [])]


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user_role(
    user_id: str,
    payload: UpdateUserRoleRequest,
    _: CurrentUser = Depends(require_admin),
    service_client: Client = Depends(get_supabase_service_client),
) -> UserOut:
    (
        service_client.table("profiles")
        .update({"role": payload.role})
        .eq("id", user_id)
        .execute()
    )

    response = (
        service_client.table("profiles")
        .select("id, role, username, created_at")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserOut.model_validate({**rows[0], "email": None})


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    _: CurrentUser = Depends(require_admin),
    service_client: Client = Depends(get_supabase_service_client),
) -> dict[str, object]:
    service_client.auth.admin.delete_user(user_id)
    return {"deleted": True, "user_id": user_id}


@router.get("/images", response_model=list[POIImageOut])
def list_pending_images(
    _: CurrentUser = Depends(require_admin),
    service_client: Client = Depends(get_supabase_service_client),
) -> list[POIImageOut]:
    response = (
        service_client.table("poi_images")
        .select("id, poi_id, user_id, image_url, status, created_at")
        .eq("status", "pending")
        .order("created_at")
        .execute()
    )

    images: list[POIImageOut] = []
    for row in response.data or []:
        images.append(
            POIImageOut(
                id=row["id"],
                poi_id=row["poi_id"],
                user_id=row["user_id"],
                image_url=generate_signed_image_url(row["image_url"], service_client),
                status=row["status"],
                created_at=row["created_at"],
            )
        )

    return images


@router.patch("/images/{image_id}", response_model=POIImageOut)
def moderate_image(
    image_id: int,
    payload: ModerationUpdateRequest,
    _: CurrentUser = Depends(require_admin),
    service_client: Client = Depends(get_supabase_service_client),
) -> POIImageOut:
    (
        service_client.table("poi_images")
        .update({"status": payload.status})
        .eq("id", image_id)
        .execute()
    )

    response = (
        service_client.table("poi_images")
        .select("id, poi_id, user_id, image_url, status, created_at")
        .eq("id", image_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    row = rows[0]
    return POIImageOut(
        id=row["id"],
        poi_id=row["poi_id"],
        user_id=row["user_id"],
        image_url=generate_signed_image_url(row["image_url"], service_client),
        status=row["status"],
        created_at=row["created_at"],
    )
