from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import HTTPException, UploadFile, status
from supabase import Client

from app.config import get_settings


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
POI_BUCKET = "poi-images"


def _extract_storage_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "message", "msg"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(payload)

    text = response.text.strip()
    if text:
        return text
    return f"HTTP {response.status_code}"


async def _ensure_bucket_exists(client: httpx.AsyncClient, settings) -> None:
    bucket_url = f"{settings.supabase_url}/storage/v1/bucket/{POI_BUCKET}"
    response = await client.get(
        bucket_url,
        headers={
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {settings.supabase_service_key}",
        },
    )
    if response.status_code == 404:
        create_response = await client.post(
            f"{settings.supabase_url}/storage/v1/bucket",
            json={
                "id": POI_BUCKET,
                "name": POI_BUCKET,
                "public": False,
                "file_size_limit": None,
                "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
            },
            headers={
                "apikey": settings.supabase_service_key,
                "Authorization": f"Bearer {settings.supabase_service_key}",
                "Content-Type": "application/json",
            },
        )
        if create_response.status_code < 400:
            return
        error_message = _extract_storage_error(create_response)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create storage bucket '{POI_BUCKET}': {error_message}",
        )

    if response.status_code >= 400:
        error_message = _extract_storage_error(response)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage bucket '{POI_BUCKET}' is not ready: {error_message}",
        )


def _normalize_signed_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    settings = get_settings()
    if url.startswith("/storage/"):
        return f"{settings.supabase_url}{url}"
    if url.startswith("/object/"):
        return f"{settings.supabase_url}/storage/v1{url}"
    return f"{settings.supabase_url}/{url.lstrip('/')}"


def build_storage_path(user_id: str, file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image format")
    return f"{user_id}/{uuid4().hex}{extension}"


async def upload_poi_image(file: UploadFile, user_id: str, service_client: Client) -> tuple[str, str]:
    settings = get_settings()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file")

    if len(file_bytes) > settings.max_image_size_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image file too large")

    storage_path = build_storage_path(user_id, file.filename or "upload.jpg")

    encoded_path = quote(storage_path, safe="/")
    upload_url = f"{settings.supabase_url}/storage/v1/object/{POI_BUCKET}/{encoded_path}"
    headers = {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": file.content_type,
        "x-upsert": "false",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        await _ensure_bucket_exists(client, settings)

        upload_response = await client.post(upload_url, content=file_bytes, headers=headers)
        if upload_response.status_code >= 400:
            error_message = _extract_storage_error(upload_response)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload image: {error_message}",
            )

        sign_url = f"{settings.supabase_url}/storage/v1/object/sign/{POI_BUCKET}/{encoded_path}"
        sign_response = await client.post(
            sign_url,
            json={"expiresIn": settings.signed_url_expires_seconds},
            headers={
                "apikey": settings.supabase_service_key,
                "Authorization": f"Bearer {settings.supabase_service_key}",
            },
        )
        if sign_response.status_code >= 400:
            error_message = _extract_storage_error(sign_response)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate signed image URL: {error_message}",
            )

    signed_payload = sign_response.json()
    signed_url = signed_payload.get("signedURL") or signed_payload.get("signedUrl") or signed_payload.get("signed_url")

    if not signed_url:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate signed image URL")

    return storage_path, _normalize_signed_url(signed_url)


def generate_signed_image_url(storage_path: str, service_client: Client) -> str:
    settings = get_settings()
    encoded_path = quote(storage_path, safe="/")
    sign_url = f"{settings.supabase_url}/storage/v1/object/sign/{POI_BUCKET}/{encoded_path}"
    response = httpx.post(
        sign_url,
        json={"expiresIn": settings.signed_url_expires_seconds},
        headers={
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {settings.supabase_service_key}",
        },
        timeout=30,
    )
    if response.status_code >= 400:
        error_message = _extract_storage_error(response)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate signed image URL: {error_message}",
        )
    signed_payload = response.json()
    signed_url = signed_payload.get("signedURL") or signed_payload.get("signedUrl") or signed_payload.get("signed_url")
    if not signed_url:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate signed image URL")
    return _normalize_signed_url(signed_url)
