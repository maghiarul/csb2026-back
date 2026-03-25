from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.schemas.user import LoginRequest, LoginResponse, RegisterRequest, UserOut
from app.services.supabase import get_supabase_client, get_supabase_service_client


router = APIRouter(prefix="/auth", tags=["auth"])


def _extract_error_message(exc: Exception, fallback: str) -> str:
    message = str(exc).strip()
    return message if message else fallback


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    service_client: Client = Depends(get_supabase_service_client),
) -> UserOut:
    try:
        existing = service_client.table("profiles").select("id").eq("username", payload.username).limit(1).execute().data or []
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to check username availability: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already in use")

    try:
        auth_response = service_client.auth.admin.create_user(
            {
                "email": payload.email,
                "password": payload.password,
                "email_confirm": True,
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User registration failed: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    created_user = getattr(auth_response, "user", None)
    if created_user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User registration failed")

    try:
        profile_response = (
            service_client.table("profiles")
            .insert(
                {
                    "id": created_user.id,
                    "role": "user",
                    "username": payload.username,
                }
            )
            .execute()
        )
    except Exception as exc:
        service_client.auth.admin.delete_user(created_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile creation failed") from exc

    profile_data = (profile_response.data or [None])[0]
    if profile_data is None:
        service_client.auth.admin.delete_user(created_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile creation failed")

    return UserOut(
        id=profile_data["id"],
        role=profile_data["role"],
        username=profile_data["username"],
        created_at=profile_data["created_at"],
        email=payload.email,
    )


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    client: Client = Depends(get_supabase_client),
    service_client: Client = Depends(get_supabase_service_client),
) -> LoginResponse:
    try:
        auth_response = client.auth.sign_in_with_password(
            {
                "email": payload.email,
                "password": payload.password,
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid email or password: {_extract_error_message(exc, 'Authentication failed')}",
        ) from exc

    session = getattr(auth_response, "session", None)
    user = getattr(auth_response, "user", None)

    if session is None or user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    try:
        profile_rows = service_client.table("profiles").select("id, role, username, created_at").eq("id", user.id).limit(1).execute().data or []
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load user profile: {_extract_error_message(exc, 'Unknown error')}",
        ) from exc

    if not profile_rows:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Profile not found")

    profile = profile_rows[0]

    return LoginResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        user=UserOut(
            id=profile["id"],
            role=profile["role"],
            username=profile["username"],
            created_at=profile["created_at"],
            email=user.email,
        ),
    )
