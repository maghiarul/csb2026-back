from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.plant import PlantSelection


class POICreateRequest(BaseModel):
    plant_id: int
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    comment: str | None = Field(default=None, max_length=1000)


class POIImageOut(BaseModel):
    id: int
    poi_id: int
    user_id: str
    image_url: str
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime


class POIListItem(BaseModel):
    id: int
    user_id: str
    plant_id: int
    latitude: float
    longitude: float
    comment: str | None
    created_at: datetime
    distance_km: float | None = None
    image_url: str | None = None


class POIDetail(BaseModel):
    id: int
    user_id: str
    plant_id: int
    latitude: float
    longitude: float
    comment: str | None
    created_at: datetime
    plant: PlantSelection
    images: list[POIImageOut]


class ModerationUpdateRequest(BaseModel):
    status: Literal["approved", "rejected"]
