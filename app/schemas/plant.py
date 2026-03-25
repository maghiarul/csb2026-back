from datetime import datetime

from pydantic import BaseModel


class PlantListItem(BaseModel):
    id: int
    name_ro: str
    name_latin: str
    image_url: str | None = None


class PlantDetail(BaseModel):
    id: int
    name_ro: str
    name_latin: str
    usable_parts: str | None = None
    health_benefits: str | None = None
    contraindications: str | None = None
    description: str | None = None
    image_url: str | None = None
    created_at: datetime


class PlantSelection(BaseModel):
    id: int
    name_ro: str
    name_latin: str
