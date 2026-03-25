from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    supabase_jwt_secret: str
    ml_model_path: str = "models/plant_identifier.pkl"
    ml_min_confidence: float = 0.6
    debug: bool = False
    signed_url_expires_seconds: int = 3600
    max_image_size_mb: int = 10

    @computed_field(return_type=int)
    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
