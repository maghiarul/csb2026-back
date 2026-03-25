from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.plant_dataset import list_plant_class_names
from app.services.supabase import get_supabase_service_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace plants in database from dataset folders")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("models") / "medicinal plants",
        help="Path to medicinal plant folders",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be imported")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    return parser.parse_args()


def batched(rows: list[dict[str, str | None]], size: int) -> list[list[dict[str, str | None]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def main() -> None:
    args = parse_args()
    class_names = list_plant_class_names(args.dataset_root)

    if not class_names:
        raise SystemExit(f"No plants with images were found in {args.dataset_root}")

    rows = [
        {
            "name_ro": plant_name,
            "name_latin": plant_name,
            "usable_parts": None,
            "health_benefits": None,
            "contraindications": None,
            "description": None,
            "image_url": None,
        }
        for plant_name in class_names
    ]

    if args.dry_run:
        print(f"Would import {len(rows)} plants")
        for row in rows:
            print(f"- {row['name_ro']}")
        return

    service_client = get_supabase_service_client()

    service_client.table("poi_images").delete().gte("id", 0).execute()
    service_client.table("points_of_interest").delete().gte("id", 0).execute()
    service_client.table("plants").delete().gte("id", 0).execute()

    for chunk in batched(rows, args.batch_size):
        service_client.table("plants").insert(chunk).execute()

    print(f"Imported {len(rows)} plants into database")


if __name__ == "__main__":
    main()
