from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.plant_dataset import list_plant_class_names
from app.services.supabase import get_supabase_service_client


CURATED_PLANTS: list[dict[str, str | None]] = [
    {
        "name_ro": "Galbenele",
        "name_latin": "Calendula officinalis",
        "usable_parts": "Flori",
        "health_benefits": "Cicatrizant, antiinflamator.",
        "contraindications": "Alergii la compozite.",
        "description": "Planta anuala cu tulpina ierboasa de 20-40 cm.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Calendula+officinalis",
    },
    {
        "name_ro": "Sunatoare",
        "name_latin": "Hypericum perforatum",
        "usable_parts": "Parti aeriene",
        "health_benefits": "Antidepresiv usor, cicatrizant.",
        "contraindications": "Fotosensibilitate severa.",
        "description": "Planta cu flori galbene, frunzele prezinta puncte translucide.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Hypericum+perforatum",
    },
    {
        "name_ro": "Usturoita",
        "name_latin": "Alliaria petiolata",
        "usable_parts": "Frunze",
        "health_benefits": "Antiseptic, diuretic si utila in tratarea ranilor usoare.",
        "contraindications": "Iritatii gastrice in cantitati mari.",
        "description": "Planta erbacee, frunzele strivite au un miros puternic de usturoi.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Alliaria+petiolata",
    },
    {
        "name_ro": "Sugel",
        "name_latin": "Lamium amplexicaule",
        "usable_parts": "Parti aeriene",
        "health_benefits": "Antireumatic usor, laxativ (uz etnobotanic).",
        "contraindications": "Nu este recomandata intern fara aviz de specialitate.",
        "description": "Buruiana mica, flori purpurii tubulare, frunze rotunjite care imbratiseaza tulpina.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Lamium+amplexicaule",
    },
    {
        "name_ro": "Urzica moarta alba",
        "name_latin": "Lamium album",
        "usable_parts": "Flori, frunze",
        "health_benefits": "Astringent, hemostatic, antiinflamator pelvic.",
        "contraindications": "Fara contraindicatii majore la doze normale.",
        "description": "Asemanatoare urzicii, dar nu inteapa si are flori albe proeminente.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Lamium+album",
    },
    {
        "name_ro": "Trepadatoare anuala",
        "name_latin": "Mercurialis annua",
        "usable_parts": "Nu se recomanda uzul casnic",
        "health_benefits": "Traditional i se atribuiau proprietati de purgativ, dar utilizarea nu este aprobata astazi.",
        "contraindications": "Planta otravitoare. Provoaca inflamatii gastrointestinale severe.",
        "description": "Planta dioica cu flori verzui-galbui si miros fetid.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Mercurialis+annua",
    },
    {
        "name_ro": "Buruiana de cartite",
        "name_latin": "Mercurialis perennis",
        "usable_parts": "Strict extern (si foarte rar)",
        "health_benefits": "Niciunul demonstrat clinic sigur.",
        "contraindications": "Foarte toxica la uz intern.",
        "description": "Planta de padure, creste din rizomi, frunze aspre, flori verzi discrete.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Mercurialis+perennis",
    },
    {
        "name_ro": "Trifoi rosu",
        "name_latin": "Trifolium pratense",
        "usable_parts": "Flori",
        "health_benefits": "Sursa de fitoestrogeni, amelioreaza simptomele menopauzei, antitusiv.",
        "contraindications": "Afectiuni hormonodependente, sarcina.",
        "description": "Inflorescente globuloase roz-rosiatice, frunze trifoliate cu un marcaj alb deschis pe ele.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Trifolium+pratense",
    },
    {
        "name_ro": "Trifoi alb",
        "name_latin": "Trifolium repens",
        "usable_parts": "Flori",
        "health_benefits": "Expectorant, antiinflamator ocular (spalaturi externe).",
        "contraindications": "Posibile reactii alergice rare.",
        "description": "Tulpini taratoare, capitule florale albe, extrem de comun in peluze.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Trifolium+repens",
    },
    {
        "name_ro": "Trifoi frag",
        "name_latin": "Trifolium fragiferum",
        "usable_parts": "Parti aeriene",
        "health_benefits": "Astringent usor, valoare medicinala secundara.",
        "contraindications": "Precautie similara restului speciilor de Trifolium.",
        "description": "Florile se umfla dupa fructificare semanand la forma cu o mica fraguta.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Trifolium+fragiferum",
    },
    {
        "name_ro": "Laptuca salbatica",
        "name_latin": "Lactuca serriola",
        "usable_parts": "Latex, frunze",
        "health_benefits": "Sedativ slab, analgezic si calmant pentru tuse.",
        "contraindications": "Supradozajul da stari de ameteala si somnolenta intensa.",
        "description": "Planta cu frunze spinos-dintate, secreta un latex alb laptos la rupere.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Lactuca+serriola",
    },
    {
        "name_ro": "Susai paduret",
        "name_latin": "Lactuca muralis",
        "usable_parts": "Frunze",
        "health_benefits": "Efect sedativ foarte slab.",
        "contraindications": "Toxicitate usoara la doze mari, nerecomandat uzului curent.",
        "description": "Frunze subtiri, crestate, se gaseste frecvent pe ziduri umbrite.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Lactuca+muralis",
    },
    {
        "name_ro": "Mac de camp",
        "name_latin": "Papaver rhoeas",
        "usable_parts": "Petale",
        "health_benefits": "Emolient, antitusiv, usor sedativ.",
        "contraindications": "Nu se administreaza copiilor sau in sarcina.",
        "description": "Flori mari, rosii, patate cu negru la baza.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Papaver+rhoeas",
    },
    {
        "name_ro": "Mac dubios",
        "name_latin": "Papaver dubium",
        "usable_parts": "Nu are utilitate practica",
        "health_benefits": "Potential usor sedativ, dar nereglementat.",
        "contraindications": "Toxic din cauza alcaloizilor care induc greata.",
        "description": "Mac cu flori de un rosu mai palid si capsula mai alungita decat macul de camp.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Papaver+dubium",
    },
    {
        "name_ro": "Mac aspru",
        "name_latin": "Papaver argemone",
        "usable_parts": "Nu se utilizeaza",
        "health_benefits": "Niciun beneficiu dovedit.",
        "contraindications": "Toxic, nu se utilizeaza intern.",
        "description": "Capsula prezinta perisori aspri, petale cu pete negre mici.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Papaver+argemone",
    },
    {
        "name_ro": "Palamida",
        "name_latin": "Cirsium arvense",
        "usable_parts": "Radacina",
        "health_benefits": "Diuretic si detoxifiant hepatic slab.",
        "contraindications": "Evitat intern fara preparare adecvata.",
        "description": "Buruiana invaziva, cu tepi pe frunze si flori mov-liliachii.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Cirsium+arvense",
    },
    {
        "name_ro": "Scaiete",
        "name_latin": "Cirsium vulgare",
        "usable_parts": "Radacina",
        "health_benefits": "Diuretic si antiinflamator.",
        "contraindications": "Nicio contraindicatie extrema, insa necesita prudenta.",
        "description": "Planta masiva, foarte spinoasa, cu o floare mov mare, solitara pe tija.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Cirsium+vulgare",
    },
    {
        "name_ro": "Feriga de padure",
        "name_latin": "Dryopteris filix-mas",
        "usable_parts": "Rizom",
        "health_benefits": "Antielmintic (eficient contra teniei).",
        "contraindications": "Extrem de toxica la supradozaj, ataca sistemul nervos central si vederea.",
        "description": "Feriga mare, comuna la umbra, rizomul are un miros specific greoi.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Dryopteris+filix-mas",
    },
    {
        "name_ro": "Feriga regala",
        "name_latin": "Osmunda regalis",
        "usable_parts": "Rizom",
        "health_benefits": "Cicatrizant si astringent pentru rani externe.",
        "contraindications": "Interzis uzul intern.",
        "description": "Feriga de umezeala toleranta la sare, cu spori grupati pe frunze modificate la varf.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Osmunda+regalis",
    },
    {
        "name_ro": "Morcov salbatic",
        "name_latin": "Daucus carota",
        "usable_parts": "Seminte, radacina",
        "health_benefits": "Carminativ, diuretic si stimulent biliar.",
        "contraindications": "Semintele sunt contraindicate in sarcina.",
        "description": "Inflorescenta alba de tip umbela, uneori cu o singura floare rosie in centru.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Daucus+carota",
    },
    {
        "name_ro": "Piciorul caprei",
        "name_latin": "Aegopodium podagraria",
        "usable_parts": "Parti aeriene (frunze tinere)",
        "health_benefits": "Diuretic excelent, folosit traditional impotriva gutei.",
        "contraindications": "Nu are toxicitate raportata.",
        "description": "Frunze divizate, face umbele florale albe.",
        "image_url": "https://placehold.co/600x400/2E8B57/FFFFFF?text=Aegopodium+podagraria",
    },
]


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
    parser.add_argument(
        "--source",
        choices=["dataset", "curated"],
        default="curated",
        help="Data source for plants table",
    )
    return parser.parse_args()


def batched(rows: list[dict[str, str | None]], size: int) -> list[list[dict[str, str | None]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def main() -> None:
    args = parse_args()
    if args.source == "curated":
        rows = CURATED_PLANTS
    else:
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
