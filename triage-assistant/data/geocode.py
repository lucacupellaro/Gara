"""Geocodifica i 49 PS del Lazio via Nominatim (1 req/s) -> data/hospitals.json
e normalizza le farmacie -> data/pharmacies.json. Idempotente: se hospitals.json
esiste, geocodifica solo le strutture mancanti."""
import csv
import json
import time
from pathlib import Path

import httpx

DATA = Path(__file__).parent
RAW = DATA / "raw"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "triage-assistant-hackathon/1.0 (progetto di gara, contatto: team)"}


def nominatim(query: str) -> tuple[float, float] | None:
    r = httpx.get(
        NOMINATIM,
        params={"q": query, "format": "json", "limit": 1, "countrycodes": "it"},
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    hits = r.json()
    time.sleep(1.1)  # rispetto del rate limit Nominatim
    if hits:
        return float(hits[0]["lat"]), float(hits[0]["lon"])
    return None


# Abbreviazioni dello snapshot che Nominatim non capisce
EXPANSIONS = [
    ("Pol. Univ.", "Policlinico Universitario"),
    ("Pol.", "Policlinico"),
    ("Osp.", "Ospedale"),
    ("S. ", "San "),
    ("-FBF", " Fatebenefratelli"),
    ("C.T.O.", "CTO Centro Traumatologico Ortopedico"),
]

# Coordinate curate a mano per strutture note che il geocoding sbaglia
OVERRIDES = {
    "Pol. Univ. A. Gemelli": (41.93261, 12.42921),
    "Pol. Univ. Umberto I": (41.90616, 12.50791),
    "Pol. Univ. Tor Vergata": (41.86153, 12.62462),
    "San Giovanni Calibita-FBF": (41.89042, 12.47752),
    "Madre G. Vannini": (41.87970, 12.53200),
    "C.T.O. Andrea Alesini": (41.85630, 12.48550),
    "Sant'Andrea": (42.00500, 12.46410),          # via di Grottarossa
    "San Filippo Neri": (41.94480, 12.43730),
    "Cristo Re": (41.93120, 12.43210),
    "Campus Biomedico": (41.79870, 12.50450),     # Trigoria
    "G. Battista Grassi": (41.73330, 12.28700),   # Ostia
    "Sandro Pertini": (41.91390, 12.55040),
}


def expand(name: str) -> str:
    for short, long in EXPANSIONS:
        name = name.replace(short, long)
    return name


def load_ps_rows() -> list[dict]:
    with open(RAW / "ps_lazio.csv", encoding="latin-1") as f:
        return list(csv.DictReader(f, delimiter=";"))


def geocode_hospitals() -> None:
    out_path = DATA / "hospitals.json"
    known: dict[str, dict] = {}
    if out_path.exists():
        known = {h["code"]: h for h in json.loads(out_path.read_text())}

    hospitals = []
    fallbacks = 0
    for row in load_ps_rows():
        code = row["CODICE"].strip()
        name, comune = row["ISTITUTO"].strip(), row["COMUNE"].strip()
        if name in OVERRIDES:
            lat, lon = OVERRIDES[name]
            hospitals.append({"code": code, "name": name, "type": row["TIPO"].strip(),
                              "comune": comune, "asl": row["ASL"].strip(),
                              "lat": lat, "lon": lon})
            print(f"[geocode] {name} -> override curato")
            continue
        if code in known and known[code].get("lat"):
            hospitals.append(known[code])
            continue
        coords = nominatim(f"{expand(name)}, {comune}, Lazio, Italia")
        if coords is None:
            coords = nominatim(f"ospedale {comune}, Lazio, Italia")
            fallbacks += 1
        if coords is None:  # ultimo fallback: centro del comune
            coords = nominatim(f"{comune}, Lazio, Italia")
        lat, lon = coords if coords else (None, None)
        hospitals.append(
            {"code": code, "name": name, "type": row["TIPO"].strip(),
             "comune": comune, "asl": row["ASL"].strip(), "lat": lat, "lon": lon}
        )
        print(f"[geocode] {name} ({comune}) -> {lat},{lon}")

    out_path.write_text(json.dumps(hospitals, ensure_ascii=False, indent=1))
    print(f"[geocode] salvate {len(hospitals)} strutture ({fallbacks} con fallback)")


def build_pharmacies() -> None:
    """Tiene solo le farmacie ancora valide (DATAFINEVALIDITA == '-') con coordinate."""
    out = []
    with open(RAW / "farmacie_lazio.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["DATAFINEVALIDITA"].strip() != "-":
                continue
            try:
                lat, lon = float(row["LATITUDINE"]), float(row["LONGITUDINE"])
            except ValueError:
                continue
            out.append(
                {"name": row["DESCRIZIONEFARMACIA"].strip().title(),
                 "address": row["INDIRIZZO"].strip(),
                 "comune": row["DESCRIZIONECOMUNE"].strip().title(),
                 "lat": lat, "lon": lon}
            )
    (DATA / "pharmacies.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[pharmacies] salvate {len(out)} farmacie valide con coordinate")


if __name__ == "__main__":
    build_pharmacies()
    geocode_hospitals()
