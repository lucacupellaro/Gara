"""Stato corrente dei Pronto Soccorso.

Lo snapshot open data e' fermo al 31/07/2021 ~16:30: finche' non c'e' un feed
live, lo stato "adesso" viene modulato con un profilo orario documentato in
letteratura (minimo notturno, picco tardo-mattutino, picco secondario
pomeridiano). Punto unico di aggancio per il futuro feed live: refresh().
"""
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
SNAPSHOT_HOUR = 16  # ora del rilevamento nello snapshot

# Peso relativo degli arrivi per ora del giorno, normalizzato a peso[16]=1.
HOURLY_PROFILE = [
    0.45, 0.38, 0.33, 0.30, 0.30, 0.35, 0.50, 0.70,  # 00-07
    0.95, 1.15, 1.30, 1.35, 1.30, 1.20, 1.10, 1.05,  # 08-15
    1.00, 1.05, 1.10, 1.00, 0.90, 0.80, 0.65, 0.55,  # 16-23
]

CODES = ("red", "yellow", "green", "white")
COL = {"red": "ROSSI", "yellow": "GIALLI", "green": "VERDI", "white": "BIANCHI"}

_hospitals: list[dict] = []          # anagrafica geocodificata
_baseline: dict[str, dict] = {}      # valori snapshot per struttura


def _load() -> None:
    global _hospitals, _baseline
    _hospitals = json.loads((DATA / "hospitals.json").read_text())
    _baseline = {}
    with open(DATA / "raw" / "ps_lazio.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f, delimiter=";"):
            code = row["CODICE"].strip()
            _baseline[code] = {
                "waiting": {c: int(row[f"{COL[c]}_ATT"] or 0) for c in CODES},
                "treating": {c: int(row[f"{COL[c]}_TRATT"] or 0) for c in CODES},
                "tot_treat": int(row["TOT_TRATT"] or 0),
            }


def _modulate(n: int, factor: float, rng: random.Random) -> int:
    # arrotondamento stocastico deterministico, cosi' i piccoli numeri non
    # collassano tutti a zero di notte
    x = n * factor * rng.uniform(0.85, 1.15)
    return max(0, int(x + rng.random()))


def get_status(code: str, now: datetime | None = None) -> dict | None:
    if not _baseline:
        _load()
    base = _baseline.get(code)
    meta = next((h for h in _hospitals if h["code"] == code), None)
    if not base or not meta:
        return None
    now = now or datetime.now()
    factor = HOURLY_PROFILE[now.hour] / HOURLY_PROFILE[SNAPSHOT_HOUR]
    # seed per giorno+ora+struttura: stato stabile entro l'ora, ma vivo nel tempo
    rng = random.Random(f"{now:%Y%m%d%H}-{code}")
    waiting = {c: _modulate(base["waiting"][c], factor, rng) for c in CODES}
    treating = {c: _modulate(base["treating"][c], factor, rng) for c in CODES}
    tot_treat = max(1, sum(treating.values()))
    sat = (4.0 * waiting["red"] + 2.0 * waiting["yellow"]
           + 1.0 * waiting["green"] + 0.5 * waiting["white"]) / tot_treat
    return {
        **meta,
        "waiting_by_code": waiting,
        "in_treatment": sum(treating.values()),
        "total_waiting": sum(waiting.values()),
        "saturation": round(sat, 2),
        "saturation_band": ("green" if sat < 0.5 else "yellow" if sat < 1.0
                            else "orange" if sat < 1.5 else "red"),
        "updated_at": now.isoformat(timespec="minutes"),
        "source": "simulato da snapshot open data 2021",
    }


def get_all_status(now: datetime | None = None) -> list[dict]:
    if not _baseline:
        _load()
    out = [get_status(h["code"], now) for h in _hospitals]
    return [s for s in out if s and s.get("lat")]


def refresh() -> None:
    """Punto di aggancio del futuro feed live: oggi ricarica i file locali."""
    _load()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
