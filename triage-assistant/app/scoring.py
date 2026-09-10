"""Ranking delle strutture: attesa stimata all'arrivo + tempo di viaggio."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from . import state
from .predictor import predict_hospital_state

DATA = Path(__file__).parent.parent / "data"
OSRM = "https://router.project-osrm.org/route/v1/driving"

WEIGHTS = {
    "meno_attesa": (0.8, 0.2),
    "meno_viaggio": (0.2, 0.8),
    "bilanciato": (0.5, 0.5),
}

CODE_KEY = {"giallo": "yellow", "arancione": "yellow", "verde": "green",
            "azzurro": "green", "bianco": "white"}


def travel_minutes(lat: float, lon: float, dest_lat: float, dest_lon: float) -> tuple[int, str]:
    """OSRM pubblico con fallback haversine a 35 km/h. Ritorna (minuti, fonte)."""
    try:
        r = httpx.get(
            f"{OSRM}/{lon},{lat};{dest_lon},{dest_lat}",
            params={"overview": "false"}, timeout=3,
        )
        r.raise_for_status()
        routes = r.json().get("routes")
        if routes:
            return max(1, round(routes[0]["duration"] / 60)), "osrm"
    except Exception:
        pass
    km = state.haversine_km(lat, lon, dest_lat, dest_lon)
    return max(1, round(km / 35 * 60)), "haversine"


def travel_route(lat: float, lon: float, dest_lat: float, dest_lon: float) -> tuple[int, str, list]:
    """Come travel_minutes ma ritorna anche la polyline [[lat,lon],...] del percorso.
    Fallback haversine: linea retta tratteggiata."""
    try:
        r = httpx.get(
            f"{OSRM}/{lon},{lat};{dest_lon},{dest_lat}",
            params={"overview": "full", "geometries": "geojson"}, timeout=4,
        )
        r.raise_for_status()
        routes = r.json().get("routes")
        if routes:
            coords = [[c[1], c[0]] for c in routes[0]["geometry"]["coordinates"]]
            return max(1, round(routes[0]["duration"] / 60)), "osrm", coords
    except Exception:
        pass
    km = state.haversine_km(lat, lon, dest_lat, dest_lon)
    return max(1, round(km / 35 * 60)), "haversine", [[lat, lon], [dest_lat, dest_lon]]


def find_pharmacies_nearby(lat: float, lon: float, limit: int = 5,
                           radius_km: float | None = None) -> list[dict]:
    pharmacies = json.loads((DATA / "pharmacies.json").read_text())
    for p in pharmacies:
        p["distance_km"] = round(state.haversine_km(lat, lon, p["lat"], p["lon"]), 2)
    pharmacies.sort(key=lambda p: p["distance_km"])
    if radius_km is not None:
        pharmacies = [p for p in pharmacies if p["distance_km"] <= radius_km]
    return pharmacies[:limit]


def list_hospitals(lat: float | None = None, lon: float | None = None,
                   name_query: str | None = None, max_wait_minutes: int | None = None,
                   triage_code: str = "verde", limit: int = 10) -> dict:
    """Elenco dei PS del Lazio con filtri.

    - lat/lon presenti  -> ordina per distanza in linea d'aria (haversine, km)
    - name_query         -> tiene solo i nomi che contengono tutte le parole date
    - max_wait_minutes   -> tiene solo chi ha attesa stimata <= soglia (per triage_code)
    Senza posizione l'ordinamento e' per attesa stimata crescente.
    """
    code_key = CODE_KEY.get((triage_code or "verde").lower().strip(), "green")
    tokens = [t for t in (name_query or "").lower().split() if t]

    rows = []
    for h in state.get_all_status():
        if tokens and not all(t in h["name"].lower() for t in tokens):
            continue
        wait = predict_hospital_state(h["code"], 0, h)["est_wait_minutes"].get(code_key, 60)
        row = {
            "code": h["code"], "name": h["name"], "type": h["type"],
            "comune": h["comune"], "asl": h["asl"],
            "waiting_by_code": h["waiting_by_code"],
            "saturation": h["saturation"], "saturation_band": h["saturation_band"],
            "est_wait_minutes": wait,
        }
        if lat is not None and lon is not None:
            row["distance_km"] = round(state.haversine_km(lat, lon, h["lat"], h["lon"]), 1)
        rows.append(row)

    if max_wait_minutes is not None:
        rows = [r for r in rows if r["est_wait_minutes"] <= max_wait_minutes]
    rows.sort(key=lambda r: r["distance_km"] if "distance_km" in r else r["est_wait_minutes"])

    return {
        "hospitals": rows[: max(1, int(limit))],
        "count_total": len(rows),
        "sorted_by": "distanza in linea d'aria" if lat is not None else "attesa stimata",
        "filters": {"name_query": name_query, "max_wait_minutes": max_wait_minutes,
                    "triage_code": triage_code},
    }


def rank_facilities(lat: float, lon: float, triage_code: str, preference: str = "bilanciato") -> dict:
    """Top-3 strutture per il codice dato. Per codice bianco include le farmacie."""
    triage_code = triage_code.lower().strip()
    if triage_code == "rosso":
        return {"error": "codice rosso: nessun ranking, chiamare subito il 112/118"}

    w_wait, w_travel = WEIGHTS.get(preference, WEIGHTS["bilanciato"])
    code_key = CODE_KEY.get(triage_code, "green")

    scored = []
    for h in state.get_all_status():
        if "bambino ges" in h["name"].lower():  # pediatrici: solo su richiesta esplicita
            continue
        t_min, t_src = travel_minutes(lat, lon, h["lat"], h["lon"])
        if t_min > 90:  # fuori raggio ragionevole
            continue
        pred = predict_hospital_state(h["code"], t_min, h)
        wait = pred["est_wait_minutes"].get(code_key, 60)
        scored.append({
            "code": h["code"], "name": h["name"], "type": h["type"],
            "comune": h["comune"], "lat": h["lat"], "lon": h["lon"],
            "travel_minutes": t_min, "travel_source": t_src,
            "est_wait_on_arrival": wait,
            "saturation_band": h["saturation_band"],
            "score": w_wait * wait + w_travel * t_min,
            "reason": (f"{t_min} min di viaggio, ~{wait} min di attesa stimata "
                       f"all'arrivo, saturazione {h['saturation_band']}"),
            "prediction_confidence": pred.get("confidence", "mock"),
        })
    scored.sort(key=lambda s: s["score"])
    result: dict = {"hospitals": scored[:3], "preference": preference, "triage_code": triage_code}
    if triage_code == "bianco":
        result["pharmacies"] = find_pharmacies_nearby(lat, lon, 3)
        result["note"] = "per un codice bianco spesso la farmacia risolve prima del PS"
    return result
