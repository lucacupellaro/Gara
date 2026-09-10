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

CODE_KEY = {"giallo": "yellow", "verde": "green", "bianco": "white"}


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


def find_pharmacies_nearby(lat: float, lon: float, limit: int = 5) -> list[dict]:
    pharmacies = json.loads((DATA / "pharmacies.json").read_text())
    for p in pharmacies:
        p["distance_km"] = round(state.haversine_km(lat, lon, p["lat"], p["lon"]), 2)
    pharmacies.sort(key=lambda p: p["distance_km"])
    return pharmacies[:limit]


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
