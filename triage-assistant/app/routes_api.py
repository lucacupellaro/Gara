from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import state
from .predictor import USING_REAL_MODEL, predict_hospital_state
from .scoring import find_pharmacies_nearby, rank_facilities, travel_route

import json
import os
from pathlib import Path

router = APIRouter(prefix="/api")
DATA = Path(__file__).parent.parent / "data"


@router.get("/config")
def config():
    """Config pubblica per il frontend (la chiave tile è comunque visibile nel browser)."""
    return {"map_key": os.getenv("MAP_API_KEY", "")}


@router.get("/hospitals")
def hospitals():
    return {"hospitals": state.get_all_status(), "model": "real" if USING_REAL_MODEL else "mock"}


@router.get("/forecast")
def forecast(minutes: int = 0, triage_code: str = "verde"):
    """Stato PREVISTO di tutte le strutture fra `minutes` minuti.

    Alimenta lo slider temporale della mappa: minutes=0 e' l'adesso, gli altri
    valori (30, 60, 120, 240, 360, 720) sono le posizioni dello slider.
    """
    code_key = {"rosso": "red", "arancione": "yellow", "giallo": "yellow",
                "azzurro": "green", "verde": "green", "bianco": "white"}.get(triage_code, "green")
    out = []
    for h in state.get_all_status():
        if minutes <= 0:
            waiting, wait_min, method, conf = (h["waiting_by_code"], None, "attuale", "osservato")
        else:
            p = predict_hospital_state(h["code"], minutes, h)
            waiting = p["waiting_by_code"]
            wait_min = p["est_wait_minutes"].get(code_key)
            method, conf = p.get("method", "?"), p.get("confidence", "?")
        tot = sum(waiting.values())
        # saturazione ricalcolata sulla coda PREVISTA, non su quella attuale:
        # altrimenti la mappa cambierebbe i numeri ma non i colori
        sat = round(tot / max(h.get("in_treatment", 1), 1), 2)
        out.append({
            "code": h["code"], "name": h["name"], "type": h["type"],
            "comune": h["comune"], "lat": h["lat"], "lon": h["lon"],
            "waiting_by_code": waiting, "total_waiting": tot,
            "est_wait_minutes": wait_min,
            "saturation": sat,
            "saturation_band": ("green" if sat < 0.5 else "yellow" if sat < 1.0
                                else "orange" if sat < 1.5 else "red"),
        })
    return {"minutes": minutes, "triage_code": triage_code, "hospitals": out,
            "method": method if minutes > 0 else "attuale",
            "confidence": conf if minutes > 0 else "osservato",
            "model": "real" if USING_REAL_MODEL else "mock"}


@router.get("/pharmacies")
def pharmacies(lat: float | None = None, lon: float | None = None,
               limit: int = 200, radius_km: float | None = None):
    if lat is not None and lon is not None:
        return {"pharmacies": find_pharmacies_nearby(lat, lon, limit, radius_km)}
    all_p = json.loads((DATA / "pharmacies.json").read_text())
    return {"pharmacies": all_p[:limit]}


@router.get("/travel")
def travel(lat: float, lon: float, hospital: str | None = None,
           dest_lat: float | None = None, dest_lon: float | None = None,
           dest_name: str = "destinazione"):
    """Viaggio in auto verso un ospedale (con attesa stimata all'arrivo)
    oppure verso coordinate libere (es. una farmacia)."""
    h = None
    if hospital:
        h = state.get_status(hospital)
        if not h:
            raise HTTPException(404, f"ospedale {hospital} non trovato")
        dest_lat, dest_lon, dest_name = h["lat"], h["lon"], h["name"]
    elif dest_lat is None or dest_lon is None:
        raise HTTPException(422, "serve 'hospital' oppure 'dest_lat'+'dest_lon'")
    minutes, source, geometry = travel_route(lat, lon, dest_lat, dest_lon)
    est_wait = predict_hospital_state(hospital, minutes, h)["est_wait_minutes"] if h else None
    return {"destination": dest_name, "travel_minutes": minutes, "source": source,
            "geometry": geometry, "est_wait_minutes": est_wait}


@router.get("/predict")
def predict(hospital: str, minutes: int = 60):
    current = state.get_status(hospital)
    if not current:
        raise HTTPException(404, f"ospedale {hospital} non trovato")
    return predict_hospital_state(hospital, minutes, current)


class RecommendBody(BaseModel):
    lat: float
    lon: float
    triage_code: str
    preference: str = "bilanciato"


@router.post("/recommend")
def recommend(body: RecommendBody):
    return rank_facilities(body.lat, body.lon, body.triage_code, body.preference)
