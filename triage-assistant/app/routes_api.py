from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import state
from .predictor import USING_REAL_MODEL, predict_hospital_state
from .scoring import find_pharmacies_nearby, rank_facilities

import json
from pathlib import Path

router = APIRouter(prefix="/api")
DATA = Path(__file__).parent.parent / "data"


@router.get("/hospitals")
def hospitals():
    return {"hospitals": state.get_all_status(), "model": "real" if USING_REAL_MODEL else "mock"}


@router.get("/pharmacies")
def pharmacies(lat: float | None = None, lon: float | None = None, limit: int = 200):
    if lat is not None and lon is not None:
        return {"pharmacies": find_pharmacies_nearby(lat, lon, limit)}
    all_p = json.loads((DATA / "pharmacies.json").read_text())
    return {"pharmacies": all_p[:limit]}


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
