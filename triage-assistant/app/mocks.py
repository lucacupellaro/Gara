"""I due modelli, per ora MOCK statici con contratti vincolanti.

1) classify_symptoms: sintomi -> categoria, condizione possibile, codice colore,
   red flag. Verra' sostituito da un classificatore vero.
2) predict_hospital_state: stato attuale -> stato dopo t minuti + attesa stimata.
   Stessa forma di ritorno di ml.predict.predict_queue (task gemello): lo swap
   avviene in app.routes_api / app.agent via try-import.
"""
from __future__ import annotations

from .state import CODES, HOURLY_PROFILE

# minuti medi di servizio per codice triage
SERVICE_MIN = {"red": 90, "yellow": 60, "green": 40, "white": 25}
PRIORITY = ["red", "yellow", "green", "white"]

# --- Mock 1: sintomi -> codice colore ---------------------------------------

RED_FLAGS = [
    "dolore al petto", "dolore toracico", "oppressione al petto",
    "non riesco a respirare", "difficolta a respirare", "difficoltà a respirare",
    "soffoc", "emorragia", "sangue abbondante", "perdo molto sangue",
    "svenut", "svenimento", "incoscien", "perdita di coscienza", "confusione improvvisa",
    "trauma cranico", "battuto la testa", "non riesco a parlare", "non muovo",
    "paralisi", "bocca storta", "convulsion",
]

# keyword -> (categoria, condizione possibile, codice)
RULES = [
    (["febbre alta", "febbre da", "febbre e", "febbre"], "infettivo", "sindrome influenzale o infezione virale", "green"),
    (["mal di gola", "raffreddore", "tosse secca", "naso chiuso"], "respiratorio lieve", "infezione delle vie aeree superiori", "white"),
    (["tosse con catarro", "tosse da giorni"], "respiratorio", "bronchite", "green"),
    (["vomito e diarrea", "diarrea", "vomito"], "gastrointestinale", "gastroenterite", "green"),
    (["dolore addominale forte", "mal di pancia fortissimo", "addome"], "addominale", "dolore addominale da valutare", "yellow"),
    (["mal di pancia", "mal di stomaco", "nausea"], "gastrointestinale", "disturbo gastrico", "white"),
    (["taglio profondo", "ferita profonda"], "trauma", "ferita da suturare", "yellow"),
    (["taglio", "ferita", "mi sono tagliato"], "trauma lieve", "ferita superficiale", "white"),
    (["frattura", "osso rotto", "caduto e non muovo"], "trauma", "sospetta frattura", "yellow"),
    (["distorsione", "caviglia gonfia", "storta"], "trauma lieve", "distorsione", "green"),
    (["mal di testa fortissimo", "cefalea improvvisa"], "neurologico", "cefalea intensa da valutare", "yellow"),
    (["mal di testa", "emicrania"], "neurologico lieve", "cefalea/emicrania", "green"),
    (["bruciore a urinare", "cistite"], "urologico", "infezione urinaria", "white"),
    (["mal di schiena", "lombalgia"], "muscoloscheletrico", "lombalgia", "white"),
    (["eruzione cutanea", "sfogo", "orticaria"], "dermatologico", "reazione cutanea", "green"),
    (["puntura di insetto", "puntura"], "dermatologico", "puntura di insetto", "white"),
    (["mal d'orecchio", "otite", "mal di orecchio"], "orl", "otite", "white"),
    (["occhio rosso", "congiuntivite"], "oculistico", "congiuntivite", "white"),
    (["dolore al fianco", "colica"], "urologico", "sospetta colica renale", "yellow"),
    (["ansia", "attacco di panico", "agitazione"], "psichico", "stato d'ansia acuto", "green"),
]

FOLLOW_UPS = {
    "infettivo": ["Da quanti giorni hai la febbre?", "Hai difficoltà a respirare?", "Hai malattie croniche?"],
    "gastrointestinale": ["Da quanto durano i sintomi?", "Riesci a bere senza vomitare?"],
    "trauma": ["Riesci a muovere la parte colpita?", "La ferita sanguina ancora?"],
    "default": ["Da quanto tempo hai questo sintomo?", "Il dolore è forte (da 1 a 10)?"],
}


def classify_symptoms(symptoms: str, answers: dict | None = None) -> dict:
    text = (symptoms or "").lower()
    if answers:
        text += " " + " ".join(str(v).lower() for v in answers.values())

    if any(flag in text for flag in RED_FLAGS):
        return {
            "category": "emergenza",
            "possible_condition": "possibile emergenza medica",
            "triage_code": "rosso",
            "red_flags": True,
            "follow_up_questions": [],
            "confidence": "mock",
        }

    for keywords, category, condition, code in RULES:
        if any(k in text for k in keywords):
            it_code = {"red": "rosso", "yellow": "giallo", "green": "verde", "white": "bianco"}[code]
            return {
                "category": category,
                "possible_condition": condition,
                "triage_code": it_code,
                "red_flags": False,
                "follow_up_questions": FOLLOW_UPS.get(category.split()[0], FOLLOW_UPS["default"])[:3],
                "confidence": "mock",
            }

    return {
        "category": "non classificato",
        "possible_condition": "da approfondire",
        "triage_code": "verde",
        "red_flags": False,
        "follow_up_questions": FOLLOW_UPS["default"],
        "confidence": "mock",
    }


# --- Mock 2: stato ospedale dopo t minuti + attesa stimata -------------------

def estimate_wait_minutes(waiting_by_code: dict, capacity_parallel: float) -> dict:
    """Teoria delle code semplificata: attesa(codice) = carico dei pazienti con
    priorita' >= alla mia / capacita' di trattamento parallela."""
    est = {}
    for i, code in enumerate(PRIORITY):
        if code == "red":
            continue  # i rossi non aspettano
        ahead_load = sum(
            waiting_by_code.get(c, 0) * SERVICE_MIN[c] for c in PRIORITY[: i + 1]
        )
        est[code] = int(ahead_load / max(1.0, capacity_parallel))
    return est


def predict_hospital_state(hospital_code: str, minutes_ahead: int, current_state: dict) -> dict:
    """Mock deterministico: coda futura = coda attuale x rapporto del profilo orario."""
    from datetime import datetime

    now_h = datetime.now().hour
    future_h = int((now_h + minutes_ahead / 60)) % 24
    factor = HOURLY_PROFILE[future_h] / HOURLY_PROFILE[now_h]

    waiting_now = current_state.get("waiting_by_code", {})
    waiting_future = {c: max(0, round(waiting_now.get(c, 0) * factor)) for c in CODES}
    capacity = max(1.0, current_state.get("in_treatment", 3) / 3)

    return {
        "waiting_by_code": waiting_future,
        "est_wait_minutes": estimate_wait_minutes(waiting_future, capacity),
        "horizon_minutes": minutes_ahead,
        "confidence": "mock",
    }
