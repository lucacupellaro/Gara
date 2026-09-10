"""Contratto pubblico della predizione di stato (task-ml-attese.md sez. 5).

app/predictor.py importa automaticamente `predict_queue` da qui: se questo
modulo esiste ed e' importabile, il backend smette di usare i mock. Nessun'altra
modifica necessaria.

STRATEGIA IBRIDA, e il motivo. Misurato su 2,06 milioni di coppie di test
(vedi ml/state_model/meta.json), il modello batte la baseline di persistenza
"la coda fra D minuti sara' come adesso" SOLO sugli orizzonti lunghi:

    +15 min  -21.8%   la persistenza vince
    +30 min   -8.1%   la persistenza vince
    +45 min   -2.3%   sostanzialmente pari
    +60 min   +1.2%   il modello inizia a vincere
    +90 min   +5.3%
   +120 min   +9.8%

Ha senso: a 15 minuti la coda non fa in tempo a cambiare, e la fotografia
attuale e' gia' la risposta migliore; oltre l'ora contano il profilo orario e
la stagionalita', che la persistenza ignora. Usare il modello ovunque
peggiorerebbe le predizioni brevi, che sono le piu' frequenti. Quindi:

    D <  SOGLIA  ->  persistenza (piu' accurata li')
    D >= SOGLIA  ->  modello

Dichiararlo apertamente: un modello che si sceglie dove serve vale piu' di un
modello usato ovunque per principio.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "state_model"
CODES = ("red", "yellow", "green", "white")
PRIORITY = CODES
SERVICE_MIN = {"red": 90.0, "yellow": 60.0, "green": 40.0, "white": 25.0}
SOGLIA_MODELLO = 50          # minuti: sotto questa soglia vince la persistenza

_state: dict = {}


def _load():
    if _state:
        return _state
    try:
        import joblib
        meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
        _state.update(models=joblib.load(MODEL_DIR / "model.joblib"), meta=meta, ok=True)
    except Exception as e:                      # senza modello si resta sulla formula
        _state.update(models=None, meta=None, ok=False, err=str(e))
    return _state


def _hourly_persistence(waiting: dict, minutes_ahead: int, now: datetime) -> dict:
    """Persistenza modulata dal profilo orario: la baseline, ma non ingenua."""
    from app.state import HOURLY_PROFILE
    h_now, h_fut = now.hour, int(now.hour + minutes_ahead / 60) % 24
    f = HOURLY_PROFILE[h_fut] / HOURLY_PROFILE[h_now]
    return {c: max(0, round(waiting.get(c, 0) * f)) for c in CODES}


def estimate_wait_minutes(waiting: dict, capacity_parallel: float) -> dict:
    """Teoria delle code: aspetti per i pazienti a priorita' PARI O SUPERIORE
    alla tua, non per la coda totale. E' la ragione per cui un codice bianco
    puo' attendere ore in un PS dove un giallo entra subito."""
    est, davanti = {}, 0.0
    cap = max(capacity_parallel, 1.0)
    for c in PRIORITY:
        # I pazienti DEL TUO STESSO codice gia' in coda vanno contati: sono
        # arrivati prima di te e a parita' di priorita' si procede in ordine di
        # arrivo. Escluderli (sommando solo dopo) azzerava l'attesa dei verdi
        # anche con 14 verdi davanti.
        davanti += waiting.get(c, 0) * SERVICE_MIN[c]
        if c != "red":
            est[c] = int(round(davanti / cap))
    return est


def predict_queue(hospital_code: str, at: datetime, current_state: dict | None = None) -> dict:
    """Coda prevista all'istante `at` (tipicamente adesso + tempo di viaggio)."""
    now = datetime.now()
    minutes_ahead = max(int((at - now).total_seconds() // 60), 0)
    cur = current_state or {}
    waiting = cur.get("waiting_by_code", {}) or {}
    in_treatment = cur.get("in_treatment", 0) or 0
    capacity = max(float(cur.get("capacity", max(in_treatment, 3))), 1.0)

    st = _load()
    usa_modello = st["ok"] and minutes_ahead >= SOGLIA_MODELLO
    metodo = "persistenza_oraria"

    if usa_modello:
        try:
            import numpy as np
            meta = st["meta"]
            hosp_id = meta["hosp_ids"].get(hospital_code, -1)
            # il modello e' addestrato su orizzonti discreti: si usa il piu' vicino
            D = min(meta["horizons"], key=lambda h: abs(h - minutes_ahead))
            riga = {**{f"wait_{c}": waiting.get(c, 0) for c in CODES},
                    "in_treatment": in_treatment, "capacity": capacity,
                    "hour": now.hour, "dow": now.weekday(), "month": now.month,
                    "holiday": 0, "horizon": D, "hosp_id": hosp_id}
            # DataFrame e non array: i modelli sono stati addestrati con nomi di
            # colonna, e passare un array nudo genera un UserWarning a ogni chiamata.
            import pandas as pd
            X = pd.DataFrame([[riga[f] for f in meta["features"]]], columns=meta["features"])
            pred = {c: max(0, int(round(float(st["models"][c].predict(X)[0])))) for c in CODES}
            metodo = "modello"
        except Exception:
            pred = _hourly_persistence(waiting, minutes_ahead, now)
    else:
        pred = _hourly_persistence(waiting, minutes_ahead, now)

    return {
        "waiting_by_code": pred,
        "est_wait_minutes": estimate_wait_minutes(pred, capacity / 3.0),
        "horizon_minutes": minutes_ahead,
        "method": metodo,
        "confidence": ("high" if metodo == "modello" and minutes_ahead <= 120
                       else "medium" if minutes_ahead <= 120 else "low"),
    }


if __name__ == "__main__":
    from datetime import timedelta
    stato = {"waiting_by_code": {"red": 0, "yellow": 2, "green": 14, "white": 3},
             "in_treatment": 18, "capacity": 22}
    for d in (20, 45, 75, 120):
        r = predict_queue("90501", datetime.now() + timedelta(minutes=d), stato)
        print(f"+{d:>3} min  metodo={r['method']:<18} coda={r['waiting_by_code']}  "
              f"attesa verde={r['est_wait_minutes'].get('green')} min")
