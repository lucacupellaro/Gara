"""Punto unico di accesso alla predizione: usa ml.predict se esiste
(modulo reale del task gemello), altrimenti il mock. L'adattatore converte
minutes_ahead nella firma a datetime del modulo reale."""
from datetime import datetime, timedelta

from . import mocks

try:
    from ml.predict import predict_queue as _real_predict_queue  # type: ignore
    USING_REAL_MODEL = True
    print("[predict] usando ml.predict reale")
except Exception:
    _real_predict_queue = None
    USING_REAL_MODEL = False
    print("[predict] usando mock")


def predict_hospital_state(hospital_code: str, minutes_ahead: int, current_state: dict) -> dict:
    if _real_predict_queue is not None:
        at = datetime.now() + timedelta(minutes=minutes_ahead)
        return _real_predict_queue(hospital_code, at, current_state)
    return mocks.predict_hospital_state(hospital_code, minutes_ahead, current_state)
