"""Estrae la stagionalita' REALE dalla serie ASST Lariana (Lombardia).

Non esiste una serie storica oraria aperta dei PS del Lazio: lo snapshot
regionale e' un solo istante (49 righe, 31/07/2021 16:32). Quello che esiste
e' una serie GIORNALIERA reale e pluriennale, pubblicata da ASST Lariana su
dati.gov.it. Da li' si ricavano i moltiplicatori (giorno della settimana,
mese, festivi) che poi modulano la simulazione del Lazio.

Il pattern di arrivo in PS e' strutturalmente simile fra regioni: stessa
stagionalita' settimanale, stesso effetto festivi. Si impara il PATTERN in
Lombardia e si applica al Lazio calibrando la SCALA sullo snapshot regionale.

Uso:  python -m ml.calibrate
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "ml" / "calibration.json"

# Il 2021 e i primi mesi post-covid hanno volumi anomali: esclusi dalla stima.
CUTOFF = date(2022, 1, 1)


def _read(path: Path) -> list[dict]:
    """I file 2024-2025 non sono utf-8 (sono cp1252): provare in ordine."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(encoding=enc) as fh:
                return list(csv.DictReader(fh, delimiter=";"))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"encoding non riconosciuto: {path}")


def load_series() -> dict[date, int]:
    """Accessi giornalieri totali (solo PS Generale, tutte le sedi sommate)."""
    per_day: dict[date, int] = defaultdict(int)
    for path in sorted(RAW.glob("lariana_*.csv")):
        for row in _read(path):
            if "generale" not in (row.get("TIPO PS") or "").lower():
                continue
            try:
                d = datetime.strptime(row["DATA"].strip(), "%d/%m/%Y").date()
                per_day[d] += int(row["NUMERO ACCESSI"])
            except (ValueError, KeyError):
                continue
    return {d: v for d, v in per_day.items() if d >= CUTOFF}


def _multipliers(groups: dict) -> dict:
    """Mediana del gruppo / mediana globale. La mediana, non la media:
    e' robusta agli outlier (giorni di picco anomalo, festivita' locali)."""
    overall = statistics.median([v for vals in groups.values() for v in vals])
    return {str(k): round(statistics.median(v) / overall, 4)
            for k, v in sorted(groups.items()) if v}


def main() -> None:
    series = load_series()
    if not series:
        raise SystemExit("Nessun dato Lariana in data/raw/ (lariana_*.csv)")
    print(f"[data] {len(series)} giorni dal {min(series)} al {max(series)}")

    try:
        import holidays
        festivi = holidays.country_holidays("IT")
    except ImportError:
        print("[warn] libreria 'holidays' assente: uso solo le domeniche come festivi")
        festivi = set()

    by_dow, by_month, by_holiday = defaultdict(list), defaultdict(list), defaultdict(list)
    for d, v in series.items():
        by_dow[d.weekday()].append(v)
        by_month[d.month].append(v)
        by_holiday[bool(d in festivi or d.weekday() == 6)].append(v)

    daily = [series[d] for d in sorted(series)]
    diffs = [abs(b - a) / a for a, b in zip(daily, daily[1:]) if a]

    cal = {
        "fonte": "ASST Lariana (dati.gov.it), PS Generale, "
                 f"{min(series)}..{max(series)}, {len(series)} giorni",
        "media_giornaliera": round(statistics.mean(daily), 1),
        "dow": _multipliers(by_dow),          # 0=lunedi ... 6=domenica
        "month": _multipliers(by_month),
        "holiday": _multipliers(by_holiday),  # "True" = festivo/domenica
        "noise_cv": round(statistics.median(diffs), 4) if diffs else 0.1,
    }
    OUT.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")

    giorni = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]
    print("[dow  ] " + "  ".join(f"{giorni[i]}={cal['dow'].get(str(i), 1):.3f}" for i in range(7)))
    print("[mese ] " + " ".join(f"{m}={cal['month'].get(str(m), 1):.2f}" for m in range(1, 13)))
    print(f"[fest ] {cal['holiday'].get('True', 1):.3f}  (feriale={cal['holiday'].get('False', 1):.3f})")
    print(f"[noise] variazione giorno-su-giorno mediana: {cal['noise_cv']:.1%}")
    print(f"[ok   ] -> {OUT}")


if __name__ == "__main__":
    main()
