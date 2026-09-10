"""Sanity check della serie sintetica. Va eseguito PRIMA di addestrare.

Criteri. Lo snapshot reale e' UN SOLO istante (31/07/2021, 16:32): con code
di Poisson di media 8 un singolo campione oscilla fra 3 e 14, quindi pretendere
che la media simulata di ogni struttura cada entro ±50% di quel singolo valore
non e' un test di correttezza, e' un test di fortuna. Si verificano invece tre
proprieta' che un singolo campione puo' davvero attestare:

  1. LIVELLO    il totale regionale simulato e' dello stesso ordine del reale
  2. STRUTTURA  le strutture affollate sono le stesse (correlazione di rango)
  3. DINAMICA   esiste un ciclo giorno/notte e le code non divergono
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODES = ("red", "yellow", "green", "white")
COL = {"red": "ROSSI", "yellow": "GIALLI", "green": "VERDI", "white": "BIANCHI"}


def main() -> bool:
    snap, names = {}, {}
    with (ROOT / "data/raw/ps_lazio.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            snap[r["CODICE"]] = sum(float(r.get(f"{COL[c]}_ATT", 0) or 0) for c in CODES)
            names[r["CODICE"]] = r["ISTITUTO"]

    sim, by_hour, per_h = defaultdict(list), defaultdict(list), defaultdict(list)
    with (ROOT / "data/synthetic.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            q = sum(int(row[f"wait_{c}"]) for c in CODES)
            per_h[row["hospital_code"]].append(q)
            by_hour[int(row["hour"])].append(q)
            # confronto sullo STESSO periodo dello snapshot: luglio, ore 16
            if int(row["month"]) == 7 and row["timestamp"][11:] in ("16:15", "16:30", "16:45"):
                sim[row["hospital_code"]].append(q)

    codes = [c for c in snap if c in sim]
    real = np.array([snap[c] for c in codes])
    s = np.array([statistics.mean(sim[c]) for c in codes])

    print("=== 1. LIVELLO — totale regionale, luglio ore 16 ===")
    ratio = s.sum() / max(real.sum(), 1)
    ok1 = 0.5 <= ratio <= 2.0
    print(f"  reale={real.sum():.0f}  simulato={s.sum():.0f}  rapporto={ratio:.2f}x   "
          f"{'PASS' if ok1 else 'FAIL'}  (atteso 0.5-2.0x)")

    print("\n=== 2. STRUTTURA — le affollate sono le stesse? ===")
    sp = np.corrcoef(real.argsort().argsort(), s.argsort().argsort())[0, 1]
    pe = np.corrcoef(real, s)[0, 1]
    ok2 = sp >= 0.6
    print(f"  Spearman={sp:.3f}  Pearson={pe:.3f}   {'PASS' if ok2 else 'FAIL'}  (atteso >=0.6)")
    top_r = [names[codes[i]][:26] for i in np.argsort(-real)[:4]]
    top_s = [names[codes[i]][:26] for i in np.argsort(-s)[:4]]
    for a, b in zip(top_r, top_s):
        print(f"    reale: {a:<28} simulato: {b}")

    print("\n=== 3. DINAMICA — ciclo e stabilita' ===")
    notte = statistics.mean([v for h in (3, 4, 5) for v in by_hour[h]])
    giorno = statistics.mean([v for h in (10, 11, 12) for v in by_hour[h]])
    r = giorno / max(notte, 0.01)
    worst = max(per_h.items(), key=lambda kv: statistics.mean(kv[1]))
    mx = statistics.mean(worst[1])
    ok3 = r > 1.5 and mx < 100
    print(f"  giorno 10-12={giorno:.1f}  notte 03-05={notte:.1f}  rapporto={r:.1f}x")
    print(f"  coda media massima={mx:.1f} ({names[worst[0]][:30]})   "
          f"{'PASS' if ok3 else 'FAIL'}  (ciclo >1.5x, nessuna coda >100)")

    tutto = ok1 and ok2 and ok3
    print(f"\n{'TUTTI I CHECK PASSATI' if tutto else 'CHECK FALLITO: non addestrare'}")
    return tutto


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
