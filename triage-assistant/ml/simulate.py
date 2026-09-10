"""Genera la serie storica sintetica dei PS del Lazio (task-ml-attese.md sez. 3).

Non esiste uno storico orario aperto: si costruisce una simulazione ancorata a
dati reali su tre lati diversi, cosi' che nessun numero sia inventato da zero:

  SCALA e MIX TRIAGE  <- snapshot reale dei 49 PS del Lazio (dati.lazio.it)
  STAGIONALITA'       <- serie giornaliera reale ASST Lariana (ml/calibration.json)
  PROFILO ORARIO      <- profilo parametrico condiviso con app/state.py

Sopra ci gira una coda M/M/c a priorita' assoluta (rosso > giallo > verde >
bianco): gli arrivi sono un processo di Poisson, il servizio consuma capacita'
e i codici bassi vengono scavalcati da quelli alti che arrivano dopo. E' quello
scavalcamento a rendere l'attesa di un codice bianco non lineare nella coda,
ed e' la ragione per cui serve un modello e non una formula.

Uso:  python -m ml.simulate [--giorni 365]
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "synthetic.csv"
CAL = ROOT / "ml" / "calibration.json"

CODES = ("red", "yellow", "green", "white")
COL = {"red": "ROSSI", "yellow": "GIALLI", "green": "VERDI", "white": "BIANCHI"}
SNAPSHOT_HOUR = 16
HOURLY_PROFILE = np.array([
    0.45, 0.38, 0.33, 0.30, 0.30, 0.35, 0.50, 0.70,
    0.95, 1.15, 1.30, 1.35, 1.30, 1.20, 1.10, 1.05,
    1.00, 1.05, 1.10, 1.00, 0.90, 0.80, 0.65, 0.55,
])
# minuti di occupazione media di un posto di trattamento, per codice
SERVICE_MIN = np.array([90.0, 60.0, 40.0, 25.0])
STEP_MIN = 15
MIX_REGIONALE = np.array([0.01, 0.15, 0.70, 0.14])


def load_snapshot() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dallo snapshot: anagrafica, scala relativa, mix triage, capacita'."""
    with (RAW / "ps_lazio.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))

    meta, scale, mix, cap, coda = [], [], [], [], []
    for r in rows:
        def num(k):
            try:
                return float(r.get(k, 0) or 0)
            except ValueError:
                return 0.0
        att = np.array([num(f"{COL[c]}_ATT") for c in CODES])
        trt = np.array([num(f"{COL[c]}_TRATT") for c in CODES])
        presenti = att + trt
        meta.append({"code": r["CODICE"], "name": r["ISTITUTO"], "type": r["TIPO"]})
        # TUTTI = censimento dei presenti alle 16:30 -> scala della struttura
        scale.append(max(num("TUTTI"), 1.0))
        # mix triage della struttura; se i numeri sono troppo piccoli, mix regionale
        mix.append(presenti / presenti.sum() if presenti.sum() >= 8 else MIX_REGIONALE)
        # capacita' = quanti pazienti tiene in trattamento contemporaneamente
        cap.append(max(num("TOT_TRATT"), 2.0))
        coda.append(att.sum())          # in attesa nello snapshot: il bersaglio
    return meta, np.array(scale), np.array(mix), np.array(cap), np.array(coda)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--giorni", type=int, default=365)
    ap.add_argument("--inizio", default="2024-01-01")
    args = ap.parse_args()

    meta, scale, mix, capacity, snapshot_queue = load_snapshot()
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    n = len(meta)
    print(f"[snap] {n} strutture | scala {scale.min():.0f}-{scale.max():.0f} "
          f"| capacita' {capacity.min():.0f}-{capacity.max():.0f}")

    dow = {int(k): v for k, v in cal["dow"].items()}
    month = {int(k): v for k, v in cal["month"].items()}
    hol = cal["holiday"]
    noise_cv = cal["noise_cv"]

    try:
        import holidays
        festivi = holidays.country_holidays("IT")
    except ImportError:
        festivi = set()

    # Tasso di arrivo di riferimento, derivato dalla CAPACITA' e non dalla scala.
    # In regime stazionario una struttura smaltisce capacity/durata_media pazienti
    # al minuto: se lambda supera quel valore la coda diverge senza mai rientrare
    # (derivandolo da `scale` alcune strutture esplodevano a >2000 in attesa).
    # Fissando l'utilizzo RHO all'ora di riferimento si ottiene una coda stabile
    # che cresce nelle ore di punta e si svuota di notte - cioe' il comportamento
    # reale di un PS. La scala della struttura resta rispettata perche' la
    # capacita' e' essa stessa proporzionale alla dimensione.
    durata_media = (mix * SERVICE_MIN).sum(axis=1)          # minuti, per struttura

    # --- utilizzo PER STRUTTURA, invertito dallo snapshot reale ---------------
    # Imporre lo stesso rho a tutte le 49 strutture non puo' riprodurre la
    # realta': nello snapshot Tor Vergata ha 23 persone in attesa e diversi PS
    # di provincia ne hanno zero. Quella differenza NON e' rumore, e' il livello
    # di congestione della singola struttura. Si ricava invertendo la Erlang-C:
    # dato il numero di serventi c e la coda osservata Lq, si cerca il rho che
    # la produce. Cosi' ogni struttura parte dal proprio livello reale.
    def erlang_c_lq(rho: float, c: float) -> float:
        """Lunghezza media della coda in un M/M/c con utilizzo rho."""
        c = max(int(round(c)), 1)
        a = rho * c
        if rho >= 0.999:
            return 1e6
        s_ = sum(a ** k / np.math.factorial(k) for k in range(c)) if c < 150 else None
        if s_ is None:                       # c grande: approssimazione stabile
            return rho / (1 - rho) * rho ** 2
        last = a ** c / (np.math.factorial(c) * (1 - rho))
        pw = last / (s_ + last)              # probabilita' di attesa (Erlang-C)
        return pw * rho / (1 - rho)

    def solve_rho(lq_target: float, c: float) -> float:
        lo, hi = 0.30, 0.985
        for _ in range(40):                  # bisezione: Lq e' monotona in rho
            mid = (lo + hi) / 2
            if erlang_c_lq(mid, c) < lq_target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # La coda media simulata risulta piu' alta del bersaglio Erlang-C perche' la
    # simulazione aggiunge varianza che la formula stazionaria non ha (profilo
    # orario, stagionalita', rumore giornaliero) e la coda e' CONVESSA in rho:
    # per la disuguaglianza di Jensen la media delle code supera la coda della
    # media. TARGET_SCALE compensa quel bias; e' calibrato una volta confrontando
    # il totale regionale simulato di luglio con quello dello snapshot.
    TARGET_SCALE = 0.11
    coda_reale = np.array([q for q in snapshot_queue]) * TARGET_SCALE
    rho_i = np.array([solve_rho(max(q, 0.02), c) for q, c in zip(coda_reale, capacity)])
    rho_i = np.clip(rho_i, 0.35, 0.95)
    print(f"[rho ] per struttura: min={rho_i.min():.2f} mediana={np.median(rho_i):.2f} max={rho_i.max():.2f}")

    # rho_i vale ALLE CONDIZIONI DELLO SNAPSHOT (31 luglio, ore 16): si divide
    # per quel riferimento cosi' la stagionalita' oscilla attorno al livello
    # reale invece di sommarcisi.
    rif_luglio = month.get(7, 1.0) * HOURLY_PROFILE[SNAPSHOT_HOUR]
    lam_ref = rho_i * capacity / durata_media * STEP_MIN / rif_luglio

    rng = np.random.default_rng(42)
    start_dt = datetime.fromisoformat(args.inizio)
    steps = args.giorni * 24 * 60 // STEP_MIN

    # Stato: numero di PAZIENTI (non di codici) in coda e in trattamento.
    # La versione precedente contava i posti occupati per codice (max 4) invece
    # che per paziente: la capacita' non si saturava mai e la coda si svuotava.
    queue = (mix * (scale * 0.30)[:, None]).round()
    treating = np.minimum((mix * capacity[:, None] * 0.7).round(), capacity[:, None])

    # Servizio esponenziale: a ogni passo se ne dimette una frazione
    # STEP_MIN / durata_media. Cosi' la capacita' e' un vincolo vero.
    mu = STEP_MIN / SERVICE_MIN

    OUT.parent.mkdir(parents=True, exist_ok=True)
    day_noise, cur_day = 1.0, None
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hospital_code", "timestamp", "hour", "dow", "month", "holiday",
                    *[f"wait_{c}" for c in CODES], "in_treatment", "arrivals", "capacity"])

        for s in range(steps):
            t = start_dt + timedelta(minutes=STEP_MIN * s)
            if t.date() != cur_day:                       # un rumore per giornata
                cur_day = t.date()
                day_noise = float(rng.normal(1.0, noise_cv))
                day_noise = min(max(day_noise, 0.6), 1.6)
            festivo = t.date() in festivi or t.weekday() == 6
            m = (HOURLY_PROFILE[t.hour] * dow.get(t.weekday(), 1.0)
                 * month.get(t.month, 1.0) * hol.get(str(festivo), 1.0) * day_noise)

            arrivi_tot = rng.poisson(np.maximum(lam_ref * m, 0.01))
            arrivi = np.array([rng.multinomial(int(a), p) for a, p in zip(arrivi_tot, mix)])
            queue += arrivi

            # 1) dimissioni: frazione dei pazienti in trattamento
            dimessi = rng.binomial(treating.astype(int), np.broadcast_to(mu, treating.shape))
            treating -= dimessi

            # 2) ammissioni per priorita' assoluta, fino a saturare la capacita'
            liberi = np.maximum(capacity - treating.sum(axis=1), 0)
            for k in range(4):                            # rosso -> giallo -> verde -> bianco
                presi = np.minimum(queue[:, k], liberi)
                queue[:, k] -= presi
                treating[:, k] += presi
                liberi -= presi
            queue = np.maximum(queue, 0)

            in_tratt = treating.sum(axis=1)
            for i, mh in enumerate(meta):
                w.writerow([mh["code"], t.isoformat(timespec="minutes"), t.hour,
                            t.weekday(), t.month, int(festivo),
                            *queue[i].astype(int), int(in_tratt[i]),
                            int(arrivi[i].sum()), int(capacity[i])])
            if s % 5000 == 0:
                print(f"\r  {s}/{steps} step", end="")
    print(f"\r[ok  ] {OUT}  ({steps * n:,} righe, {args.giorni} giorni x {n} strutture)")


if __name__ == "__main__":
    main()
