"""Addestra il modello che predice lo STATO di un PS all'istante di arrivo.

Domanda a cui risponde: "se parto adesso e ci metto D minuti, quante persone
per codice troveró in attesa quando arrivo?"

  input : stato ORA (coda per codice, in trattamento, capacita') + calendario
          + D = minuti che mancano all'arrivo
  output: coda per codice a t+D

Baseline da battere: la PERSISTENZA ("la coda fra D minuti sara' come adesso").
E' la baseline giusta perche' e' esattamente cio' che fanno le app di
monitoraggio esistenti, che mostrano la fotografia corrente. Se il modello non
la batte, il progetto non aggiunge nulla e va detto.

Uso:  python -m ml.train_state
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SERIE = ROOT / "data" / "synthetic.csv"
OUT = ROOT / "ml" / "state_model"
CODES = ("red", "yellow", "green", "white")
STEP_MIN = 15
ORIZZONTI = (15, 30, 45, 60, 90, 120)          # minuti di viaggio plausibili


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per ogni orizzonte D costruisce le coppie (stato_t, D) -> stato_{t+D}."""
    df = df.sort_values(["hospital_code", "timestamp"]).reset_index(drop=True)
    pezzi = []
    for D in ORIZZONTI:
        shift = D // STEP_MIN
        g = df.groupby("hospital_code", sort=False)
        blocco = df.copy()
        blocco["horizon"] = D
        for c in CODES:
            blocco[f"y_{c}"] = g[f"wait_{c}"].shift(-shift)
        pezzi.append(blocco.dropna(subset=[f"y_{c}" for c in CODES]))
    return pd.concat(pezzi, ignore_index=True)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    import joblib

    print("[data] lettura serie sintetica…")
    df = pd.read_csv(SERIE, parse_dates=["timestamp"])
    print(f"       {len(df):,} righe, {df.hospital_code.nunique()} strutture")

    ds = build_dataset(df)
    print(f"[pair] {len(ds):,} coppie (stato_t, D) -> stato_t+D  su {len(ORIZZONTI)} orizzonti")

    # SPLIT TEMPORALE, mai casuale: con una serie storica lo split casuale
    # mette istanti adiacenti in train e test e gonfia il punteggio.
    taglio = ds["timestamp"].quantile(0.8)
    tr, te = ds[ds.timestamp < taglio], ds[ds.timestamp >= taglio]
    print(f"[split] train fino a {taglio:%Y-%m-%d} ({len(tr):,}) | test ({len(te):,})")

    feat = ([f"wait_{c}" for c in CODES]
            + ["in_treatment", "capacity", "hour", "dow", "month", "holiday", "horizon"])
    ds["hosp_id"] = ds["hospital_code"].astype("category").cat.codes
    tr, te = ds[ds.timestamp < taglio], ds[ds.timestamp >= taglio]
    feat = feat + ["hosp_id"]

    OUT.mkdir(parents=True, exist_ok=True)
    modelli, righe = {}, []
    for c in CODES:
        m = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.08, max_depth=7,
            min_samples_leaf=40, random_state=42)
        m.fit(tr[feat], tr[f"y_{c}"])
        pred = np.maximum(m.predict(te[feat]), 0)
        mae_m = mean_absolute_error(te[f"y_{c}"], pred)
        mae_b = mean_absolute_error(te[f"y_{c}"], te[f"wait_{c}"])   # persistenza
        modelli[c] = m
        righe.append((c, mae_m, mae_b))
        print(f"[{c:<6}] MAE modello={mae_m:.3f}  baseline(persistenza)={mae_b:.3f}  "
              f"{'MEGLIO' if mae_m < mae_b else 'PEGGIO'} ({(1-mae_m/max(mae_b,1e-9)):+.1%})")

    # per orizzonte: il vantaggio deve crescere con la distanza temporale
    print("\n[per orizzonte, codice verde]")
    for D in ORIZZONTI:
        s = te[te.horizon == D]
        if s.empty:
            continue
        p = np.maximum(modelli["green"].predict(s[feat]), 0)
        mm = mean_absolute_error(s["y_green"], p)
        mb = mean_absolute_error(s["y_green"], s["wait_green"])
        print(f"   +{D:>3} min:  modello={mm:.3f}  persistenza={mb:.3f}  {(1-mm/max(mb,1e-9)):+.1%}")

    joblib.dump(modelli, OUT / "model.joblib")
    (OUT / "meta.json").write_text(json.dumps({
        "features": feat, "codes": list(CODES), "horizons": list(ORIZZONTI),
        "hosp_ids": {k: int(v) for k, v in
                     zip(ds["hospital_code"], ds["hosp_id"])},
        "mae": {c: {"model": round(a, 4), "baseline": round(b, 4)} for c, a, b in righe},
        "train_until": str(taglio), "n_train": len(tr), "n_test": len(te),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    vinti = sum(1 for _, a, b in righe if a < b)
    print(f"\n[ok] modello salvato in {OUT}")
    print(f"     batte la persistenza su {vinti}/{len(CODES)} codici "
          f"-> {'ACCETTATO' if vinti >= 3 else 'RIFIUTATO: tenere la baseline'}")


if __name__ == "__main__":
    main()
