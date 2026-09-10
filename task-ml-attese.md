# TASK — Modulo ML di predizione attese PS (serie sintetica + modello)

> Task autonomo e self-contained per un agente. Nessun contesto pregresso richiesto: tutto il necessario è in questo documento.

## Obiettivo

Costruire il modulo `ml/` del progetto **triage-assistant** (`/home/alex/Desktop/ai2b/triage-assistant/`): dato che **non esiste una serie storica oraria open** dei pronto soccorso del Lazio, bisogna **generare una serie sintetica realistica** ancorata a dati reali, **addestrarci un modello** e **esporre una funzione di predizione** che il backend/chatbot consulterà. La predizione risponde a: *"quante persone in coda (per codice colore) all'ospedale X all'istante T, e quindi quanto aspetterei?"* — dove T può essere adesso, adesso+tempo di viaggio, o un orario futuro.

## Dati di partenza (tutti verificati e raggiungibili, nessuna API key)

1. **Snapshot PS Lazio (ancora per struttura e mix triage)** — CSV, sep `;`, 49 strutture:
   `https://dati.lazio.it/dataset/144e577e-8a7e-4613-9830-48cbb1d7ee0f/resource/12c31624-f1a4-4874-a903-8954549ddb81/download/output_1627742164504.csv`
   Colonne: `CODICE;ISTITUTO;TIPO;COMUNE;ASL;DATA;STATO_IN_PS;ROSSI_ATT;GIALLI_ATT;VERDI_ATT;BIANCHI_ATT;NONESEG_ATT;TOT_ATT;...;ROSSI_TRATT;GIALLI_TRATT;VERDI_TRATT;BIANCHI_TRATT;NONESEG_TRATT;TOT_TRATT;...;ROSSI_OB;GIALLI_OB;VERDI_OB;BIANCHI_OB;TOT_OB;TOT_RT;TUTTI`
   (`_ATT` = in attesa, `_TRATT` = in trattamento, `_OB` = osservazione breve; snapshot del 31/07/2021 ore ~16:30, un rilevamento per struttura.)
   ⚠️ Il file usa encoding latin-1 e date `dd/mm/YYYY HH:MM`.

2. **Serie storica REALE giornaliera (calibrazione stagionalità)** — ASST Lariana (Lombardia), accessi PS al giorno per sede, sep `;`, date `dd/mm/YYYY`:
   - 2021: `https://www.asst-lariana.it/wp-content/uploads/2025/08/ACCESSI_PS_2021.csv`
   - 2022: `https://www.asst-lariana.it/wp-content/uploads/2025/08/ACCESSI_PS_2022.csv`
   - 2023: `https://www.asst-lariana.it/wp-content/uploads/2025/08/ACCESSI_PS_2023.csv`
   - 2024/2025: stessa convenzione URL (`ACCESSI_PS_2024.csv`, `ACCESSI_PS_2025.csv`) — provarle; se 404 cercare "Accessi Pronto Soccorso" su dati.gov.it (organizzazione: ASST Lariana). Se irraggiungibili, bastano 2021–2023.
   Colonne: `DATA;SEDE;TIPO PS;NUMERO ACCESSI` (usare solo `TIPO PS == "PS Generale"`).

3. **Profilo orario**: non esiste fonte open oraria → usare un profilo parametrico documentato in letteratura: minimo notturno 03–06, picco principale 10–13, picco secondario 15–18, calo serale; weekend con mattina più piatta. Implementarlo come vettore di 24 pesi normalizzati, commentando la scelta.

## Cosa produrre (file e contratti)

Lavorare in `/home/alex/Desktop/ai2b/triage-assistant/ml/` (creare le cartelle mancanti; anche `data/raw/` per i CSV scaricati). Python 3.13, dipendenze ammesse: `pandas, numpy, scikit-learn, joblib, holidays, httpx/requests`. Creare/aggiornare `requirements.txt` nella root del progetto.

### 1. `ml/download_data.py`
Scarica i CSV di cui sopra in `data/raw/` (idempotente: salta se già presenti). Gestire encoding e separatore.

### 2. `ml/calibrate.py`
Dalla serie Lariana (PS Generale, tutte le sedi sommate) estrae e salva in `ml/calibration.json`:
- moltiplicatori per giorno della settimana (media normalizzata, es. lun ≈ 1.1);
- moltiplicatori mensili (stagionalità annuale);
- effetto festivi (usare libreria `holidays`, calendario IT);
- deviazione residua giorno-su-giorno (per il rumore).
Metodo suggerito: decomposizione moltiplicativa semplice (media per gruppo / media globale), robusta agli outlier (mediane). Escludere il periodo gen–mag 2021 se visibilmente anomalo (covid).

### 3. `ml/simulate.py`
Genera `data/synthetic.csv`: serie a passo **15 minuti per 2 anni** per **ciascuna delle 49 strutture** del Lazio. Logica:
1. **Tasso di arrivo** λ(ospedale, t) = base_ospedale × peso_orario[h] × molt_giorno_settimana × molt_mese × molt_festivo × rumore. `base_ospedale` derivata dallo snapshot Lazio: usare `TUTTI` (censimento presente alle 16:30) per stimare la scala relativa di ogni struttura.
2. **Mix triage** per ospedale dallo snapshot (quote di rossi/gialli/verdi/bianchi su `TOT_ATT+TOT_TRATT`); se una struttura ha numeri troppo piccoli, usare il mix medio regionale (~1% rossi, 15% gialli, 70% verdi, 14% bianchi).
3. **Arrivi** ~ Poisson(λ·Δt) ripartiti sul mix triage.
4. **Coda simulata** (event-based semplificato a passo 15'): capacità di servizio per struttura stimata da `TOT_TRATT` dello snapshot; priorità assoluta rossi>gialli>verdi>bianchi; tempi medi di servizio: rosso 90', giallo 60', verde 40', bianco 25'.
5. Output per riga: `hospital_code, hospital_name, timestamp, waiting_red, waiting_yellow, waiting_green, waiting_white, in_treatment, arrivals_15m`.
**Sanity check obbligatorio in coda allo script**: la media simulata delle 16:30 del sabato deve stare entro ±50% dei valori dello snapshot reale per struttura (stampare confronto); le code non devono divergere (nessuna struttura con attesa media > 100 persone).

### 4. `ml/train.py`
- Feature: hospital_code (categoria), ora del giorno (sin/cos), giorno settimana, mese (sin/cos), festivo, lag: stato coda a t-1h e t-2h (disponibili al momento della predizione dal feed live).
- Target: `waiting_green` e `waiting_yellow` a t+30', t+60', t+120' (multi-output o un modello per orizzonte — scegliere e motivare nel README).
- Modello: `HistGradientBoostingRegressor` (sklearn). Split temporale (ultimi 3 mesi = test, MAI split casuale).
- Stampare MAE per orizzonte e baseline di confronto (persistenza: "coda futura = coda attuale"). **Accettazione: MAE del modello < MAE della baseline.**
- Salvare `ml/model.joblib` + `ml/feature_meta.json`.

### 5. `ml/predict.py` — contratto pubblico (usato dal backend)
```python
def predict_queue(hospital_code: str, at: datetime, current_state: dict | None = None) -> dict:
    """current_state: {"waiting_green": int, "waiting_yellow": int, ...} dal feed live, se disponibile.
    Ritorna: {"waiting_by_code": {"red":..,"yellow":..,"green":..,"white":..},
              "est_wait_minutes": {"yellow":..,"green":..,"white":..},
              "horizon_minutes": int, "confidence": "model"|"heuristic"}"""
```
- `est_wait_minutes` con teoria delle code: attesa(codice) = Σ pazienti davanti (codici a priorità ≥, in coda prevista) × tempo medio servizio / capacità parallela stimata della struttura.
- Se `at` oltre l'orizzonte del modello (>2h) o modello non caricabile → fallback euristico: profilo orario × media storica sintetica (e `confidence: "heuristic"`).
- Includere `compare_times(hospital_code, now, travel_minutes)` che ritorna attesa adesso vs. all'arrivo vs. l'orario migliore nelle successive 6 ore.

### 6. `ml/README.md`
Come rigenerare tutto (`download_data → calibrate → simulate → train`), le assunzioni fatte, i numeri di validazione ottenuti, e i limiti (dichiarare esplicitamente che il modello è addestrato su dati sintetici calibrati su pattern reali, non su storico Lazio — onestà richiesta in gara).

## Vincoli

- Tutto deve girare **offline dopo il download iniziale** (la demo di gara non può dipendere dalla rete).
- Nessuna API key, nessun servizio a pagamento.
- Tempo di `simulate.py` + `train.py` complessivo < 10 minuti su laptop; `predict_queue` < 100 ms a chiamata.
- Seed fissato (`numpy.random.default_rng(42)`) per riproducibilità.
- Non toccare file fuori da `triage-assistant/` (eccetto la creazione della cartella stessa se assente).

## Verifica finale (da eseguire e riportare)

1. `python ml/download_data.py && python ml/calibrate.py && python ml/simulate.py && python ml/train.py` completa senza errori; riportare i MAE e il confronto con la baseline.
2. Il sanity check di `simulate.py` passa (confronto con snapshot reale stampato).
3. Smoke test predizione:
   ```python
   from ml.predict import predict_queue, compare_times
   predict_queue("90501", datetime.now()+timedelta(minutes=45))   # Gemelli tra 45'
   compare_times("92000", datetime.now(), travel_minutes=25)      # Tor Vergata
   ```
   Output plausibili (attese verdi tipiche 30–300 min, mai negative, notte < giorno).
4. Grafico di controllo salvato in `ml/validation_plot.png`: settimana simulata di un grande DEA II (es. Gemelli 90501) con pattern giorno/notte visibile.
