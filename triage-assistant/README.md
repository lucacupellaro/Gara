# Emergency Triage Assistant — Lazio (prototipo di gara)

Mappa live dei pronto soccorso del Lazio + chatbot agentico che aiuta il cittadino a
capire **dove conviene andare** per codici a bassa intensità. Basato sugli open data
del Portale Open Data Regione Lazio (vincolo di gara).

> ⚠️ Prototipo dimostrativo: non fornisce consigli medici. In emergenza: **112**.

## Avvio rapido

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data/download.py      # scarica i CSV open data in data/raw/
.venv/bin/python data/geocode.py       # farmacie + geocoding ospedali (~1 min la prima volta)
cp .env.example .env                   # poi incolla la tua DEEPSEEK_API_KEY
.venv/bin/uvicorn app.main:app --port 8000
# → http://localhost:8000
```

Con `MOCK_LLM=1` nel `.env` il chatbot usa un copione fisso che **chiama comunque i
tool veri** (ranking, farmacie, mappa): demo 100% offline, zero costi API.
Con `MOCK_LLM=0` e una chiave DeepSeek valida il chatbot è un vero agente LLM.

## Architettura

```
dati.lazio.it ──> data/raw/*.csv ──> data/*.json (geocodificati)
                                          │
static/ (Leaflet)  <── /api/hospitals ── app/state.py   stato "adesso" + saturazione
   mappa+heatmap   <── /api/pharmacies                  │
   chat UI ──POST /api/chat──> app/agent.py ──tools──> app/mocks.py | ml/predict.py
                                (DeepSeek,               classify_symptoms
                                 function calling)       predict_hospital_state
                               /api/recommend ── app/scoring.py (attesa+viaggio, OSRM)
```

- **Stato PS**: lo snapshot open data (31/07/2021) viene modulato con un profilo orario
  realistico finché non c'è un feed live; punto di aggancio: `app/state.py::refresh()`.
- **Indice di saturazione**: attesa pesata per codice / pazienti in trattamento;
  fasce verde/giallo/arancio/rosso su mappa e heatmap.
- **I due modelli sono mock** (`app/mocks.py`) con contratti fissati:
  1. `classify_symptoms(sintomi) -> categoria, condizione possibile, codice colore, red_flags`
  2. `predict_hospital_state(ospedale, minuti, stato) -> coda futura + attesa stimata`
- **Swap automatico col modello vero**: se esiste `ml/predict.py` (task del modulo ML),
  `app/predictor.py` lo importa e lo usa al posto del mock (log all'avvio:
  `[predict] usando ml.predict reale`). Nessun'altra modifica necessaria.

## Guardrail del chatbot

- Mai diagnosi definitive; codice colore *proposto e concordato*, mai imposto.
- Red flag (dolore toracico, dispnea grave, emorragia, incoscienza, trauma cranico,
  deficit neurologici) → stop immediato e **112**, nessun ranking.
- Codice bianco → prima la farmacia più vicina.
- Ospedali pediatrici esclusi dal ranking generico.

## API

| Endpoint | Descrizione |
|---|---|
| `GET /api/hospitals` | stato di tutte le strutture (coda per codice, saturazione) |
| `GET /api/pharmacies?lat=&lon=&limit=` | farmacie (tutte o le più vicine) |
| `GET /api/predict?hospital=90501&minutes=60` | coda prevista + attesa stimata |
| `POST /api/recommend` `{lat,lon,triage_code,preference}` | top-3 strutture motivate |
| `POST /api/chat` `{session_id,message,position}` | chatbot; ritorna `reply` + `map_action` |

## Fonti dati

- [Pronto Soccorso — Accessi in tempo reale](https://dati.lazio.it/dataset/pronto-soccorso-accessi-in-tempo-reale) (Regione Lazio, CC-BY)
- [Farmacie della Regione Lazio](https://dati.lazio.it/dataset/farmacie-della-regione-lazio) (Regione Lazio, CC-BY)
- Geocoding: [Nominatim/OpenStreetMap](https://nominatim.openstreetmap.org) · Routing: [OSRM](http://project-osrm.org) (fallback haversine)
