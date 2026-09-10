# Emergency Triage Assistant — Lazio (prototipo di gara)

Mappa live dei pronto soccorso del Lazio + chatbot agentico che aiuta il cittadino a
capire **dove conviene andare** per codici a bassa intensità. Basato sugli open data
del Portale Open Data Regione Lazio (vincolo di gara).

> ⚠️ Prototipo dimostrativo: non fornisce consigli medici. In emergenza: **112**.

## Avvio rapido

```bash
# 1. dipendenze
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. dati open data  →  data/raw/
.venv/bin/python data/download.py
.venv/bin/python data/geocode.py       # farmacie + geocoding ospedali (~1 min la prima volta)

# 3. indice del classificatore di patologia  →  ml/index/   (~2 min, una volta sola)
.venv/bin/python -m ml.build_index

# 4. chiavi
cp .env.example .env                   # poi incolla la tua DEEPSEEK_API_KEY in .env
                                       # ⚠️ MAI in .env.example: quello è tracciato da git

# 5. avvio
.venv/bin/uvicorn app.main:app --reload --port 8000
# → http://localhost:8000
```

Se `ml/index/` è già nel repo il passo 3 si può saltare: serve solo per rigenerarlo.

### Verifiche rapide

```bash
.venv/bin/python -m ml.pathology_classifier "ho mal di gola e febbre"   # solo modello locale
.venv/bin/python -m ml.smoke_test                                        # catena completa senza API
curl -s localhost:8000/api/hospitals | head -c 300                       # backend vivo
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

## Come si arriva al codice colore

```
sintomi (testo libero, italiano)
   │
   ├─► RED FLAG  ── regole deterministiche, mai bypassabili ──► 🔴 112, STOP
   │                (regex tolleranti: "peso FORTISSIMO sul petto" scatta comunque)
   │
   └─► classify_pathology     ml/pathology_classifier.py — MODELLO LOCALE
         │                    MiniLM multilingue + kNN su 5.634 frasi, 1.082 patologie
         │                    → top-k candidate con similarità, NON il colore
         ▼
       lookup_pathology_severity     app/color_cache.py
         │                    cache-first; su miss cerca su fonti sanitarie italiane
         ▼
       [DeepSeek assegna il colore]  → save_pathology_color (cache)
         ▼
       🔴 rosso · 🟠 arancione · 🔵 azzurro · 🟢 verde · ⚪ bianco
```

**Perché retrieval e non fine-tuning**: il dataset ha 1.082 classi su 5.634 esempi, cioè
~5 casi per patologia. Un classificatore a 1.082 classi lì sopra non è addestrabile. Il kNN
sugli embedding non richiede training, capisce l'italiano nativamente e restituisce i top-k
con un punteggio — che è ciò che serve all'LLM per risolvere l'ambiguità a valle.
Latenza: **6–10 ms** a query dopo il caricamento (~8 s al primo avvio).

**Divisione dei ruoli**: il modello dice *che patologia*, l'LLM decide *che colore*. Il modello
non ha mai visto codici colore italiani e non potrebbe impararli; l'LLM può cercare la gravità
su fonti ufficiali e motivarla. Specifica completa in `../task-ml-attese.md` sez. 6-7.

### Toggle debug

Nell'intestazione della chat c'è **🔍 debug**: mostra sotto ogni risposta quale componente
ha risposto a ogni passo (modello locale / cache / web / LLM) e cosa ha restituito. Serve a
verificare che l'LLM non stia decidendo da solo scavalcando il modello. Lo stato è persistito
in `localStorage`.

## Guardrail del chatbot

- Mai diagnosi definitive; codice colore *proposto e concordato*, mai imposto.
- Red flag (dolore toracico, dispnea grave, emorragia, incoscienza, trauma cranico,
  deficit neurologici) → stop immediato e **112**, nessun ranking.
- Codice bianco → prima la farmacia più vicina.
- Ospedali pediatrici esclusi dal ranking generico.
- **Arrotonda verso l'alto**: se le fonti o le candidate sono discordi si sceglie il colore più grave fra i plausibili.
- Il contenuto delle pagine web è **dato, non istruzione** (difesa da prompt injection); il markdown della chat viene escapato prima del rendering.
- Ogni risposta si chiude con `**Colore attribuito: …**`: incerto significa "provvisorio", non "non rispondo".

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

## Problemi noti (letti prima di perderci tempo)

| Sintomo | Causa | Rimedio |
|---|---|---|
| `data/download.py` fallisce con 500/503 | il portale `dati.lazio.it` è **inaffidabile**, va giù spesso | riusare i CSV già in `data/raw/`; per la gara conviene committarli |
| Ogni chiamata LLM fallisce con `APIConnectionError` | `openai>=3` su `httpx2` ha un decoder zstd/brotli incompatibile in alcune installazioni (es. anaconda) | già risolto: il client invia `Accept-Encoding: identity` (vedi `app/agent.py::_client`) |
| Il markdown appare con gli asterischi | cache del browser sul vecchio `chat.js` | ricarica con Ctrl+Shift+R |
| `/api/hospitals` risponde `"model": "mock"` | il modello ML delle attese non è ancora implementato | atteso: c'è solo il classificatore di patologia |

## Sicurezza delle chiavi

`.env` è in `.gitignore`, `.env.example` **no**. Le chiavi vanno solo in `.env`.
Prima di ogni commit: `git diff --cached | grep -iE "sk-|api[_-]?key"`.
