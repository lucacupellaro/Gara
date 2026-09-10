# TASK — Frontend (mappa + chat) e Agente LLM con tool (modelli mockati)

> Task autonomo e self-contained per un agente. Nessun contesto pregresso richiesto: tutto il necessario è in questo documento.
> Task gemello: `task-ml-attese.md` (modulo ML, svolto da un'altra persona in parallelo) — **coordinamento solo tramite i contratti definiti qui sotto**.

## Obiettivo

Costruire in `/home/alex/Desktop/ai2b/triage-assistant/` (creare la cartella se assente) il prototipo web dell'**Emergency Triage Assistant** per la regione Lazio:

- **Mappa del Lazio** (Leaflet + OpenStreetMap) con tutti i pronto soccorso e le farmacie, colorata a **heatmap** secondo un indice di saturazione ricavato dagli accessi per codice triage;
- **Chatbot agentico** (LLM DeepSeek via API, function calling) che dialoga in italiano col cittadino, lo aiuta a categorizzare il problema, *concorda* con lui il codice colore e — in base a posizione e preferenze (meno attesa vs meno viaggio) — raccomanda un ospedale o, per casi lievi, una farmacia;
- I **due modelli sono per ora MOCK STATICI** con contratti fissati (sezione "Mock"): (1) sintomi → patologia probabile + codice colore; (2) stato attuale ospedale → stato dopo t minuti + attesa stimata. Il n.2 verrà sostituito dal modulo ML vero (`ml/predict.py`) sviluppato in parallelo.

## Dati di partenza (verificati, nessuna API key)

1. **Snapshot PS Lazio** — CSV, sep `;`, encoding latin-1, date `dd/mm/YYYY HH:MM`, 49 strutture:
   `https://dati.lazio.it/dataset/144e577e-8a7e-4613-9830-48cbb1d7ee0f/resource/12c31624-f1a4-4874-a903-8954549ddb81/download/output_1627742164504.csv`
   Colonne: `CODICE;ISTITUTO;TIPO;COMUNE;ASL;DATA;STATO_IN_PS;ROSSI_ATT;GIALLI_ATT;VERDI_ATT;BIANCHI_ATT;NONESEG_ATT;TOT_ATT;...;ROSSI_TRATT;...;TOT_TRATT;...;ROSSI_OB;...;TOT_OB;TOT_RT;TUTTI`
   (`_ATT`=in attesa, `_TRATT`=in trattamento, `_OB`=osservazione breve; snapshot statico del 31/07/2021 ~16:30.)
2. **Farmacie** — dati.lazio.it: dataset `farmacie-della-regione-lazio` e `elenco-delle-farmacie-comunali-di-roma` (quest'ultimo già con lat/lon). Trovare l'URL della risorsa CSV via API CKAN: `https://dati.lazio.it/api/3/action/package_show?id=<nome-dataset>`.
3. **Geocoding ospedali** (il CSV non ha coordinate): Nominatim (`https://nominatim.openstreetmap.org/search`), max 1 richiesta/secondo, header `User-Agent` custom obbligatorio. Query: `"<ISTITUTO>, <COMUNE>, Lazio, Italia"`; per i falliti riprovare con solo `"ospedale <COMUNE>"`; risultato salvato in `data/hospitals.json` (così il geocoding gira una volta sola). Verificare a campione che i grandi DEA II di Roma (Gemelli 90501, Umberto I 90600, Tor Vergata 92000, Sant'Andrea 91900) cadano nel comune giusto.

## Struttura da produrre

```
triage-assistant/
├── .env.example              # DEEPSEEK_API_KEY=, DEEPSEEK_BASE_URL=https://api.deepseek.com,
│                             # MODEL=deepseek-chat, MOCK_LLM=0
├── .gitignore                # .env, data/raw/, __pycache__
├── requirements.txt          # fastapi, uvicorn, httpx, pandas, openai, python-dotenv
├── README.md                 # setup, avvio, architettura, come sostituire i mock
├── data/
│   ├── download.py           # scarica CSV PS + farmacie in data/raw/ (idempotente)
│   ├── geocode.py            # Nominatim → data/hospitals.json (+ pharmacies.json normalizzato)
│   └── (raw/, hospitals.json, pharmacies.json)
├── app/
│   ├── main.py               # FastAPI, monta static/, include router, load .env
│   ├── state.py              # stato "adesso" dei PS (vedi sotto) + indice saturazione
│   ├── mocks.py              # i due modelli mock (contratti sotto)
│   ├── scoring.py            # ranking strutture (attesa, viaggio, idoneità)
│   ├── routes_api.py         # /api/hospitals /api/pharmacies /api/predict /api/recommend
│   ├── agent.py              # loop function-calling DeepSeek + definizione tools + system prompt
│   └── routes_chat.py        # POST /api/chat
└── static/
    ├── index.html            # mappa 70% + chat 30%, disclaimer fisso, responsive
    ├── vendor/               # leaflet scaricato in locale (demo offline, niente CDN a runtime)
    ├── map.js
    ├── chat.js
    └── style.css
```

**Divieto:** non creare né modificare nulla in `ml/` — è di competenza del task gemello.

## Stato "adesso" dei PS (`app/state.py`)

Lo snapshot è del 2021 e fermo alle 16:30. Finché non c'è un feed live:
- definire un profilo orario a 24 pesi (minimo notturno 03–06, picco 10–13, picco secondario 15–18, calo serale) normalizzato con peso(16)=1;
- stato_attuale(ospedale) = valori snapshot × peso(ora corrente) con arrotondamento e piccolo rumore deterministico (seed = giorno) per non avere mappe identiche a ogni refresh;
- **indice di saturazione**: `sat = (1.0·VERDI_ATT + 2.0·GIALLI_ATT + 4.0·ROSSI_ATT + 0.5·BIANCHI_ATT) / max(TOT_TRATT, 1)` → fasce: 🟢 <0.5, 🟡 0.5–1, 🟠 1–1.5, 🔴 >1.5;
- il modulo espone `get_all_status()` e `get_status(hospital_code)` e ha un punto unico dove in futuro si aggancerà il feed live.

## I due MOCK (`app/mocks.py`) — contratti vincolanti

```python
def classify_symptoms(symptoms: str, answers: dict | None = None) -> dict:
    """Mock statico a regole. Ritorna:
    {"category": str,                      # es. "respiratorio", "trauma", "gastro", ...
     "possible_condition": str,            # es. "sindrome influenzale" (linguaggio prudente)
     "triage_code": "rosso"|"giallo"|"verde"|"bianco",
     "red_flags": bool,                    # True => il chiamante DEVE indirizzare a 112/118
     "follow_up_questions": list[str],     # domande per raffinare (max 3)
     "confidence": "mock"}"""

def predict_hospital_state(hospital_code: str, minutes_ahead: int, current_state: dict) -> dict:
    """Mock deterministico. STESSA FORMA di ml.predict.predict_queue (task gemello):
    {"waiting_by_code": {"red": int, "yellow": int, "green": int, "white": int},
     "est_wait_minutes": {"yellow": int, "green": int, "white": int},
     "horizon_minutes": int,
     "confidence": "mock"}"""
```

- `classify_symptoms`: tabella statica di ~20 voci keyword→(categoria, condizione, codice). **Red flag obbligatorie** (→ `red_flags=True`, codice rosso): dolore/oppressione toracica, difficoltà respiratoria grave, emorragia abbondante, alterazione di coscienza/svenimento, trauma cranico, deficit improvviso di parola/movimento. Esempi non-red-flag: febbre→verde, mal di gola/raffreddore→bianco, taglio piccolo→bianco, dolore addominale forte→giallo, frattura sospetta→giallo/verde.
- `predict_hospital_state`: coda futura = `current_state` × `profilo_orario[(h+t)%24] / profilo_orario[h]`; attesa stimata per codice = teoria delle code: somma dei pazienti con priorità ≥ del mio codice × tempo medio servizio (rosso 90', giallo 60', verde 40', bianco 25') / max(1, capacità_parallela) dove capacità_parallela = `TOT_TRATT` snapshot / 3.
- **Swap automatico**: all'avvio provare `from ml.predict import predict_queue`; se l'import riesce, usare quello e loggare `"[predict] usando ml.predict reale"`, altrimenti mock e log `"[predict] usando mock"`. Nessun altro accoppiamento con `ml/`.
- **Adattatore firme**: il modulo reale ha firma `predict_queue(hospital_code, at: datetime, current_state)` — l'adattatore converte `minutes_ahead` in `at = datetime.now() + timedelta(minutes=minutes_ahead)`. Tutto il resto del codice usa solo la firma con `minutes_ahead`.

## Agente LLM (`app/agent.py`)

- Client `openai` con `base_url=DEEPSEEK_BASE_URL`, `api_key=DEEPSEEK_API_KEY`, modello da `.env`. **La chiave vive solo in `.env`** (mai committata; `.env.example` con placeholder).
- **Loop function-calling**: inviare messaggi+tools; se la risposta contiene `tool_calls`, eseguire le funzioni Python corrispondenti, appendere ogni risultato come messaggio `role:"tool"` (JSON serializzato) e reinviare; ripetere finché arriva testo; **max 8 iterazioni** poi risposta di cortesia.
- **Tools** (JSON-schema, descrizioni in italiano):
  1. `classify_symptoms(symptoms, answers)` → mock 1
  2. `get_hospitals_status()` → elenco strutture con coda per codice + saturazione
  3. `predict_hospital_state(hospital_code, minutes_ahead)` → mock 2 / ml reale
  4. `travel_time(lat, lon, hospital_code)` → OSRM pubblico `router.project-osrm.org/route/v1/driving/{lon},{lat};{lon2},{lat2}` con timeout 3s e fallback haversine × 35 km/h
  5. `rank_facilities(lat, lon, triage_code, preference)` → scoring.py; `preference ∈ {"meno_attesa","meno_viaggio","bilanciato"}`; per codice bianco include le farmacie
  6. `find_pharmacies_nearby(lat, lon, limit)`
  7. `show_on_map(hospital_codes, center_lat, center_lon)` → non calcola nulla: il backend la intercetta e la inoltra al frontend nel campo `map_action` della risposta
- **System prompt (italiano)** con guardrail espliciti: non fare mai diagnosi definitive ("possibile", "compatibile con"); il codice colore va *proposto e concordato*, mai imposto; se `red_flags=True` o sintomi gravi → interrompere subito ogni ranking e indirizzare a **112/118**; codice bianco/sintomi lievi → suggerire prima la farmacia; massimo 2–3 domande di follow-up prima di proporre; chiedere la posizione (o usare quella condivisa dal browser) prima di raccomandare; sempre ricordare che è un prototipo dimostrativo.
- **API**: `POST /api/chat` body `{"session_id": str, "message": str, "position": {"lat":..,"lon":..}|null}` → `{"reply": str, "map_action": {...}|null}`. Storia conversazione tenuta in memoria per `session_id` (dict, basta per la demo).
- **`MOCK_LLM=1`** in `.env`: bypassa DeepSeek con un copione fisso (saluto → domanda sintomi → proposta codice → raccomandazione chiamando davvero i tool 2–5) per sviluppare/demo-are la UI senza consumare API. Il flusso mock deve comunque passare dai tool reali.

## Frontend (`static/`)

- **`index.html`**: layout flex — mappa ~70%, pannello chat ~30% (colonna su mobile); banner disclaimer sempre visibile: *"Prototipo dimostrativo — non fornisce consigli medici. In emergenza chiama 112"*.
- **`map.js`**: Leaflet (file in `static/vendor/`, scaricati in fase di build della demo — nessun CDN a runtime); tile OSM standard; per ogni ospedale un `circleMarker` con colore = fascia saturazione e raggio ∝ `TOT_ATT`; popup con nome, tipo (PS/DEA I/DEA II), coda per codice colore, attesa stimata verde; layer farmacie (icona diversa) con toggle; pulsante "Usa la mia posizione" (geolocation API, fallback Roma centro 41.9028, 12.4964); legenda fasce colore; refresh dati ogni 60s da `/api/hospitals`.
- **`chat.js`**: bolle utente/bot, spinner durante la richiesta, invio con Enter; pulsanti rapidi ("Ho la febbre", "Mi sono tagliato", "Dove conviene andare ora?"); se la risposta contiene `map_action` → evidenziare i marker indicati (bordo lampeggiante) e centrare la mappa; passare la posizione del browser nel body di `/api/chat`.

## Scoring (`app/scoring.py`)

`score = w_a · attesa_stimata_minuti + w_v · viaggio_minuti`, pesi: meno_attesa (0.8/0.2), meno_viaggio (0.2/0.8), bilanciato (0.5/0.5). L'attesa usata è quella **all'arrivo** (`predict_hospital_state` con `minutes_ahead = viaggio`). Ritorna top-3 con motivazione testuale per ciascuna ("15 min di viaggio, ~40 min di attesa stimata all'arrivo, saturazione bassa"). Codice rosso/giallo con red flag → mai ranking, solo 112.

## Vincoli

- Python 3.13; solo le dipendenze elencate in `requirements.txt`.
- Demo eseguibile **offline** dopo download iniziale, tranne LLM e OSRM (che degrada su haversine). Con `MOCK_LLM=1` la demo è offline al 100%.
- Nessun segreto nel repo: la API key DeepSeek solo in `.env` (in `.gitignore`).
- Tempo risposta `/api/hospitals` < 200 ms; chat < 15 s per turno.
- Codice commentato con parsimonia, README chiaro per il team di gara.

## Verifica finale (da eseguire e riportare)

1. `pip install -r requirements.txt && python data/download.py && python data/geocode.py` completa; `hospitals.json` con 49 strutture geocodificate (riportare quante hanno richiesto il fallback).
2. `uvicorn app.main:app` → su http://localhost:8000 la mappa mostra ospedali colorati con popup e farmacie; la legenda e il disclaimer sono visibili; refresh automatico funziona.
3. Con `MOCK_LLM=1`: conversazione completa da UI, con `map_action` che evidenzia la struttura consigliata.
4. Con DeepSeek reale: *"ho la febbre alta da due giorni"* → il bot fa 1–3 domande, propone codice verde e lo concorda, chiama i tool (loggarli su stdout), raccomanda con attesa+viaggio motivati; *"ho un forte dolore al petto"* → risposta 112 immediata, nessun ranking.
5. `POST /api/recommend` (lat/lon Roma centro, codice verde, "bilanciato") → JSON top-3 con motivazioni.
6. Test swap: creando un file fittizio `ml/predict.py` con una `predict_queue` che ritorna valori fissi, al riavvio il log dichiara l'uso del modulo reale e `/api/predict` riflette i valori fissi. Rimuovere il file fittizio dopo il test.
