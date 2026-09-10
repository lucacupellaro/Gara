# Piano: Prototipo "Emergency Triage Assistant" — mappa Lazio + chatbot agentico + ML attese

## L'idea (ed è realizzabile ✅)

Sito con **mappa del Lazio** (ospedali/PS, farmacie e altre strutture) colorata a **heatmap** in base a occupazione/accessi in tempo reale, più un **chatbot** che dialoga col cittadino, lo aiuta a categorizzare il problema e concordare il codice colore, e in base a preferenze (tempo d'attesa vs tempo di viaggio) e posizione raccomanda un ospedale — o, per casi lievi, una farmacia. Un **modello ML** prevede il tempo d'attesa (attuale, all'arrivo dopo il viaggio, o in un orario successivo migliore) ed è consultato *agenticamente* dal chatbot (LLM DeepSeek via API, function calling).

**Fattibilità verificata sui dati reali:**

- **Heatmap: fattibile subito.** Il CSV `pronto-soccorso-accessi-in-tempo-reale` (dati.lazio.it) ha 49 strutture con colonne: `ISTITUTO;TIPO;COMUNE;ASL;DATA` + pazienti **in attesa / in trattamento / in osservazione per colore triage** (`ROSSI_ATT, GIALLI_ATT, VERDI_ATT, BIANCHI_ATT, ..._TRATT, ..._OB, TOT_*`). È uno snapshot del 31/07/2021.
  URL diretto: `https://dati.lazio.it/dataset/144e577e-8a7e-4613-9830-48cbb1d7ee0f/resource/12c31624-f1a4-4874-a903-8954549ddb81/download/output_1627742164504.csv`
- **Feed live: esiste ma va "sniffato".** Il nuovo salutelazio.it è una app Next.js (`/it/strutture?facilityTypeIds=009`) che mostra tempi d'attesa e triage (nel codice: chiavi `waitingTimes`, `triageCode`, path API `/external-services/facility/structures/...`); l'endpoint dati completo è iniettato a runtime → lo catturiamo con browser headless (Playwright) guardando le chiamate XHR. **Fallback pronto:** simulatore che ricampiona lo snapshot 2021 con pattern orari/settimanali realistici.
- **Geocoding:** il CSV non ha lat/lon degli ospedali → passo una tantum con Nominatim (49 strutture, risultato salvato in JSON).
- **ML: fattibile con onestà.** Non esistono serie storiche orarie open → doppia strategia: (a) da subito un **logger** che salva il feed ogni 10 minuti per costruire il dataset vero; (b) intanto training su serie sintetica ricampionata. La stima d'attesa usa la teoria delle code: pazienti davanti a te per codice × tempo medio di trattamento, con precedenza rossi>gialli>verdi>bianchi.

**Scelte fatte:** stack **FastAPI + Leaflet vanilla** · progetto in **`ai2b/triage-assistant`** (nuova cartella) · **DeepSeek** con API key in `.env` (mai committata).

## Struttura del progetto

```
triage-assistant/
├── .env.example              # DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL
├── requirements.txt          # fastapi, uvicorn, httpx, pandas, scikit-learn, openai, python-dotenv
├── README.md                 # setup + architettura (per il team di gara)
├── data/
│   ├── download.py           # scarica CSV PS + farmacie da dati.lazio.it
│   ├── geocode.py            # geocoding ospedali (Nominatim, rate-limited) → hospitals.json
│   ├── logger.py             # ogni 10 min salva lo stato PS in data/history.csv (per ML futuro)
│   └── (raw/, hospitals.json, pharmacies.json, history.csv)
├── ml/
│   ├── simulate.py           # serie storica sintetica realistica dal CSV 2021
│   │                         #   (picco 10-13, minimo notturno, lunedì +15%, moltiplicatore caldo)
│   ├── train.py              # GradientBoosting sklearn: (ospedale, ora, giorno, festivo, temperatura)
│   │                         #   → pazienti in attesa per codice · salva model.joblib
│   └── predict.py            # predict(hospital, datetime) → coda prevista → stima attesa (teoria code)
├── app/
│   ├── main.py               # FastAPI: monta static/, include routers
│   ├── state.py              # stato PS corrente (live se disponibile, altrimenti simulato "now")
│   ├── routes_api.py         # GET /api/hospitals (stato+coords+indice saturazione)
│   │                         # GET /api/pharmacies · GET /api/predict?hospital&at=
│   │                         # POST /api/recommend {lat,lon,code,pref}
│   ├── agent.py              # ciclo agentico DeepSeek: tools get_live_status, predict_wait,
│   │                         #   rank_facilities, find_pharmacies_nearby + guardrail
│   ├── routes_chat.py        # POST /api/chat {messages} → agent loop → risposta
│   └── scoring.py            # indice saturazione + ranking (attesa stimata, viaggio, idoneità)
└── static/
    ├── index.html            # layout: mappa (70%) + pannello chat (30%)
    ├── map.js                # Leaflet + OSM, marker ospedali (colore = saturazione),
    │                         #   layer heatmap, marker farmacie (toggle), geolocalizzazione browser
    ├── chat.js               # UI chat → /api/chat; il bot evidenzia sulla mappa la struttura consigliata
    └── style.css
```

## Dettagli chiave

### Indice di saturazione (heatmap)
`sat = (TOT_ATT pesato per codice) / capacità_stimata`, con capacità_stimata = media storica di TOT_TRATT per struttura. Colori: 🟢 <0.5 · 🟡 0.5–1 · 🟠 1–1.5 · 🔴 >1.5. Su Leaflet: cerchi con raggio/intensità proporzionali.

### Tempo di viaggio
OSRM pubblico (`router.project-osrm.org`, gratuito, senza chiave) con fallback haversine × 35 km/h.

### Come funziona la parte agentica (la spiegazione che chiedevi)
Il chatbot **non conosce i dati**: ha a disposizione degli *strumenti* (funzioni Python del nostro backend) descritti in JSON. Il ciclo è:

1. Mandi a DeepSeek i messaggi della conversazione + l'elenco `tools` (client `openai` con `base_url=https://api.deepseek.com`, modello `deepseek-chat` — API compatibile OpenAI, quindi il function calling funziona identico).
2. Il modello risponde **o** con testo **o** con una `tool_call` tipo `predict_wait(hospital="Gemelli", at="+45min")`.
3. Il backend esegue davvero la funzione Python, appende il risultato come messaggio `role:"tool"` e rimanda tutto al modello.
4. Si ripete finché il modello produce la risposta testuale finale per l'utente.

È un loop di ~30 righe. I nostri tool: `get_live_status()`, `predict_wait(hospital, at)`, `rank_facilities(lat, lon, code, pref)`, `find_pharmacies_nearby(lat, lon)`.

**Guardrail nel system prompt (i giudici li cercheranno):** mai diagnosi; sintomi red-flag (dolore toracico, dispnea grave, emorragia, alterazione di coscienza…) → stop immediato e indirizzo al 112/118; il codice colore si *concorda* con l'utente, il bot non lo impone; disclaimer sempre visibile nella UI.

## Ordine di implementazione

1. Scaffold progetto + requirements + `.env.example`
2. `data/download.py` + `geocode.py` → `hospitals.json`, `pharmacies.json`
3. `app/main.py` + `/api/hospitals` + mappa Leaflet con heatmap → **prima demo visiva funzionante**
4. `ml/simulate.py` + `train.py` + `predict.py` + `/api/predict`
5. `scoring.py` + `/api/recommend` (con OSRM)
6. `agent.py` + `/api/chat` + UI chat (DeepSeek function calling)
7. `data/logger.py` (da lanciare subito: ogni giorno di log = dati veri per il modello)
8. Best-effort: caccia all'endpoint live di salutelazio.it con Playwright
9. README per il team

## Verifica finale

- `uvicorn app.main:app` → http://localhost:8000: mappa con 49 ospedali colorati + farmacie.
- `POST /api/recommend` con lat/lon di Roma centro, codice verde, pref "tempo_totale" → top-3 strutture motivate.
- Chat end-to-end: *"ho la febbre alta da due giorni"* → il bot fa domande, propone il codice, chiama i tool (visibili nei log), raccomanda struttura/farmacia. *"Ho un forte dolore al petto"* → risposta 112 immediata, senza tool.
- `python ml/train.py` stampa l'errore (MAE) su validation; `/api/predict?...&at=+2h` mostra attesa ora vs tra 2 ore.
