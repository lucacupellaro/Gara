# Recap — Classificatore sintomi → colore (implementato)

Stato: **funzionante end-to-end**, integrato nella chat. Implementa `task-ml-attese.md` sez. 6-7.

## Dove sta il modello (importante per la demo)

Il "modello locale" è **due cose separate**, e solo una è nel repo:

| Pezzo | Dove | Dimensione | Nel repo? |
|---|---|---|---|
| **Indice semantico** (i vettori delle 5.634 frasi) | `triage-assistant/ml/index/` | 8,5 MB | ✅ **sì**, committato |
| **Pesi dell'encoder** MiniLM multilingue | `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/` | **458 MB** | ❌ no, cache locale |
| Codice | `ml/*.py`, `app/color_cache.py` | 20 KB | ✅ sì |

⚠️ **Su una macchina pulita il primo avvio scarica 458 MB da HuggingFace.** L'indice committato evita di *ricalcolare* gli embedding, ma non evita il download dell'encoder. Prima della gara: avviare l'app almeno una volta sulla macchina della demo, così la cache è già calda. Non farlo davanti alla giuria.

## File creati

```
triage-assistant/
├── ml/
│   ├── build_index.py           scarica il dataset ed embedda → ml/index/
│   ├── pathology_classifier.py  red flag + kNN semantico → patologie candidate
│   ├── smoke_test.py            verifica la catena senza chiamare l'API
│   └── index/                   index.npz (5.634×384) + label_names.json
├── app/
│   └── color_cache.py           lookup gravità (cache-first → web) + salvataggio
└── data/
    └── pathology_color_cache.json   cache patologia→colore, cresce con l'uso
```

Modificati: `app/agent.py` (3 tool + prompt sez. 7 + trace), `static/chat.js` + `style.css` + `index.html` (markdown + toggle debug), `README.md`, `requirements.txt`.

## Come funziona

```
sintomi (italiano, testo libero)
   │
   ├─► RED FLAG — regole, mai bypassabili ──────────► 🔴 112, STOP
   │
   └─► classify_pathology        MODELLO LOCALE (MiniLM + kNN)
         │                       → top-k patologie con similarità, NON il colore
         ▼
       lookup_pathology_severity cache-first, poi ricerca web su fonti italiane
         ▼
       [DeepSeek decide il colore] → save_pathology_color
         ▼
       🔴 rosso · 🟠 arancione · 🔵 azzurro · 🟢 verde · ⚪ bianco
```

**Il modello dice *che patologia*, l'LLM decide *che colore*.** Il modello non ha mai visto codici colore italiani e non potrebbe impararli (non esistono dati etichettati in italiano); l'LLM può cercare la gravità su fonti ufficiali e motivarla.

## Scelte tecniche, con il motivo

**Retrieval invece di fine-tuning.** Il dataset ha **1.082 patologie su 5.634 esempi** — circa 5 casi per classe. Un classificatore a 1.082 classi lì sopra non è addestrabile e non andava nemmeno tentato. Il kNN sugli embedding non richiede training, capisce l'italiano nativamente (encoder multilingue) e restituisce i top-k con un punteggio: che è esattamente ciò che serve all'LLM per risolvere l'ambiguità a valle. Latenza **6-10 ms** a query.

**Le red flag sono regole, non modello.** Su un'emergenza la varianza di un modello non è accettabile. Quando scattano, il classificatore non viene nemmeno interrogato.

**Arrotondare verso l'alto.** Se le fonti o le candidate sono discordi si sceglie il colore più grave fra i plausibili: è la direzione in cui l'errore costa meno.

**Incerto ≠ non rispondo.** Il bot propone comunque un codice provvisorio e poi fa domande.

## Bug trovati e corretti

| Bug | Perché era grave | Fix |
|---|---|---|
| `"peso FORTISSIMO sul petto"` **non** attivava la red flag | il match per sottostringa si rompe con una parola in mezzo: un mancato match su una red flag è l'errore peggiore possibile | regex che tollerano fino a 2 parole intercalate — **12/12 casi, 0 falsi positivi** |
| DeepSeek non chiamava `lookup_pathology_severity` quando il modello era incerto | e quindi non usciva mai un colore, proprio nel caso più frequente | il passo è ora obbligatorio, sulla candidata **più grave** fra le plausibili |
| Il markdown appariva con gli asterischi | la chat usava `textContent`: nessun markdown veniva renderizzato | renderer minimale in `chat.js`, **con escaping HTML prima** del markdown |
| Ogni chiamata LLM falliva con `APIConnectionError` | `openai>=3` su `httpx2` ha un decoder zstd rotto in alcune installazioni; l'errore sembra di rete e manda a caccia nel posto sbagliato | `Accept-Encoding: identity` sul client |
| La chiave DeepSeek era finita in `.env.example` | file **tracciato**: al primo commit finiva su GitHub in chiaro | spostata in `.env` (gitignorato) prima di qualunque commit — nessuna fuga |

## Sicurezza

- Il contenuto delle pagine web è **dato, non istruzione**: il prompt impone di ignorare comandi trovati nei risultati di ricerca.
- L'HTML viene escapato **prima** del markdown: il testo passa per un LLM che legge pagine web, quindi non deve poter iniettare HTML nel browser.
- Le chiavi solo in `.env`. Prima di ogni commit: `git diff --cached | grep -iE "sk-|api[_-]?key"`.

## Toggle debug

Bottone **🔍 debug** nell'intestazione della chat: sotto ogni risposta mostra quale componente ha risposto a ogni passo (modello locale / cache / web / LLM) e cosa ha restituito. Se l'LLM risponde senza interrogare il modello, il pannello lo dice esplicitamente.

## Cosa NON è ancora fatto

- Il modello ML delle **attese** (`task-ml-attese.md` sez. 1-5 e `predizione-tempo-attesa-arrivo.md`): `/api/hospitals` riporta ancora `"model": "mock"`.
- `data/raw/` è gitignorato e **`dati.lazio.it` risponde spesso 503**: se è giù il giorno della gara l'app non parte. Valutare di committare i CSV.
- Tre correzioni pendenti in `task-ml-attese.md`: path `/home/alex/...`, encoding dichiarato al contrario, URL Lariana 2025 (quello giusto è sotto `/2026/03/`).

## Comandi

```bash
cd triage-assistant
python3 -m ml.pathology_classifier "ho mal di gola e febbre"   # solo modello locale
python3 -m ml.smoke_test                                        # catena completa senza API
python3 -m ml.build_index                                       # rigenera l'indice (~2 min)
uvicorn app.main:app --reload --port 8000                       # app completa
```
