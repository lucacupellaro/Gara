# TASK — Predizione del tempo di attesa all'arrivo

> Task autonomo e self-contained per un agente. Tutto il necessario è in questo documento.

## Posizionamento rispetto agli altri documenti

**Leggere prima questa sezione: senza, si rischia di riscrivere codice già specificato altrove.**

| Documento | Domanda a cui risponde | Output |
|---|---|---|
| `task-ml-attese.md` sez. 1-5 | *Quante persone ci sono in coda, per colore, alla struttura X all'istante T?* | stato del sistema |
| `task-ml-attese.md` sez. 6-7 | *Che codice colore ha questo cittadino?* | colore |
| **questo documento** | *Quanti minuti aspetto **io**, con **il mio** colore, se arrivo là alle T?* | minuti |

I primi due descrivono **il sistema**, questo descrive **l'esperienza del singolo utente**. Questo modulo **consuma** `predict_queue()` invece di riscriverlo: la coda prevista è un suo input, non il suo output.

Il posto nella catena:

```
sintomi → colore (doc precedente)
             ↓
   l'utente indica dove vuole andare / chiede consiglio
             ↓
   PER OGNI struttura candidata:  ←──── questo documento
        ospedale       → attesa che dipende DAL COLORE
        farmacia / CdC → attesa che NON dipende dal colore (affluenza)
             ↓
   struttura che minimizza (viaggio + attesa)
```

## Obiettivo

Costruire `ml/wait_on_arrival/`: un **secondo modello**, distinto da quello della coda, che stima i minuti di attesa a partire dal momento in cui l'utente **arriva** alla struttura. Viene chiamato dall'LLM come tool, una volta per ogni struttura candidata.

Due rami con logiche diverse, entrambi obbligatori:

- **Ramo A — ospedali/PS.** L'attesa dipende dal colore: un codice bianco passa dietro a tutti i rossi, gialli e verdi che arrivano nel frattempo, quindi la sua attesa non è la coda diviso la capacità. Modellato su `(struttura, colore, orario, stato della coda)`.
- **Ramo B — farmacie e strutture territoriali.** Il colore **non serve**: non c'è triage, si viene serviti in ordine di arrivo. L'attesa si stima dall'**affluenza** nella fascia oraria.

## Dati di partenza (tutti verificati)

**1. Snapshot PS Lazio** — [dataset](https://dati.lazio.it/dataset/pronto-soccorso-accessi-in-tempo-reale/resource/12c31624-f1a4-4874-a903-8954549ddb81), CSV diretto:
`https://dati.lazio.it/dataset/144e577e-8a7e-4613-9830-48cbb1d7ee0f/resource/12c31624-f1a4-4874-a903-8954549ddb81/download/output_1627742164504.csv`
Separatore `;`, **encoding utf-8** (non latin-1), date `dd/mm/YYYY HH:MM`.
Colonne rilevanti: `CODICE;ISTITUTO;TIPO;COMUNE;ASL;DATA` + per ogni colore `_ATT` (in attesa), `_TRATT` (in trattamento), `_OB` (osservazione breve) + `TOT_ATT`, `TOT_TRATT`, `TOT_OB`, `TUTTI`.
⚠️ **Non è una serie storica.** Sono **49 righe, un solo istante** (31/07/2021, 16:32–16:34). Serve come **ancora**: scala relativa delle strutture, mix triage per struttura, e sanity check. Non ci si può addestrare sopra.

**2. Serie sintetica prodotta da `ml/simulate.py`** (specificato in `task-ml-attese.md` sez. 3) — **è qui che sta il training set vero di questo modulo**. Vedi sezione A1.

**3. Farmacie del Lazio, già geolocalizzate** (nessun geocoding da fare):
`https://dati.lazio.it/dataset/551579e1-a65d-4e7e-80d7-9bfe07fb2bdc/resource/7658322d-b629-4d77-a9f1-e4aad7c8f83b/download/farmaciereglaziolatlon.csv`
Separatore `;`, encoding utf-8, colonne `CODICEIDENTIFICATIVOFARMACIA;INDIRIZZO;DESCRIZIONEFARMACIA;CAP;DESCRIZIONECOMUNE;SIGLAPROVINCIA;DATAINIZIOVALIDITA;DATAFINEVALIDITA;DESCRIZIONETIPOLOGIA;LATITUDINE;LONGITUDINE`.
⚠️ **Filtrare obbligatoriamente per validità.** Il file ha **1.546 righe ma solo 986 farmacie ancora valide**: 560 record hanno `DATAFINEVALIDITA` già passata (ce ne sono di chiuse dal 2010). Senza il filtro si consigliano ai cittadini farmacie che non esistono più. Le 986 valide hanno tutte lat/lon.
⚠️ **Il dataset non contiene gli orari di apertura** (nessuna colonna orari/turni): vanno presi altrove, vedi sezione B1.

**4. Affluenza tipo "orari di punta" di Google** — vedi sezione B2: **non esiste un'API ufficiale**, il documento specifica tre strade con i relativi compromessi.

## Ramo A — ospedali

### A0. L'istante della predizione: **l'arrivo**, non adesso

Questa è la regola che governa tutto il ramo A. **La predizione non va mai fatta sull'istante in cui l'utente fa la domanda, ma sull'istante in cui metterà piede nella struttura.**

```
t_partenza = adesso (+ eventuale ritardo dichiarato dall'utente)
t_arrivo   = t_partenza + travel_minutes
              ↑
        è QUESTO l'istante su cui si predice
```

#### Da dove viene `travel_minutes`

**OSRM pubblico** (`router.project-osrm.org`, gratuito, senza chiave), profilo `driving`, dalle coordinate dell'utente a quelle della struttura. Fallback se irraggiungibile o fuori tempo massimo: distanza haversine × 35 km/h in area urbana, 60 km/h fuori. Il valore va messo in cache per coppia (origine arrotondata a ~500 m, struttura): durante una conversazione l'utente non si sposta, e ricalcolarlo a ogni chiamata brucia latenza inutilmente.

#### Perché non basta predire "adesso"

Con 25 minuti di viaggio verso un DEA II a metà mattina, la coda cambia sensibilmente durante il tragitto: si arriva su uno stato diverso da quello che il sito mostrava alla partenza. Consigliare una struttura sulla base della coda **attuale** è l'errore che rende inutili le app di monitoraggio esistenti — mostrano una fotografia già vecchia quando l'utente arriva. **Questo è il valore aggiunto del progetto e va detto esplicitamente in demo.**

#### Il doppio salto temporale (la parte che si sbaglia facilmente)

L'attesa **non** è semplicemente `coda_all_arrivo / capacità`. Ci sono due orizzonti distinti, e vanno modellati entrambi:

| Salto | Cosa succede | Chi lo calcola |
|---|---|---|
| **1. Durante il viaggio** (`now → t_arrivo`) | La coda evolve: arrivano nuovi pazienti, altri vengono presi in carico | `predict_queue(hospital, at=t_arrivo)` — documento precedente |
| **2. Durante l'attesa** (`t_arrivo → presa in carico`) | **Continuano ad arrivare pazienti**, e quelli a priorità superiore ti **scavalcano** | il modello di questo documento |

Il secondo salto è quello che conta per i codici bassi. Un codice bianco che arriva con 12 persone davanti non aspetta "12 × tempo di servizio": aspetta anche tutti i rossi, arancioni e azzurri che entrano nel frattempo e gli passano davanti. Nei momenti di picco un codice bianco può essere scavalcato per ore — ed è esattamente il fenomeno che la traccia di gara chiede di risolvere dirottandolo altrove.

Un modello addestrato sulle attese osservate nella simulazione (sez. A1) cattura questo effetto automaticamente, perché nel simulatore lo scavalcamento avviene davvero. La formula analitica usata come baseline **no**: tende a sottostimare i codici bassi. Se nel confronto della sez. A3 il modello batte la baseline soprattutto su bianco e verde, è il segno che ha imparato la cosa giusta.

#### Orizzonte e ricalcolo

- Se `t_arrivo` cade **oltre 2 ore** da adesso (struttura lontana, o utente che dice "ci vado stasera"), il modello della coda esce dal suo orizzonte: passare a `confidence: "low"` e usare il profilo di densità della fascia (sez. A2.1) invece della predizione puntuale.
- Se l'utente cambia orario di partenza durante la conversazione (*"e se ci vado fra un'ora?"*), **ricalcolare `t_arrivo` e rifare la predizione**, non riusare il risultato precedente. È lo stesso meccanismo di `compare_times()`.
- `t_arrivo` va sempre restituito nel risultato: l'utente deve poter vedere su quale istante è stata fatta la stima.

### A1. Il target e da dove viene la ground truth

Il problema: **nessun dato aperto italiano riporta il tempo di attesa osservato** di un singolo paziente. Non si può quindi addestrare su osservazioni reali.

La soluzione: **strumentare il simulatore**. `ml/simulate.py` simula già la coda a passo 15' con priorità assoluta rossi>gialli>verdi>bianchi; per costruirla deve sapere, per ogni paziente virtuale, quando arriva e quando viene preso in carico. Va quindi esteso per emettere anche `data/waits.csv`, una riga per paziente virtuale:

```
hospital_code, arrival_ts, color, queue_red, queue_yellow, queue_green, queue_white,
in_treatment, capacity, wait_minutes
```

dove `wait_minutes` è la differenza fra presa in carico e arrivo. Su 2 anni × 49 strutture sono milioni di righe: campionarne **500k** stratificando per struttura e colore.

Il modello ML diventa così un **surrogato veloce della simulazione**: impara in un colpo solo ciò che il simulatore calcola in molti passi, e risponde in millisecondi — che è ciò che serve per interrogare 49 strutture a ogni richiesta dell'utente.

**Dichiararlo apertamente nel README**: il modello è addestrato su una simulazione calibrata su dati reali, non su attese osservate. È un limite noto, non un difetto nascosto.

### A2. Feature — i tre ingredienti della predizione

La stima si regge su **tre segnali**, che vanno tenuti distinti e calcolati separatamente. Sono i tre modi diversi in cui una struttura può essere "piena".

#### (1) Densità media della fascia oraria — *quanto è tipicamente affollata questa struttura a quest'ora*

Statistica storica precalcolata, non una feature grezza. Da `data/synthetic.csv` costruire `ml/wait_on_arrival/density_profile.json`:

```json
{"90501": {"lun": {"08": 24.1, "09": 31.7, ..., "23": 12.4},
           "sab": {...}, "festivo": {...}}}
```

cioè, per **ogni struttura × giorno della settimana × ora**, la media di pazienti in attesa. Serve a due cose:
- dà al modello un **prior forte**: sa già che il Gemelli alle 11 di lunedì è mediamente a 32 persone, e deve solo correggere rispetto a quella media;
- è **leggibile da un umano**, quindi alimenta la frase di spiegazione mostrata all'utente e permette a un giudice di verificare che i numeri siano sensati.

Feature derivate: `density_mean` (media della fascia), `density_ratio` = coda attuale / media della fascia — cioè *"oggi è più o meno pieno del solito?"*, che è spesso più informativo del valore assoluto.

#### (2) Occupazione della struttura — *quanto è satura rispetto alla sua capacità*

`occupancy = (TOT_ATT + TOT_TRATT) / capacity`, dove `capacity` è stimata da `TOT_TRATT` dello snapshot Lazio (quanti pazienti la struttura riesce a tenere in trattamento contemporaneamente).

È diverso dalla densità: 20 persone in attesa sono poche al Gemelli e tantissime a un PS di provincia. Senza questa normalizzazione il modello confonde "struttura grande" con "struttura affollata".

#### (3) Stato dei colori presenti — *chi ho davanti, e con che priorità*

La composizione della coda al momento dell'arrivo, non il solo totale:

`queue_rosso, queue_arancione, queue_azzurro, queue_verde, queue_bianco, in_treatment`

più le due derivate che contano davvero per l'utente:
- `ahead_of_me` = somma dei pazienti a **priorità pari o superiore** al colore dell'utente — sono gli unici che lo faranno aspettare;
- `behind_me` = quelli a priorità inferiore, che **non** lo rallentano (ma servono al modello per stimare il carico complessivo sulla struttura).

È qui che il colore entra nel calcolo: **due utenti nella stessa struttura allo stesso istante hanno `ahead_of_me` diversi e quindi attese diverse.** Un codice bianco con 30 verdi davanti aspetta ore; un giallo nella stessa coda entra quasi subito.

⚠️ Al momento della predizione questi valori **non sono osservati** — l'utente non è ancora arrivato. Si ottengono da `predict_queue(hospital, at = now + travel_minutes)` del documento precedente. **È questo l'incastro fra i due moduli**, e il motivo per cui questo modello non deve riscrivere la predizione della coda.

#### (4) Contesto di calendario

`hospital_code` (categoriale), `TIPO` della struttura (`PS` / `DEA I` / `DEA II` / `PS SPEC.`), `color` dell'utente, ora del giorno (sin/cos), giorno della settimana, mese (sin/cos), festivo (`holidays`, calendario IT).

### A3. Modello e criterio di accettazione

- `HistGradientBoostingRegressor` (sklearn), target `wait_minutes`.
- **Split temporale**, mai casuale: ultimi 3 mesi della serie = test.
- **Baseline obbligatoria da battere**: la formula analitica di teoria delle code già prevista nel piano — `attesa(colore) = Σ pazienti a priorità ≥ × tempo medio di servizio / capacità parallela`. Se il modello non batte la formula, **si tiene la formula** e lo si dichiara: è più semplice, più veloce e più spiegabile.
- Riportare il MAE **per colore**, non solo aggregato: l'errore sui codici bianchi (attese lunghe, alta varianza) è strutturalmente diverso da quello sui gialli.
- Salvare `ml/wait_on_arrival/model.joblib` + `feature_meta.json`.

### A4. Contratto

```python
def predict_wait_on_arrival(hospital_code: str, color: str,
                            depart_at: datetime, travel_minutes: int,
                            current_state: dict | None = None) -> dict:
    """Si passano PARTENZA e VIAGGIO, non l'istante di arrivo: l'arrivo lo calcola
    la funzione (t_arrivo = depart_at + travel_minutes), cosi' il chiamante non puo'
    sbagliarlo ne' dimenticarsi di sommare il tragitto.

    Ritorna:
    {"arrival_at": datetime,           # l'istante su cui e' stata fatta la stima
     "wait_minutes": int,              # attesa DA t_arrivo alla presa in carico
     "range_minutes": [int, int],      # incertezza, non un numero secco
     "total_minutes": int,             # travel_minutes + wait_minutes
     "queue_at_arrival": {"rosso":.., "arancione":.., "azzurro":.., "verde":.., "bianco":..},
     "ahead_of_me": int,               # pazienti a priorita' >= al colore dell'utente
     "method": "model" | "queue_formula" | "density_profile",
     "confidence": "high" | "medium" | "low"}
    """
```

Restituire sempre un **intervallo** oltre al valore puntuale: dire "aspetterai 47 minuti" è falsa precisione e in demo si ritorce contro; "fra 40 e 70 minuti" è difendibile.

## Ramo B — farmacie e strutture territoriali

Qui **il colore non entra nel calcolo**. Va accettato solo per coerenza di firma e ignorato.

### B1. Dati

- Farmacie: dataset del punto 3, **filtrato sulle 986 valide**.
- Orari di apertura: assenti dal dataset regionale → prenderli da **OpenStreetMap via Overpass API** (gratis, senza chiave), interrogando `amenity=pharmacy` nel riquadro del Lazio e leggendo il tag `opening_hours`. Copertura parziale: dove manca, assumere l'orario tipico italiano (08:30–13:00 / 16:00–19:30, chiuso la domenica) e marcare `confidence: "low"`.
- Case della Comunità: non esiste un dataset aperto dedicato (verificato). Se non se ne trova uno, popolare a mano una lista ridotta per Roma e dichiararla come tale.

### B2. Affluenza — nessuna API ufficiale, tre strade

**Google Places API non espone gli "orari di punta"**: la busyness è mostrata nell'interfaccia di Maps ma non è un campo delle API ufficiali. Le opzioni reali sono:

| Strada | Costo | Affidabilità | Note |
|---|---|---|---|
| **Profilo parametrico** (nessun servizio esterno) | 0 | media, ma stabile | **Consigliata per la gara**: nessuna dipendenza, funziona offline, zero rischi in demo |
| `populartimes` (scraping non ufficiale) | 0 | fragile | Si rompe quando Google cambia il markup; verificare che il fork usato sia aggiornato |
| BestTime.app / Outscraper | a pagamento | buona | Fuori dai vincoli del progetto (nessun servizio a pagamento) |

**Decisione da prendere e documentare.** Se si tenta lo scraping, deve esserci il fallback parametrico dietro, e i risultati vanno messi in cache su disco: la demo non può dipendere da uno scraper che si rompe il giorno della presentazione.

### B3. Stima parametrica

Attesa in farmacia = tempo di servizio × persone davanti. Ordini di grandezza da usare come default, documentandoli:
- tempo di servizio medio: **3 minuti** per cliente
- persone davanti: profilo orario a 24 pesi (picchi 09–11 e 17–19, minimo 14–16), scalato per la dimensione del comune
- risultato tipico: **2–20 minuti**, mai più di 30

Il punto importante da comunicare all'utente non è il numero esatto, ma **l'ordine di grandezza rispetto al PS**: 10 minuti contro 4 ore è la ragione per cui la raccomandazione ha senso.

### B4. Contratto

```python
def estimate_wait_territorial(facility_id: str, at: datetime) -> dict:
    """Il colore non è un parametro: in farmacia non c'è triage.
    Ritorna:
    {"wait_minutes": int,
     "range_minutes": [int, int],
     "is_open": bool,               # se chiusa -> None sui minuti e prossima apertura
     "next_open": datetime | None,
     "method": "affluence" | "parametric",
     "confidence": "high" | "medium" | "low"}
    """
```

⚠️ **Controllare sempre `is_open` prima di consigliare.** Una farmacia chiusa con attesa stimata 5 minuti è il tipo di errore che affonda una demo.

## Il tool unico esposto all'LLM

L'LLM non deve sapere quale dei due rami si applica: sceglie il modulo in base al tipo di struttura.

```python
def estimate_wait(facility_id: str, color: str | None, arrival_at: datetime) -> dict:
    """Router unico. Se la struttura è un PS/DEA -> ramo A (color obbligatorio).
    Altrimenti -> ramo B (color ignorato).
    Aggiunge sempre al risultato:
      "facility_type": "ospedale" | "farmacia" | "casa_comunita",
      "color_used": bool,
      "explanation": str   # frase in italiano, es. "Con codice verde, al Gemelli alle 18:30
                           #  ci sono circa 24 persone davanti a te con priorità pari o superiore."
    """
```

Registrare lo schema JSON in `ml/tools_schema.json` insieme agli altri tool.

**Come l'LLM lo usa:** una chiamata per ogni struttura candidata (max 10, per latenza), poi ordina per `travel_minutes + wait_minutes` e presenta le prime 3 con la `explanation`. Il campo `explanation` è ciò che rende la raccomandazione convincente in demo: senza, è una lista di numeri.

## Vincoli

- Offline dopo il download iniziale: cache su disco per qualunque dato esterno (affluenza, orari OSM), committata nel repo.
- Nessuna API key, nessun servizio a pagamento.
- `estimate_wait` < **100 ms** a chiamata: viene invocato fino a 10 volte per singola richiesta utente.
- Seed fissato (`numpy.random.default_rng(42)`).
- Non presentare mai un numero secco senza intervallo, e mai una struttura senza aver verificato che sia aperta.

## Verifica finale (da eseguire e riportare)

1. `simulate.py` produce `data/waits.csv` con `wait_minutes` non negativi e coerenti con la priorità: a parità di struttura e istante, `attesa(rosso) < attesa(giallo) < attesa(verde) < attesa(bianco)`. Se questa disuguaglianza non regge, il simulatore è rotto e il modello non va addestrato.
2. `density_profile.json` generato e ispezionabile a occhio: per un grande DEA II la densità notturna (03–05) deve essere nettamente inferiore a quella di metà mattina (10–12), e il sabato diverso dal mercoledì. Se il profilo è piatto, il simulatore a monte non sta funzionando.
3. `train.py` del ramo A stampa MAE **per colore** e il confronto con la baseline analitica; dichiarare quale delle due si tiene.
4. `predict_wait_on_arrival("90501", "verde", now+45min)` → valore plausibile con intervallo (attese verdi tipiche 30–300 min); di notte deve essere sensibilmente minore che a mezzogiorno.
5. `estimate_wait_territorial` su una farmacia chiusa → `is_open: false` e `next_open` valorizzato.
6. Confronto che deve comparire nella demo: stessa ora, stesso utente, codice **bianco** → il PS grande dà un'attesa **molto maggiore** della farmacia vicina. È la dimostrazione visiva della tesi del progetto.
7. Test del router: `estimate_wait(<farmacia>, color="verde", ...)` deve ignorare il colore e restituire `color_used: false`.
