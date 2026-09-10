# Architettura dei modelli e orchestrazione

> ⚠️ **Precisazione necessaria.** Nel progetto **non c'è una rete neurale addestrata da noi.**
> I modelli sono due, di natura diversa:
>
> | | Cos'è | Chi l'ha addestrato |
> |---|---|---|
> | **Classificatore di patologia** | rete neurale *transformer* (MiniLM), usata come encoder | pre-addestrata da terzi; noi **non** l'abbiamo addestrata |
> | **Predittore delle attese** | **gradient boosting su alberi decisionali** | addestrato da noi sui dati sintetici |
>
> Il modello che abbiamo addestrato **non ha layer né funzioni di attivazione**: sono alberi
> decisionali. Chiamarlo "rete neurale" in presentazione è un errore che una giuria tecnica
> nota subito. Qui sotto c'è l'architettura reale di entrambi.

---

## 1. Predittore delle attese — gradient boosting (il modello che abbiamo addestrato)

**File**: `ml/train_state.py` · **Modello salvato**: `ml/state_model/model.joblib`

### Perché alberi e non una rete neurale

Il problema è tabellare: 12 feature numeriche, nessuna struttura spaziale o sequenziale da
sfruttare. Su dati tabellari il gradient boosting batte sistematicamente le reti neurali a
parità di sforzo, si addestra in minuti invece che in ore, non richiede normalizzazione delle
feature né tuning del learning rate, e resta ispezionabile. Una rete qui sarebbe stata una
scelta peggiore presa per ragioni di immagine.

### Architettura

`HistGradientBoostingRegressor` (scikit-learn). Non ha layer: è una **somma di 220 alberi
decisionali**, ciascuno addestrato sull'errore residuo dei precedenti.

```
predizione(x) = albero₁(x) + 0.08·albero₂(x) + 0.08·albero₃(x) + … + 0.08·albero₂₂₀(x)
```

| Iperparametro | Valore | Significato |
|---|---|---|
| `loss` | `squared_error` | errore quadratico: penalizza gli sbagli grossi |
| `max_iter` | **220** | numero di alberi (l'equivalente della "profondità" qui) |
| `learning_rate` | **0.08** | quanto ogni albero corregge: basso = più alberi, meno overfitting |
| `max_depth` | 7 | profondità massima di ogni albero |
| `max_leaf_nodes` | 31 | foglie per albero |
| `min_samples_leaf` | 40 | minimo di campioni per foglia: evita foglie che memorizzano rumore |
| `max_bins` | 255 | le feature continue sono discretizzate in 255 bin (è l'"Hist" del nome, ciò che lo rende veloce su milioni di righe) |
| `l2_regularization` | 0.0 | non serve: la regolarizzazione arriva da `min_samples_leaf` e dal learning rate |
| `random_state` | 42 | riproducibilità |

**Funzioni di attivazione: nessuna.** Non esistono in un albero decisionale. Ogni nodo è un
confronto `feature < soglia`, ogni foglia un valore costante. È la ragione per cui il modello
è ispezionabile: si può leggere *perché* ha predetto un numero.

**Quattro modelli separati**, uno per codice colore (`red`, `yellow`, `green`, `white`):
le dinamiche sono diverse — i rossi non fanno coda, i bianchi la fanno enorme — e quattro
modelli specializzati battono un multi-output unico.

### Feature (12)

```
wait_red, wait_yellow, wait_green, wait_white   stato della coda ADESSO
in_treatment, capacity                          quanto la struttura sta gia' smaltendo
hour, dow, month, holiday                       contesto temporale
horizon                                         ← fra quanti minuti voglio la predizione
hosp_id                                         identita' della struttura
```

`horizon` è la feature che rende il modello utilizzabile a orizzonti diversi con un solo
addestramento: invece di allenare un modello per "+30 min", uno per "+2 ore" ecc., l'orizzonte
è un input.

### Dati di addestramento

Non esiste uno storico orario aperto dei PS del Lazio. La serie è **sintetica** (`ml/simulate.py`),
ancorata al reale su tre lati:

| Cosa | Da dove |
|---|---|
| scala e mix triage per struttura | snapshot reale dei 49 PS del Lazio (dati.lazio.it) |
| stagionalità (giorno, mese, festivi) | **1.461 giorni reali** di ASST Lariana |
| profilo orario | parametrico, condiviso con `app/state.py` |

Sopra ci gira una **coda M/M/c a priorità assoluta** (rosso > giallo > verde > bianco): arrivi
di Poisson, servizio esponenziale, i codici bassi scavalcati da quelli alti che arrivano dopo.
L'utilizzo ρ di ogni struttura è ricavato **invertendo la formula di Erlang-C** sulla coda
osservata nello snapshot, così ogni ospedale parte dal proprio livello reale di congestione.

Validazione del simulatore (`ml/sanity.py`, da eseguire **prima** di addestrare):
livello entro il **24%** del totale regionale reale, correlazione di rango **Spearman 0.85**.

### Addestramento

```
serie sintetica     1.716.960 righe   (15 min × 365 giorni × 49 strutture)
coppie generate     9.000.000         (stato_t, D) → stato_{t+D} su 9 orizzonti
                                      D ∈ {15, 30, 45, 60, 90, 120, 240, 360, 720} minuti
split               TEMPORALE 80/20 — mai casuale: con una serie storica lo split
                    casuale mette istanti adiacenti in train e test e gonfia il punteggio
train / test        7.199.786 / 1.800.214
```

```bash
python -m ml.calibrate     # stagionalità dai dati reali Lariana
python -m ml.simulate      # serie sintetica
python -m ml.sanity        # se fallisce, NON addestrare
python -m ml.train_state   # addestramento
```

### Risultato — e la conclusione onesta

Baseline: la **persistenza** ("la coda fra D minuti sarà come adesso"). È la baseline giusta
perché è esattamente ciò che fanno le app di monitoraggio esistenti.

MAE sul codice verde, 1,8 M coppie di test:

| Orizzonte | Modello | Persistenza | Guadagno |
|---|---|---|---|
| 15 min | 0.411 | 0.298 | **−38.2%** |
| 30 min | 0.505 | 0.441 | −14.5% |
| 45 min | 0.595 | 0.562 | −6.0% |
| 60 min | 0.667 | 0.661 | −0.9% |
| 90 min | 0.815 | 0.859 | **+5.1%** |
| 120 min | 0.933 | 1.019 | **+8.4%** |
| 240 min | 1.318 | 1.554 | **+15.2%** |
| 360 min | 1.609 | 1.927 | **+16.5%** |
| 720 min | 1.997 | 2.520 | **+20.7%** |

**Il modello perde sotto l'ora e vince sopra.** A 15 minuti la coda non fa in tempo a cambiare
e la fotografia attuale è già la risposta migliore; a 12 ore la persistenza è semplicemente
sbagliata, perché predice le code notturne uguali a quelle diurne.

Da qui la **strategia ibrida** in `ml/predict.py`: sotto i 75 minuti risponde la persistenza
modulata dal profilo orario, sopra risponde il modello. Un modello usato dove serve vale più
di un modello imposto ovunque.

---

## 2. Classificatore di patologia — questa sì è una rete neurale

**File**: `ml/build_index.py`, `ml/pathology_classifier.py`
**Rete**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

⚠️ **Non l'abbiamo addestrata né messa a punto**: è usata così com'è, come *encoder*. Non c'è
stato alcun fine-tuning.

### Architettura reale (dal `config.json` del modello scaricato)

| Parametro | Valore |
|---|---|
| tipo | transformer BERT-like |
| **layer (blocchi transformer)** | **12** |
| dimensione nascosta | 384 |
| teste di attenzione | 12 |
| dimensione feed-forward | 1536 |
| **funzione di attivazione** | **GELU** |
| normalizzazione | LayerNorm (ε = 1e-12) |
| dropout | 0.1 (nascosto e attenzione) |
| lunghezza massima | 512 token |
| vocabolario | 250.037 (multilingue) |
| **parametri totali** | **117.653.760** — di cui 96,2 M di embedding e 21,4 M di encoder |

Ogni blocco è: `self-attention multi-testa → LayerNorm → feed-forward (384→1536→384, GELU) → LayerNorm`,
con connessioni residue. L'output è **mean pooling mascherato** sui token, poi normalizzazione L2
così il prodotto scalare fra due vettori è il coseno.

### Perché retrieval e non fine-tuning

Il dataset (`dux-tecblic/symptom-disease-dataset`) ha **1.082 patologie su 5.634 esempi**: circa
5 casi per classe. Un classificatore a 1.082 classi su quei dati non è addestrabile, e non andava
nemmeno tentato. Il kNN sugli embedding non richiede addestramento, capisce l'italiano
nativamente (l'encoder è multilingue) e restituisce i top-k con un punteggio di similarità —
che è precisamente ciò che serve all'LLM per risolvere l'ambiguità a valle.

### Come funziona

```
offline:  5.634 frasi  → encoder → matrice 5.634 × 384 normalizzata (ml/index/index.npz)
runtime:  frase utente → encoder → vettore 384
          similarità = prodotto scalare con tutte le righe (= coseno)
          top-40 vicini → aggregati per patologia (max-similarity) → top-k candidate
```

Latenza: **6–10 ms** a query dopo il caricamento (~8 s al primo avvio).

### Lo stadio che precede la rete: le red flag

Prima del modello girano **regole deterministiche** su dolore toracico, dispnea, deficit
neurologico, alterazione di coscienza, emorragia, trauma maggiore, convulsioni, anafilassi.
Se scattano, il modello **non viene nemmeno interrogato**: si esce con "chiama il 112".

Su un'emergenza la varianza di un modello non è accettabile. Le regole usano regex tolleranti
alle parole intercalate — il match per sottostringa non trovava *"peso **fortissimo** sul petto"*,
e su una red flag il mancato riconoscimento è l'errore peggiore possibile.
Verifica attuale: **12 casi su 12, zero falsi positivi.**

---

## 3. Come DeepSeek orchestra i due modelli

DeepSeek **non contiene conoscenza del dominio**: non sa quali ospedali esistono, quanta gente
c'è in coda, né che gravità abbia una patologia. Ha solo degli *strumenti* descritti in JSON e
decide quando chiamarli (*function calling*). Il codice è in `app/agent.py`.

### I ruoli, e perché sono separati così

| Componente | Sa fare | Non sa fare |
|---|---|---|
| **Regole** (red flag) | riconoscere un'emergenza in modo deterministico | tutto il resto |
| **MiniLM** (rete neurale) | dal testo libero → patologie candidate | dire che colore sia |
| **Gradient boosting** | dalla coda attuale → coda futura | capire i sintomi |
| **DeepSeek** (LLM) | dialogare, tradurre, cercare, decidere il colore, spiegare | conoscere i dati |

Il punto chiave: **il modello dice *che patologia*, l'LLM decide *che colore*.** MiniLM non ha
mai visto codici colore italiani e non potrebbe impararli — non esistono dati etichettati in
italiano. L'LLM invece può cercare la gravità su fonti sanitarie ufficiali e motivarla.

### Il flusso completo

```
utente: "ho un forte mal di testa pulsante e la luce mi dà fastidio"
   │
   ├─1─► classify_pathology(text)              ► REGOLE + MINILM
   │        red flag? → sì: "chiama il 112", STOP, nessun altro tool
   │        no → top-k: Migraine 0.85 · Hypertension 0.82 · Dengue 0.78
   │
   ├─2─► [DeepSeek traduce: "Migraine" → "Emicrania"]
   │
   ├─3─► lookup_pathology_severity(pathology, nome_italiano)
   │        cache hit? → usa quel colore, niente rete
   │        miss → ricerca web su fonti sanitarie italiane → estratti
   │
   ├─4─► [DeepSeek ASSEGNA IL COLORE leggendo gli estratti]
   │        regole: arrotonda verso l'alto · pondera con le parole dell'utente
   │        · se incerto propone un colore provvisorio E fa domande
   │
   ├─5─► save_pathology_color(...)             ► cache, per non ricercare più
   │
   ├─6─► rank_facilities(lat, lon, colore)     ► GRADIENT BOOSTING
   │        per ogni struttura: viaggio (OSRM) → predizione a t+viaggio
   │        → attesa PER QUEL COLORE → ordina per viaggio + attesa
   │
   └─7─► show_on_map(...) e risposta motivata

output: **Colore attribuito: 🟢 VERDE** — provvisorio, da confermare
```

### I tool esposti

| Tool | Chi risponde davvero |
|---|---|
| `classify_pathology` | regole + MiniLM (locale) |
| `lookup_pathology_severity` | cache su disco, poi ricerca web |
| `save_pathology_color` | cache |
| `predict_hospital_state` | gradient boosting |
| `forecast_hospital_wait` | gradient boosting |
| `rank_facilities` | gradient boosting + OSRM |
| `list_hospitals`, `find_pharmacies_nearby`, `travel_time`, `show_on_map` | dati e servizi |

### Guardrail

- **L'emergenza esce dal flusso**: red flag → 112 e stop, nessuna ricerca, nessuna struttura.
- **Arrotonda verso l'alto**: fonti o candidate discordi → si sceglie il colore più grave fra i plausibili.
- **Il testo web è dato, non istruzione**: difesa contro prompt injection nei risultati di ricerca.
- **Incerto ≠ non rispondo**: colore provvisorio + domande.
- L'HTML è escapato **prima** del markdown nella chat: il testo passa per un LLM che legge pagine web.

### Il pannello 🔍 debug

Nell'intestazione della chat, mostra sotto ogni risposta **quale componente ha risposto a ogni
passo** e con che punteggio. Se DeepSeek risponde senza interrogare i modelli, il pannello lo
dichiara. Serve a verificarlo invece di doversi fidare.

---

## Riassunto in una riga

Una **rete neurale pre-addestrata** (MiniLM, 12 layer, GELU, 117 M parametri) trasforma i
sintomi in patologie candidate; un **gradient boosting addestrato da noi** (220 alberi, 12
feature, 9 M coppie) predice la coda futura; **DeepSeek** fa da direttore d'orchestra —
traduce, cerca, decide il colore e spiega — senza conoscere i dati, che chiede ai tool.
