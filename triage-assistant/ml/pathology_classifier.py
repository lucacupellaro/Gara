"""Classificatore locale sintomi -> patologia (task-ml-attese.md sez. 6).

Due stadi, nell'ordine:
  A) RED FLAG deterministiche: se scattano si esce subito con emergency=True e
     il modello NON viene interrogato. Le regole non sono mai bypassabili.
  B) Retrieval semantico: la frase dell'utente viene confrontata con 5.634
     descrizioni di sintomi tramite un encoder multilingue locale; le patologie
     dei vicini piu' simili sono le candidate.

NON restituisce il codice colore: quello lo determina l'LLM (sez. 7), perche'
il colore dipende anche da come sta l'utente adesso, non solo dalla patologia.
"""
from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).resolve().parent / "index"
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Stadio A: red flag ------------------------------------------------------
# Sinonimi colloquiali italiani, non terminologia medica: l'utente scrive come
# parla. Fonte dei criteri: protocollo regionale "Triage 5 codici" (Lazio 2019).
RED_FLAGS: dict[str, list[str]] = {
    "dolore toracico": ["dolore al petto", "dolore toracico", "oppressione al petto",
                        "peso sul petto", "fitta al cuore", "stretta al petto",
                        "male al petto", "bruciore al petto"],
    "dispnea grave": ["non riesco a respirare", "non respiro", "difficolta a respirare",
                      "difficoltà a respirare", "faccio fatica a respirare", "soffoc",
                      "mi manca il fiato", "fame d'aria"],
    "deficit neurologico": ["non riesco a parlare", "non muovo", "paralisi",
                            "bocca storta", "si è storta la bocca", "faccia storta",
                            "braccio non risponde", "formicolio a mezzo corpo",
                            "vedo doppio improvvis"],
    "alterazione coscienza": ["svenut", "svenimento", "incoscien", "perdita di coscienza",
                              "confusione improvvisa", "non risponde", "non si sveglia"],
    "emorragia": ["emorragia", "sangue abbondante", "perdo molto sangue",
                  "sanguina tanto", "non si ferma il sangue", "vomito sangue"],
    "trauma maggiore": ["trauma cranico", "battuto la testa", "caduto dall'alto",
                        "incidente stradale", "schiacciato"],
    "convulsioni": ["convulsion", "crisi epilettica", "scosse in tutto il corpo"],
    "anafilassi": ["gonfiore alla gola", "lingua gonfia", "labbra gonfie",
                   "shock anafilattico", "puntura e gonfiore"],
}

_lock = threading.Lock()
_state: dict = {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


@lru_cache(maxsize=1)
def _flag_patterns() -> tuple:
    """Compila le keyword in regex tolleranti alle parole intercalate.

    Il match per sottostringa non basta: "peso sul petto" non trova "peso
    FORTISSIMO sul petto", e su una red flag un mancato match e' l'errore
    peggiore possibile. Fra una parola e l'altra si ammettono fino a 2 parole
    (avverbi e intensificatori: "molto", "fortissimo", "da due ore").
    """
    gap = r"\s+(?:\w+\s+){0,2}"
    out = []
    for flag, kws in RED_FLAGS.items():
        pats = []
        for kw in kws:
            parts = [re.escape(w) for w in kw.split()]
            pats.append(gap.join(parts))
        out.append((flag, re.compile("|".join(pats))))
    return tuple(out)


def check_red_flags(text: str) -> list[str]:
    """Ritorna i nomi delle red flag attivate. Vuoto = nessuna emergenza rilevata."""
    t = _normalize(text)
    return [flag for flag, rx in _flag_patterns() if rx.search(t)]


# --- Stadio B: retrieval -----------------------------------------------------

def _load():
    """Carica indice ed encoder una sola volta (thread-safe, lazy)."""
    with _lock:
        if _state:
            return _state
        import torch
        from transformers import AutoModel, AutoTokenizer

        blob = np.load(INDEX_DIR / "index.npz")
        names = json.loads((INDEX_DIR / "label_names.json").read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state.update(
            vectors=blob["vectors"], labels=blob["labels"], names=names, device=device,
            torch=torch,
            tokenizer=AutoTokenizer.from_pretrained(ENCODER),
            model=AutoModel.from_pretrained(ENCODER).to(device).eval(),
        )
        return _state


def _embed_query(text: str) -> np.ndarray:
    st = _load()
    torch = st["torch"]
    enc = st["tokenizer"]([text], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(st["device"])
    with torch.no_grad():
        out = st["model"](**enc)
        tok, mask = out.last_hidden_state, enc["attention_mask"]
        m = mask.unsqueeze(-1).expand(tok.size()).float()
        v = (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v.cpu().numpy().astype("float32")[0]


@lru_cache(maxsize=512)
def _neighbours(text: str, k: int = 40) -> tuple:
    st = _load()
    sims = st["vectors"] @ _embed_query(text)          # coseno: vettori normalizzati
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return tuple((int(st["labels"][i]), float(sims[i])) for i in idx)


def classify_pathology(text: str, top_k: int = 5) -> dict:
    """Sintomi in linguaggio naturale -> patologie candidate.

    Non ritorna il codice colore (lo decide l'LLM): ritorna le candidate con la
    loro plausibilita', piu' l'esito delle red flag che ha precedenza assoluta.
    """
    flags = check_red_flags(text)
    if flags:
        return {
            "emergency": True,
            "code": "rosso",
            "matched_flags": flags,
            "candidates": [],
            "top_prob": 1.0,
            "uncertain": False,
            "raw_text": text,
            "source": "rule",
            "note": "Red flag attivate: l'LLM deve indicare il 112 e fermarsi.",
        }

    st = _load()
    agg: dict[int, float] = {}
    for label, sim in _neighbours(_normalize(text)):
        # max-similarity per patologia: una sola frase molto simile vale piu'
        # di molte frasi vagamente simili
        agg[label] = max(agg.get(label, 0.0), sim)

    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k]
    total = sum(s for _, s in ranked) or 1.0
    candidates = [{"label": lab,
                   "name": st["names"].get(str(lab), f"label_{lab}"),
                   "similarity": round(sim, 4),
                   "prob": round(sim / total, 4)} for lab, sim in ranked]

    top_sim = candidates[0]["similarity"] if candidates else 0.0
    gap = (top_sim - candidates[1]["similarity"]) if len(candidates) > 1 else 1.0
    return {
        "emergency": False,
        "code": None,
        "matched_flags": [],
        "candidates": candidates,
        "top_prob": round(top_sim, 4),
        # incerto se nulla assomiglia davvero, o se le prime due sono appaiate
        "uncertain": bool(top_sim < 0.45 or gap < 0.03),
        "raw_text": text,
        "source": "model",
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "ho mal di gola e febbre da due giorni"
    print(json.dumps(classify_pathology(q), ensure_ascii=False, indent=2))
