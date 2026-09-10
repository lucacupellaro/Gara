"""Costruisce l'indice semantico sintomi -> patologia.

Scarica il dataset dux-tecblic/symptom-disease-dataset (5.634 frasi in prima
persona, 1.082 patologie) e ne calcola gli embedding con un encoder
MULTILINGUE locale, cosi' una domanda in italiano trova frasi in inglese.

Perche' retrieval e non fine-tuning: 1.082 classi su 5.634 esempi fanno ~5 casi
per patologia. Un classificatore a 1.082 classi su questi dati non e'
addestrabile in modo affidabile; il kNN sugli embedding non richiede training,
funziona in italiano da subito e restituisce naturalmente i top-k con un
punteggio di similarita' - che e' esattamente cio' che serve all'LLM a valle
per risolvere l'ambiguita' (vedi task-ml-attese.md sez. 6-7).

Uso:  python -m ml.build_index
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HF = "https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset/resolve/main"
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "ml" / "index"


def _download() -> tuple[Path, Path]:
    import httpx
    RAW.mkdir(parents=True, exist_ok=True)
    files = {"mapping.json": RAW / "mapping.json",
             "symptom-disease-train-dataset.csv": RAW / "symptom-disease-train.csv"}
    for remote, local in files.items():
        if local.exists():
            print(f"[skip] {local.name} gia' presente")
            continue
        print(f"[get ] {remote}")
        r = httpx.get(f"{HF}/{remote}", follow_redirects=True, timeout=120)
        r.raise_for_status()
        local.write_bytes(r.content)
    return files["mapping.json"], files["symptom-disease-train-dataset.csv"]


def _mean_pool(out, mask):
    """Mean pooling mascherato: e' cosi' che paraphrase-multilingual-MiniLM
    produce l'embedding di frase (nessuna dipendenza da sentence-transformers)."""
    tok = out.last_hidden_state
    m = mask.unsqueeze(-1).expand(tok.size()).float()
    return (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed(texts: list[str], tokenizer, model, device, batch: int = 128) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            v = _mean_pool(model(**enc), enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1)   # cosine = dot product
        vecs.append(v.cpu().numpy().astype("float32"))
        print(f"\r  embedding {min(i + batch, len(texts))}/{len(texts)}", end="")
    print()
    return np.vstack(vecs)


def main() -> None:
    map_path, csv_path = _download()

    name_to_id = json.loads(map_path.read_text(encoding="utf-8"))
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(f"[map ] {len(name_to_id)} patologie")

    with csv_path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("text") and r.get("label")]
    texts = [r["text"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype="int32")
    print(f"[data] {len(texts)} frasi, {len(set(labels.tolist()))} patologie coperte")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[enc ] {ENCODER} su {device}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(device).eval()

    mat = embed(texts, tokenizer, model, device)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "index.npz", vectors=mat, labels=labels)
    (OUT / "label_names.json").write_text(
        json.dumps({str(k): v for k, v in id_to_name.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (OUT / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"[ok  ] indice salvato in {OUT}  ({mat.shape[0]}x{mat.shape[1]})")


if __name__ == "__main__":
    main()
