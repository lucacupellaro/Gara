"""Scarica i dataset open data necessari in data/raw/ (idempotente)."""
from pathlib import Path

import httpx

RAW = Path(__file__).parent / "raw"

SOURCES = {
    # Snapshot accessi PS Lazio (dati.lazio.it, latin-1, sep ';')
    "ps_lazio.csv": "https://dati.lazio.it/dataset/144e577e-8a7e-4613-9830-48cbb1d7ee0f/resource/12c31624-f1a4-4874-a903-8954549ddb81/download/output_1627742164504.csv",
    # Farmacie Regione Lazio con lat/lon (dati.lazio.it)
    "farmacie_lazio.csv": "https://dati.lazio.it/dataset/551579e1-a65d-4e7e-80d7-9bfe07fb2bdc/resource/7658322d-b629-4d77-a9f1-e4aad7c8f83b/download/farmaciereglaziolatlon.csv",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[download] {name} già presente, salto")
            continue
        print(f"[download] {name} <- {url}")
        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"[download] {name}: {len(r.content)} byte")


if __name__ == "__main__":
    main()
