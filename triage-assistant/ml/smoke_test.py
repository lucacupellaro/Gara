"""Simula la sequenza di tool che DeepSeek eseguira', senza chiamare l'API.
Serve a verificare la catena locale: classify_pathology -> lookup -> save.
Uso:  python -m ml.smoke_test
"""
from app.agent import _exec_tool

CASI = [
    "ho un peso fortissimo sul petto e sudo freddo",
    "ho un forte mal di testa pulsante e la luce mi da fastidio",
    "ho mal di gola e febbre da due giorni",
]
# traduzioni che nella catena reale produce l'LLM
IT = {"Migraine": "Emicrania", "Common Cold": "Raffreddore comune",
      "Bronchial Asthma": "Asma bronchiale", "Allergy": "Allergia",
      "Hypertension": "Ipertensione", "Typhoid": "Tifo"}


def main() -> None:
    ctx = {"map_action": None}
    for testo in CASI:
        print(f"\n{'=' * 74}\nUTENTE: {testo}")
        cls = _exec_tool("classify_pathology", {"text": testo, "top_k": 3}, ctx)

        if cls["emergency"]:
            print(f"  1. classify_pathology -> EMERGENZA {cls['matched_flags']}")
            print("  2. STOP: l'LLM dice 112. Nessuna ricerca, nessun altro tool.")
            print("  ESITO: rosso (source=rule)")
            continue

        cand = cls["candidates"][0]
        print(f"  1. classify_pathology -> {[c['name'] for c in cls['candidates']]}"
              f"  (uncertain={cls['uncertain']})")
        sev = _exec_tool("lookup_pathology_severity",
                         {"pathology": cand["name"],
                          "nome_italiano": IT.get(cand["name"], cand["name"])}, ctx)
        print(f"  2. lookup_pathology_severity -> source={sev['source']}")
        for r in sev.get("results", [])[:2]:
            mark = "OK" if r["fonte_attendibile"] else "  "
            print(f"       [{mark}] {r['titolo'][:58]}")
        if sev["source"] == "cache":
            print(f"       (cache hit: codice={sev.get('codice')}, nessuna rete)")
            continue
        # qui, nella catena reale, decide l'LLM leggendo gli estratti
        colore = "verde"
        fonte = sev["results"][0]["url"] if sev.get("results") else ""
        print(f"  3. [decisione LLM] colore = {colore}")
        print(f"  4. save_pathology_color -> "
              f"{_exec_tool('save_pathology_color', {'pathology': cand['name'], 'codice': colore, 'fonte': fonte}, ctx)}")

    print(f"\n{'=' * 74}\nRi-esecuzione: la cache deve evitare la rete")
    r = _exec_tool("lookup_pathology_severity", {"pathology": "Migraine"}, ctx)
    print(f"  Migraine -> source={r['source']}  codice={r.get('codice')}")


if __name__ == "__main__":
    main()
