"""Agente LLM (DeepSeek, API compatibile OpenAI) con function calling.

Loop: messaggi+tools -> se tool_calls esegue le funzioni Python, appende i
risultati come role:"tool" e reinvia; max MAX_TURNS iterazioni.
Con MOCK_LLM=1 un copione fisso pilota comunque i tool reali (demo offline).
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from . import mocks, state
from .predictor import predict_hospital_state
from .scoring import find_pharmacies_nearby, rank_facilities, travel_minutes

load_dotenv()
MAX_TURNS = 8

SYSTEM_PROMPT = """Sei l'assistente dell'Emergency Triage Assistant del Lazio, un prototipo dimostrativo.
Parli italiano, tono calmo ed empatico. NON sei un medico e NON fai diagnosi definitive: usa sempre formule prudenti ("possibile", "compatibile con").

REGOLE VINCOLANTI:
1. Se i sintomi indicano una possibile emergenza (classify_symptoms ritorna red_flags=true, o l'utente descrive dolore toracico, difficoltà respiratoria grave, emorragia abbondante, perdita di coscienza, deficit di parola/movimento, trauma cranico): interrompi TUTTO e di' di chiamare SUBITO il 112. Niente ranking, niente altre domande.
2. Il codice colore va PROPOSTO e concordato con l'utente, mai imposto: spiega perché e chiedi conferma.
3. Fai al massimo 2-3 domande di follow-up prima di proporre il codice.
4. Prima di raccomandare strutture ti serve la posizione: usa quella condivisa dal browser (nel contesto) o chiedila.
5. Per codice bianco o sintomi lievi suggerisci prima la farmacia più vicina.
6. Chiedi la preferenza tra "meno_attesa", "meno_viaggio" o "bilanciato" se non è chiara.
7. Quando raccomandi strutture chiama sempre show_on_map per evidenziarle sulla mappa.
8. Ricorda quando serve che questo è un prototipo dimostrativo basato su open data, non un servizio sanitario.

Flusso tipico: ascolta i sintomi -> classify_symptoms -> eventuali follow-up -> proponi e concorda il codice -> rank_facilities (o farmacie) -> spiega la raccomandazione con attesa e viaggio -> show_on_map."""

TOOLS = [
    {"type": "function", "function": {
        "name": "classify_symptoms",
        "description": "Classifica i sintomi descritti: categoria, condizione possibile, codice colore proposto, red flag, domande di follow-up.",
        "parameters": {"type": "object", "properties": {
            "symptoms": {"type": "string", "description": "descrizione dei sintomi dell'utente"},
            "answers": {"type": "object", "description": "risposte alle domande di follow-up", "additionalProperties": True},
        }, "required": ["symptoms"]}}},
    {"type": "function", "function": {
        "name": "get_hospitals_status",
        "description": "Stato attuale di tutti i pronto soccorso del Lazio: coda per codice colore e indice di saturazione.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "predict_hospital_state",
        "description": "Prevede coda e attesa stimata di un ospedale tra N minuti (es. dopo il viaggio).",
        "parameters": {"type": "object", "properties": {
            "hospital_code": {"type": "string"},
            "minutes_ahead": {"type": "integer"},
        }, "required": ["hospital_code", "minutes_ahead"]}}},
    {"type": "function", "function": {
        "name": "travel_time",
        "description": "Minuti di viaggio in auto dalla posizione dell'utente a un ospedale.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
            "hospital_code": {"type": "string"},
        }, "required": ["lat", "lon", "hospital_code"]}}},
    {"type": "function", "function": {
        "name": "rank_facilities",
        "description": "Classifica le migliori strutture per il codice colore concordato, pesando attesa stimata all'arrivo e tempo di viaggio. Per codice bianco include anche le farmacie vicine.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
            "triage_code": {"type": "string", "enum": ["giallo", "verde", "bianco"]},
            "preference": {"type": "string", "enum": ["meno_attesa", "meno_viaggio", "bilanciato"]},
        }, "required": ["lat", "lon", "triage_code"]}}},
    {"type": "function", "function": {
        "name": "find_pharmacies_nearby",
        "description": "Le farmacie più vicine alla posizione dell'utente.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["lat", "lon"]}}},
    {"type": "function", "function": {
        "name": "show_on_map",
        "description": "Evidenzia sulla mappa le strutture raccomandate e centra la vista. Da chiamare sempre dopo una raccomandazione.",
        "parameters": {"type": "object", "properties": {
            "hospital_codes": {"type": "array", "items": {"type": "string"}},
            "center_lat": {"type": "number"}, "center_lon": {"type": "number"},
        }, "required": ["hospital_codes"]}}},
]


def _exec_tool(name: str, args: dict, ctx: dict) -> dict | list:
    """Esegue il tool richiesto dal modello. ctx raccoglie le map_action."""
    print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)[:200]})")
    if name == "classify_symptoms":
        return mocks.classify_symptoms(args.get("symptoms", ""), args.get("answers"))
    if name == "get_hospitals_status":
        return [{k: h[k] for k in ("code", "name", "type", "comune", "waiting_by_code",
                                   "total_waiting", "saturation", "saturation_band")}
                for h in state.get_all_status()]
    if name == "predict_hospital_state":
        current = state.get_status(args["hospital_code"])
        if not current:
            return {"error": f"ospedale {args['hospital_code']} non trovato"}
        return predict_hospital_state(args["hospital_code"], int(args["minutes_ahead"]), current)
    if name == "travel_time":
        h = state.get_status(args["hospital_code"])
        if not h:
            return {"error": "ospedale non trovato"}
        minutes, source = travel_minutes(args["lat"], args["lon"], h["lat"], h["lon"])
        return {"travel_minutes": minutes, "source": source}
    if name == "rank_facilities":
        return rank_facilities(args["lat"], args["lon"], args["triage_code"],
                               args.get("preference", "bilanciato"))
    if name == "find_pharmacies_nearby":
        return find_pharmacies_nearby(args["lat"], args["lon"], int(args.get("limit", 5)))
    if name == "show_on_map":
        ctx["map_action"] = {
            "highlight": args.get("hospital_codes", []),
            "center": ([args["center_lat"], args["center_lon"]]
                       if args.get("center_lat") is not None else None),
        }
        return {"ok": True}
    return {"error": f"tool sconosciuto: {name}"}


# --- sessioni in memoria (basta per la demo) ---------------------------------
_sessions: dict[str, list[dict]] = {}


def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                  base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def chat(session_id: str, message: str, position: dict | None) -> dict:
    history = _sessions.setdefault(session_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    user_msg = message
    if position and position.get("lat"):
        user_msg += (f"\n[contesto: posizione utente lat={position['lat']:.5f}, "
                     f"lon={position['lon']:.5f}]")
    history.append({"role": "user", "content": user_msg})

    ctx: dict = {"map_action": None}
    if os.getenv("MOCK_LLM", "0") == "1":
        reply = _mock_llm_reply(history, position, ctx)
    else:
        reply = _agent_loop(history, ctx)
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "map_action": ctx["map_action"]}


def _agent_loop(history: list[dict], ctx: dict) -> str:
    client = _client()
    model = os.getenv("MODEL", "deepseek-chat")
    messages = list(history)
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.5)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "…"
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _exec_tool(tc.function.name, args, ctx)
            except Exception as e:  # il modello deve poter reagire all'errore
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return ("Mi sono perso tra i dati, scusami. Se non è urgente riprova a "
            "riformulare; se pensi sia grave chiama il 112.")


# --- copione MOCK_LLM: pilota i tool reali senza API -------------------------

def _mock_llm_reply(history: list[dict], position: dict | None, ctx: dict) -> str:
    user_turns = [m for m in history if m["role"] == "user"]
    last = user_turns[-1]["content"].lower()
    cls = mocks.classify_symptoms(last)

    if cls["red_flags"]:
        return ("⚠️ Quello che descrivi può essere un'emergenza. "
                "Chiama SUBITO il 112. Non metterti alla guida da solo.")

    if len(user_turns) == 1 and cls["category"] == "non classificato":
        return ("Ciao! Sono un prototipo dimostrativo, non un servizio medico. "
                "Raccontami che sintomi hai e da quanto tempo; se è un'emergenza chiama subito il 112.")

    if not (position and position.get("lat")):
        return (f"Capisco: potrebbe trattarsi di {cls['possible_condition']} "
                f"(codice {cls['triage_code']} da confermare). Per consigliarti dove andare "
                "mi serve la tua posizione: premi «Usa la mia posizione» sulla mappa.")

    lat, lon = position["lat"], position["lon"]
    if cls["triage_code"] == "bianco":
        pharm = find_pharmacies_nearby(lat, lon, 3)
        nearest = pharm[0]
        return (f"Per un disturbo lieve come questo ({cls['possible_condition']}, codice bianco) "
                f"ti consiglio prima una farmacia: la più vicina è {nearest['name']} "
                f"in {nearest['address']} a {nearest['distance_km']} km. "
                "Se i sintomi peggiorano, sentiamoci di nuovo.")

    ranking = rank_facilities(lat, lon, cls["triage_code"], "bilanciato")
    if not ranking.get("hospitals"):
        return "Non trovo strutture raggiungibili entro 90 minuti dalla tua posizione."
    best = ranking["hospitals"][0]
    ctx["map_action"] = {"highlight": [h["code"] for h in ranking["hospitals"]],
                         "center": [best["lat"], best["lon"]]}
    others = "; ".join(f"{h['name']} ({h['reason']})" for h in ranking["hospitals"][1:])
    return (f"Potrebbe trattarsi di {cls['possible_condition']}: proporrei un codice "
            f"{cls['triage_code']} (confermamelo tu). La struttura che ti consiglio è "
            f"{best['name']} a {best['comune']}: {best['reason']}. "
            f"Alternative: {others}. Le ho evidenziate sulla mappa. "
            "Ricorda: sono un prototipo, in caso di dubbio chiama il 112.")
