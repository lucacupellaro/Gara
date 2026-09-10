from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

from . import routes_api, routes_chat  # noqa: E402 (dopo load_dotenv)

app = FastAPI(title="Emergency Triage Assistant — Lazio")
app.include_router(routes_api.router)
app.include_router(routes_chat.router)

STATIC = Path(__file__).parent.parent / "static"
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
