from fastapi import APIRouter
from pydantic import BaseModel

from . import agent

router = APIRouter(prefix="/api")


class ChatBody(BaseModel):
    session_id: str
    message: str
    position: dict | None = None


@router.post("/chat")
def chat(body: ChatBody):
    return agent.chat(body.session_id, body.message, body.position)
