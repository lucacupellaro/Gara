/* Chat UI: parla con /api/chat e applica le map_action ricevute. */
const messagesEl = document.getElementById("chat-messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sessionId = "s-" + Math.random().toString(36).slice(2);

function addMsg(text, who, extraClass = "") {
  const div = document.createElement("div");
  div.className = `msg ${who} ${extraClass}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function send(text) {
  addMsg(text, "user");
  const thinking = addMsg("sto pensando…", "bot", "thinking");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        position: window.getUserPosition ? window.getUserPosition() : null,
      }),
    });
    const data = await res.json();
    thinking.remove();
    addMsg(data.reply, "bot");
    if (data.map_action && window.applyMapAction) window.applyMapAction(data.map_action);
  } catch (err) {
    thinking.remove();
    addMsg("Errore di connessione col server. Riprova tra poco.", "bot");
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  send(text);
});

document.querySelectorAll("#quick-buttons .quick").forEach((btn) =>
  btn.addEventListener("click", () => send(btn.textContent))
);

addMsg(
  "Ciao! Sono l'assistente di orientamento sanitario del Lazio (prototipo dimostrativo). " +
  "Raccontami come ti senti e ti aiuto a capire dove conviene andare. " +
  "⚠️ In caso di emergenza chiama subito il 112.",
  "bot"
);
