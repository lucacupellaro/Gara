/* Chat UI: parla con /api/chat e applica le map_action ricevute. */
const messagesEl = document.getElementById("chat-messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sessionId = "s-" + Math.random().toString(36).slice(2);

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

/* Toggle debug: mostra CHI ha risposto a ogni passo — il modello locale
   (MiniLM/red flag) o l'LLM remoto — e cosa ha restituito. Serve a verificare
   che DeepSeek non stia decidendo da solo scavalcando il modello. */
let debugOn = localStorage.getItem("triage-debug") === "1";

function renderTrace(data) {
  const box = document.createElement("div");
  box.className = "trace";
  box.hidden = !debugOn;
  const steps = (data.trace || [])
    .map((t, i) => `<div class="trace-step"><b>${i + 1}. ${t.tool}</b>
        <span class="trace-by">${t.by}</span>
        <div class="trace-res">${escapeHtml(t.result)}</div></div>`)
    .join("");
  box.innerHTML = `<div class="trace-head">🔍 chi ha risposto</div>${
    steps || '<div class="trace-step"><i>nessun tool chiamato: ha risposto solo l\'LLM</i></div>'
  }<div class="trace-step"><b>risposta finale</b>
      <span class="trace-by">LLM remoto: ${escapeHtml(data.llm || "?")}</span></div>`;
  return box;
}



/* Markdown minimale. Si parte SEMPRE dal testo escapato: il contenuto arriva
   dall'LLM e non deve poter iniettare HTML nella pagina. */
function mdToHtml(raw) {
  const lines = escapeHtml(raw).split("\n");
  let html = "", inList = false;
  for (let line of lines) {
    const item = line.match(/^\s*[-*]\s+(.*)$/);
    const num = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    let body = item ? item[1] : num ? num[2] : line;

    body = body
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/_([^_\n]+?)_/g, "<em>$1</em>")
      .replace(/`([^`\n]+?)`/g, "<code>$1</code>");

    if (item || num) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${body}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += body.trim() ? `<p>${body}</p>` : "";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function addMsg(text, who, extraClass = "") {
  const div = document.createElement("div");
  div.className = `msg ${who} ${extraClass}`;
  if (who === "bot" && !extraClass) div.innerHTML = mdToHtml(text);
  else div.textContent = text;
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
    messagesEl.appendChild(renderTrace(data));
    messagesEl.scrollTop = messagesEl.scrollHeight;
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

const dbgBtn = document.getElementById("debug-toggle");
function syncDebug() {
  dbgBtn.textContent = debugOn ? "🔍 debug ON" : "🔍 debug";
  dbgBtn.classList.toggle("on", debugOn);
  document.querySelectorAll(".trace").forEach((el) => { el.hidden = !debugOn; });
}
dbgBtn.addEventListener("click", () => {
  debugOn = !debugOn;
  localStorage.setItem("triage-debug", debugOn ? "1" : "0");
  syncDebug();
});
syncDebug();

document.querySelectorAll("#quick-buttons .quick").forEach((btn) =>
  btn.addEventListener("click", () => send(btn.textContent))
);

addMsg(
  "Ciao! Sono l'assistente di orientamento sanitario del Lazio (prototipo dimostrativo). " +
  "Raccontami come ti senti e ti aiuto a capire dove conviene andare. " +
  "⚠️ In caso di emergenza chiama subito il 112.",
  "bot"
);
