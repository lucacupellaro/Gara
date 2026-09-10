/* Mappa Leaflet: ospedali colorati per saturazione + heatmap + farmacie. */
const ROME = [41.9028, 12.4964];
const BAND_COLORS = { green: "#2e9e4f", yellow: "#e0b400", orange: "#e07b00", red: "#d1342f" };

const map = L.map("map", { zoomControl: false }).setView(ROME, 9);
L.control.zoom({ position: "bottomright" }).addTo(map);

/* Basemap stilizzate CARTO (richiedono la chiave, servita da /api/config).
   Fallback OSM se la chiave manca. Positron è il default. */
const CARTO_ATTR = '&copy; OpenStreetMap &copy; CARTO | Dati: dati.lazio.it';
const OSM_LAYER = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; OpenStreetMap | Dati: dati.lazio.it',
});

fetch("/api/config")
  .then((r) => r.json())
  .then(({ map_key }) => {
    if (!map_key) { OSM_LAYER.addTo(map); return; }
    const carto = (path) =>
      L.tileLayer(`https://{s}.basemaps.cartocdn.com/${path}/{z}/{x}/{y}{r}.png?key=${map_key}`,
        { attribution: CARTO_ATTR, subdomains: "abcd", maxZoom: 20 });
    const BASEMAPS = {
      "Chiaro (Positron)": carto("light_all"),
      "Scuro (Dark Matter)": carto("dark_all"),
      "Voyager": carto("rastertiles/voyager"),
      "OSM classica": OSM_LAYER,
    };
    BASEMAPS["Chiaro (Positron)"].addTo(map);
    L.control.layers(BASEMAPS, null, { position: "topright", collapsed: true }).addTo(map);
  })
  .catch(() => OSM_LAYER.addTo(map));

const hospitalLayer = L.layerGroup().addTo(map);
/* Heatmap vera (leaflet.heat): intensità = affollamento pesato per gravità. */
const heatLayer = L.heatLayer([], {
  radius: 38, blur: 28, maxZoom: 12, max: 1.0, minOpacity: 0.25,
  gradient: { 0.15: "#2e9e4f", 0.4: "#e0b400", 0.65: "#e07b00", 0.9: "#d1342f" },
}).addTo(map);
/* pane dedicato sopra tutto il resto: così il canvas della heatmap non
   "mangia" i click sulle farmacie (era questo il bug) */
map.createPane("pharmacies");
map.getPane("pharmacies").style.zIndex = 650;
const pharmacyRenderer = L.svg({ pane: "pharmacies" });
const pharmacyLayer = L.layerGroup();
const rangeLayer = L.layerGroup().addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const markersByCode = {};
let hospitalsData = [];
let selectedCode = null;
let userPosition = null;
let userMarker = null;

const CODE_LABELS = { red: "🔴 rossi", yellow: "🟡 gialli", green: "🟢 verdi", white: "⚪ bianchi" };

function haversineKm(lat1, lon1, lat2, lon2) {
  const rad = Math.PI / 180, R = 6371;
  const dLat = (lat2 - lat1) * rad, dLon = (lon2 - lon1) * rad;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function infoHtml(h) {
  const rows = Object.entries(h.waiting_by_code)
    .map(([c, n]) => `<tr><td>${CODE_LABELS[c]}</td><td><b>${n}</b> in attesa</td></tr>`)
    .join("");
  return `<div class="hospital-popup">
    <h4>${h.name}</h4>
    <div>${h.type} · ${h.comune} · ASL ${h.asl}</div>
    <table>${rows}</table>
    <div>In trattamento: <b>${h.in_treatment}</b> · Saturazione: <b>${h.saturation}</b></div>
    <div style="color:#777;font-size:.78rem">${h.source} · agg. ${h.updated_at.replace("T", " ")}</div>
    <button class="btn-travel" data-code="${h.code}">🚗 Calcola viaggio</button>
    <div class="travel-result" id="travel-${h.code}"></div>
  </div>`;
}

/* --- selezione ospedale: pannello in alto a sinistra + marker evidenziato -- */

function highlightMarker(code, on) {
  const m = markersByCode[code];
  if (!m) return;
  m.setStyle(on ? { color: "#111", weight: 4 } : { color: "#fff", weight: 2 });
  if (m._path) m._path.classList.toggle("highlighted", on);
}

function selectHospital(h) {
  if (selectedCode) highlightMarker(selectedCode, false);
  selectedCode = h.code;
  highlightMarker(h.code, true);

  const panel = document.getElementById("info-panel");
  panel.innerHTML = `<span class="close" title="chiudi">✕</span>` + infoHtml(h);
  panel.hidden = false;
  panel.querySelector(".close").addEventListener("click", () => {
    panel.hidden = true;
    highlightMarker(selectedCode, false);
    selectedCode = null;
  });
  panel.querySelector(".btn-travel").addEventListener("click", () => calcTravel(h.code));
}

function selectPharmacy(p) {
  if (selectedCode) { highlightMarker(selectedCode, false); selectedCode = null; }
  const dist = userPosition
    ? ` · ${haversineKm(userPosition.lat, userPosition.lon, p.lat, p.lon).toFixed(1)} km da te`
    : "";
  const panel = document.getElementById("info-panel");
  panel.innerHTML = `<span class="close" title="chiudi">✕</span>
    <div class="hospital-popup">
      <h4>💊 ${p.name}</h4>
      <div>${p.address}, ${p.comune}${dist}</div>
      <button class="btn-travel">🚗 Calcola viaggio</button>
      <div class="travel-result" id="travel-pharm"></div>
    </div>`;
  panel.hidden = false;
  panel.querySelector(".close").addEventListener("click", () => { panel.hidden = true; });
  panel.querySelector(".btn-travel").addEventListener("click", () =>
    calcTravelCoords(p.lat, p.lon, p.name, "travel-pharm"));
}

/* --- calcolo viaggio -------------------------------------------------------- */

function ensurePosition() {
  // usa la posizione già nota, altrimenti la chiede al browser, altrimenti Roma centro
  return new Promise((resolve) => {
    if (userPosition) return resolve(userPosition);
    if (!navigator.geolocation)
      return resolve({ lat: ROME[0], lon: ROME[1], fallback: true });
    navigator.geolocation.getCurrentPosition(
      (p) => {
        userPosition = { lat: p.coords.latitude, lon: p.coords.longitude };
        updateNearest();
        resolve(userPosition);
      },
      () => resolve({ lat: ROME[0], lon: ROME[1], fallback: true }),
      { timeout: 5000 }
    );
  });
}

function hideRouteBanner() {
  document.getElementById("route-banner").hidden = true;
  routeLayer.clearLayers();
}

function showRouteBanner(html) {
  const banner = document.getElementById("route-banner");
  banner.innerHTML =
    `<span class="route-text">${html}</span>` +
    `<button type="button" class="close" title="chiudi (Esc)" aria-label="chiudi">✕</button>`;
  banner.hidden = false;
  banner.querySelector(".close").addEventListener("click", hideRouteBanner);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !document.getElementById("route-banner").hidden) hideRouteBanner();
});

function calcTravel(code) {
  return runTravel(`hospital=${code}`, `travel-${code}`);
}

function calcTravelCoords(lat, lon, name, outId) {
  return runTravel(
    `dest_lat=${lat}&dest_lon=${lon}&dest_name=${encodeURIComponent(name)}`, outId);
}

async function runTravel(destQuery, outId) {
  const out = document.getElementById(outId);
  if (out) out.textContent = "calcolo il percorso…";
  const pos = await ensurePosition();
  try {
    const res = await fetch(`/api/travel?lat=${pos.lat}&lon=${pos.lon}&${destQuery}`);
    if (!res.ok) throw new Error();
    const d = await res.json();
    routeLayer.clearLayers();
    const line = L.polyline(d.geometry, {
      color: "#1971c2", weight: 4, opacity: 0.85,
      dashArray: d.source === "haversine" ? "6 8" : null,
    }).addTo(routeLayer);
    // etichetta fissa sul percorso + banner in alto sulla mappa
    line.bindTooltip(`🚗 ${d.travel_minutes} min`, {
      permanent: true, direction: "center", className: "route-label",
    }).openTooltip(d.geometry[Math.floor(d.geometry.length / 2)]);
    const waitTxt = d.est_wait_minutes
      ? ` · attesa all'arrivo ~${d.est_wait_minutes.green} min` : "";
    showRouteBanner(
      `🚗 <b>${d.travel_minutes} min</b> → ${d.destination}${waitTxt}`);
    map.fitBounds(L.latLngBounds(d.geometry).pad(0.2));
    if (out)
      out.innerHTML =
        `🚗 <b>${d.travel_minutes} min</b> di viaggio in auto` +
        (pos.fallback ? " <i>(da Roma centro — premi 📍 per la tua posizione)</i>" : "") +
        (d.est_wait_minutes
          ? `<br>attesa stimata all'arrivo: <b>~${d.est_wait_minutes.green} min</b> (codice verde)`
          : "");
  } catch {
    if (out) out.textContent = "errore nel calcolo, riprova tra poco";
  }
}

/* --- pannello "più vicini" in basso a sinistra ------------------------------ */

async function updateNearest() {
  const panel = document.getElementById("nearest-panel");
  if (!hospitalsData.length) return;
  const pos = userPosition || { lat: ROME[0], lon: ROME[1], fallback: true };

  let best = null;
  for (const h of hospitalsData) {
    const km = haversineKm(pos.lat, pos.lon, h.lat, h.lon);
    if (!best || km < best.km) best = { h, km };
  }

  let pharmRow = "", nearestPharm = null;
  try {
    const r = await fetch(`/api/pharmacies?lat=${pos.lat}&lon=${pos.lon}&limit=1`);
    nearestPharm = (await r.json()).pharmacies[0];
    pharmRow = `<div class="nearest-row" data-pharm="1">
      💊 ${nearestPharm.name}<br><small>${nearestPharm.address}, ${nearestPharm.comune} · ${nearestPharm.distance_km} km</small></div>`;
  } catch { /* farmacia non disponibile: mostra solo l'ospedale */ }

  panel.innerHTML =
    `<b>Più vicini ${pos.fallback ? "(da Roma centro)" : "a te"}</b>
     <div class="nearest-row" data-code="${best.h.code}">
       🏥 ${best.h.name}<br><small>${best.h.comune} · ${best.km.toFixed(1)} km ·
       saturazione ${best.h.saturation_band}</small></div>` + pharmRow;
  panel.hidden = false;

  panel.querySelectorAll(".nearest-row").forEach((row) =>
    row.addEventListener("click", () => {
      if (row.dataset.code) {
        const h = hospitalsData.find((x) => x.code === row.dataset.code);
        if (h) { selectHospital(h); map.setView([h.lat, h.lon], 13); }
      } else if (nearestPharm) {
        selectPharmacy(nearestPharm);
        map.setView([nearestPharm.lat, nearestPharm.lon], 15);
      }
    })
  );
}

/* --- caricamento dati -------------------------------------------------------- */

/* Slider temporale: posizioni -> minuti nel futuro. 0 = adesso.
   I valori oltre le 2 ore sono quelli su cui il modello batte davvero la
   persistenza (+15% a 4h, +21% a 12h), quindi e' li' che lo slider mostra
   qualcosa che una semplice fotografia non saprebbe dire. */
const TIME_STEPS = [0, 30, 60, 120, 240, 360, 720];
const TIME_LABELS = ["adesso", "+30 min", "+1 ora", "+2 ore", "+4 ore", "+6 ore", "+12 ore"];
let forecastMinutes = 0;

async function loadHospitals() {
  const url = forecastMinutes > 0
    ? `/api/forecast?minutes=${forecastMinutes}&triage_code=verde`
    : "/api/hospitals";
  const res = await fetch(url);
  const data = await res.json();
  updateTimePanel(data);
  hospitalsData = data.hospitals;
  hospitalLayer.clearLayers();
  // intensità relativa al PS più carico del momento (coda pesata per gravità)
  const load = (h) => {
    const w = h.waiting_by_code;
    return 4 * (w.red || 0) + 2 * (w.yellow || 0) + (w.green || 0) + 0.5 * (w.white || 0);
  };
  const maxLoad = Math.max(1, ...data.hospitals.map(load));
  heatLayer.setLatLngs(
    data.hospitals.map((h) => [h.lat, h.lon, Math.max(0.08, load(h) / maxLoad)])
  );
  for (const h of data.hospitals) {
    const m = L.circleMarker([h.lat, h.lon], {
      radius: 7 + Math.min(10, h.total_waiting / 3),
      color: "#fff", weight: 2,
      fillColor: BAND_COLORS[h.saturation_band], fillOpacity: 0.95,
    }).on("click", () => selectHospital(h)).addTo(hospitalLayer);
    markersByCode[h.code] = m;
  }
  if (selectedCode) highlightMarker(selectedCode, true); // sopravvive al refresh 60s
  updateNearest();
}

function pharmacyCenter() {
  // centro del raggio: la posizione utente se nota, altrimenti il centro mappa
  if (userPosition) return { lat: userPosition.lat, lon: userPosition.lon };
  const c = map.getCenter();
  return { lat: c.lat, lon: c.lng };
}

async function loadPharmacies() {
  const c = pharmacyCenter();
  const radiusKm = parseFloat(document.getElementById("rng-pharmacies").value);
  const res = await fetch(
    `/api/pharmacies?lat=${c.lat}&lon=${c.lon}&radius_km=${radiusKm}&limit=800`);
  const data = await res.json();
  pharmacyLayer.clearLayers();
  for (const p of data.pharmacies) {
    L.circleMarker([p.lat, p.lon], {
      renderer: pharmacyRenderer,
      radius: 7, color: "#0b6e4f", weight: 1.5, fillColor: "#19a974", fillOpacity: 0.9,
      bubblingMouseEvents: false,
    }).on("click", (ev) => { L.DomEvent.stop(ev); selectPharmacy(p); })
      .addTo(pharmacyLayer);
  }
  map.addLayer(pharmacyLayer);
  // alone che mostra il raggio di ricerca (layer separato, non intercetta i click)
  rangeLayer.clearLayers();
  L.circle([c.lat, c.lon], {
    radius: radiusKm * 1000, color: "#19a974", weight: 1,
    fillColor: "#19a974", fillOpacity: 0.06, interactive: false,
  }).addTo(rangeLayer);
  document.getElementById("rng-pharmacies-val").textContent =
    `${radiusKm} km · ${data.pharmacies.length}`;
}

document.getElementById("chk-pharmacies").addEventListener("change", (e) => {
  document.getElementById("pharm-range").hidden = !e.target.checked;
  if (e.target.checked) {
    loadPharmacies();
  } else {
    map.removeLayer(pharmacyLayer);
    rangeLayer.clearLayers();
  }
});

document.getElementById("rng-pharmacies").addEventListener("input", (e) => {
  document.getElementById("rng-pharmacies-val").textContent = `${e.target.value} km`;
});
document.getElementById("rng-pharmacies").addEventListener("change", () => {
  if (document.getElementById("chk-pharmacies").checked) loadPharmacies();
});

document.getElementById("chk-heatmap").addEventListener("change", (e) => {
  if (e.target.checked) map.addLayer(heatLayer);
  else map.removeLayer(heatLayer);
});

document.getElementById("btn-locate").addEventListener("click", () => {
  const setPos = (lat, lon, label) => {
    userPosition = { lat, lon };
    if (userMarker) userMarker.remove();
    userMarker = L.marker([lat, lon]).addTo(map).bindPopup(label).openPopup();
    map.setView([lat, lon], 12);
    updateNearest();
    if (document.getElementById("chk-pharmacies").checked) loadPharmacies();
  };
  if (!navigator.geolocation) return setPos(...ROME, "Posizione predefinita (Roma)");
  navigator.geolocation.getCurrentPosition(
    (pos) => setPos(pos.coords.latitude, pos.coords.longitude, "📍 Sei qui"),
    () => setPos(ROME[0], ROME[1], "Posizione predefinita (Roma centro)"),
    { timeout: 5000 }
  );
});

/* API usata dalla chat per evidenziare le raccomandazioni */
window.applyMapAction = function (action) {
  if (!action) return;
  for (const code of Object.keys(markersByCode)) highlightMarker(code, false);
  for (const code of action.highlight || []) highlightMarker(code, true);
  const first = hospitalsData.find((h) => h.code === (action.highlight || [])[0]);
  if (first) selectHospital(first);
  if (action.center) map.setView(action.center, 12);
};
window.getUserPosition = () => userPosition;

loadHospitals();
setInterval(loadHospitals, 60_000);


/* ---------- slider temporale ---------- */
function updateTimePanel(data) {
  const lab = document.getElementById("tp-label");
  const meta = document.getElementById("tp-meta");
  const tot = document.getElementById("tp-tot");
  if (!lab) return;
  const i = TIME_STEPS.indexOf(forecastMinutes);
  lab.textContent = TIME_LABELS[i < 0 ? 0 : i];
  const somma = (data.hospitals || []).reduce((a, h) => a + (h.total_waiting || 0), 0);
  tot.textContent = `${somma} persone in attesa nel Lazio`;
  if (forecastMinutes === 0) {
    meta.textContent = "stato osservato";
    meta.className = "tp-meta obs";
  } else {
    const m = data.method === "modello" ? "modello ML" : "profilo orario";
    meta.textContent = `previsione · ${m}`;
    meta.className = "tp-meta pred";
  }
  document.getElementById("time-panel").classList.toggle("forecasting", forecastMinutes > 0);
}

const rngTime = document.getElementById("rng-time");
if (rngTime) {
  rngTime.addEventListener("input", () => {
    forecastMinutes = TIME_STEPS[Number(rngTime.value)];
    const i = TIME_STEPS.indexOf(forecastMinutes);
    document.getElementById("tp-label").textContent = TIME_LABELS[i];
    loadHospitals();
  });
}
