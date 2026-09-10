/* Mappa Leaflet: ospedali colorati per saturazione + heatmap + farmacie. */
const ROME = [41.9028, 12.4964];
const BAND_COLORS = { green: "#2e9e4f", yellow: "#e0b400", orange: "#e07b00", red: "#d1342f" };

const map = L.map("map", { zoomControl: false }).setView(ROME, 9);
L.control.zoom({ position: "bottomright" }).addTo(map);

/* Basemap stilizzate (CARTO, gratuite, senza chiave). Positron è il default. */
const CARTO_ATTR = '&copy; OpenStreetMap &copy; CARTO | Dati: dati.lazio.it';
const BASEMAPS = {
  "Chiaro (Positron)": L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", { attribution: CARTO_ATTR, subdomains: "abcd", maxZoom: 20 }),
  "Scuro (Dark Matter)": L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { attribution: CARTO_ATTR, subdomains: "abcd", maxZoom: 20 }),
  "Voyager": L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", { attribution: CARTO_ATTR, subdomains: "abcd", maxZoom: 20 }),
  "OSM classica": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: '&copy; OpenStreetMap | Dati: dati.lazio.it' }),
};
BASEMAPS["Chiaro (Positron)"].addTo(map);
L.control.layers(BASEMAPS, null, { position: "topright", collapsed: true }).addTo(map);

const hospitalLayer = L.layerGroup().addTo(map);
/* Heatmap vera (leaflet.heat): intensità = affollamento pesato per gravità. */
const heatLayer = L.heatLayer([], {
  radius: 38, blur: 28, maxZoom: 12, max: 1.0, minOpacity: 0.25,
  gradient: { 0.15: "#2e9e4f", 0.4: "#e0b400", 0.65: "#e07b00", 0.9: "#d1342f" },
}).addTo(map);
const pharmacyLayer = L.layerGroup();
const markersByCode = {};
let userPosition = null;
let userMarker = null;

const CODE_LABELS = { red: "🔴 rossi", yellow: "🟡 gialli", green: "🟢 verdi", white: "⚪ bianchi" };

function popupHtml(h) {
  const rows = Object.entries(h.waiting_by_code)
    .map(([c, n]) => `<tr><td>${CODE_LABELS[c]}</td><td><b>${n}</b> in attesa</td></tr>`)
    .join("");
  return `<div class="hospital-popup">
    <h4>${h.name}</h4>
    <div>${h.type} · ${h.comune} · ASL ${h.asl}</div>
    <table>${rows}</table>
    <div>In trattamento: <b>${h.in_treatment}</b> · Saturazione: <b>${h.saturation}</b></div>
    <div style="color:#777;font-size:.78rem">${h.source} · agg. ${h.updated_at.replace("T", " ")}</div>
  </div>`;
}

async function loadHospitals() {
  const res = await fetch("/api/hospitals");
  const data = await res.json();
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
    }).bindPopup(popupHtml(h)).addTo(hospitalLayer);
    markersByCode[h.code] = m;
  }
}

async function loadPharmacies() {
  const res = await fetch("/api/pharmacies?limit=1600");
  const data = await res.json();
  pharmacyLayer.clearLayers();
  for (const p of data.pharmacies) {
    L.circleMarker([p.lat, p.lon], {
      radius: 4, color: "#0b6e4f", weight: 1, fillColor: "#19a974", fillOpacity: 0.8,
    }).bindPopup(`<b>💊 ${p.name}</b><br>${p.address}, ${p.comune}`).addTo(pharmacyLayer);
  }
}

document.getElementById("chk-pharmacies").addEventListener("change", (e) => {
  if (e.target.checked) { loadPharmacies(); map.addLayer(pharmacyLayer); }
  else map.removeLayer(pharmacyLayer);
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
  for (const m of Object.values(markersByCode)) m.setStyle({ color: "#fff", weight: 2 });
  for (const code of action.highlight || []) {
    const m = markersByCode[code];
    if (m) { m.setStyle({ color: "#111", weight: 5 }); m.openPopup(); }
  }
  if (action.center) map.setView(action.center, 12);
};
window.getUserPosition = () => userPosition;

loadHospitals();
setInterval(loadHospitals, 60_000);
