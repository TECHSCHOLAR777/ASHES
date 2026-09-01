const $ = (sel, el = document) => el.querySelector(sel);
const app = $("#app");
const ACTION_ORDER = ["evacuate_site", "protect_asset", "prepare", "inspect_after", "monitor", "no_action"];
const FLAG_LABEL = {
  perimeter_unofficial: "Unofficial perimeter",
  no_ros_high_sigma: "Evacuate suppressed (high sigma)",
  fhsz_missing: "FHSZ missing",
  firms_only: "FIRMS only",
  stale_E: "Stale weather/E",
  aoi_coarse_advisory: "AOI coarsened (advisory)",
  aoi_isotropic_buffer: "AOI buffer isotropic (no wind)",
  aoi_empty: "AOI empty",
  firms_unavailable: "FIRMS unavailable",
  degraded: "Degraded inputs",
};

let lastReport = null;
let mapRef = null;
let playbackOverlay = null;
let playbackTimer = null;
let playbackHour = 0;

function route() {
  const h = (location.hash || "#/").replace(/^#/, "") || "/";
  document.querySelectorAll("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === h || (h.startsWith("/card") && a.dataset.route === "/ask"));
  });
  if (h === "/" || h === "") return renderWatch();
  if (h === "/ask") return renderAsk(false);
  if (h === "/simulate") return renderAsk(true);
  if (h === "/audit") return renderAudit();
  if (h.startsWith("/card/")) return renderCard(h.split("/")[2]);
  if (h.startsWith("/incidents/")) return renderIncident(decodeURIComponent(h.split("/")[2]));
  return renderWatch();
}

function na(v) {
  if (v === null || v === undefined || v === "") return '<span class="na">n/a</span>';
  return String(v);
}

function fmtKm(m) {
  if (m === null || m === undefined) return '<span class="na">n/a</span>';
  const n = Number(m);
  if (!Number.isFinite(n)) return '<span class="na">n/a</span>';
  if (n >= 1000) return `${(n / 1000).toFixed(1)} km`;
  return `${Math.round(n)} m`;
}

function pill(action) {
  if (!action) return '<span class="na">no card</span>';
  return `<span class="pill ${action}">${action.replaceAll("_", " ").toUpperCase()}</span>`;
}

function chips(flags) {
  return (flags || []).map((f) => `<span class="chip">${FLAG_LABEL[f] || f}</span>`).join(" ");
}

async function loadHealth() {
  const el = $("#health");
  try {
    const h = await (await fetch("/api/health")).json();
    const bits = [
      h.openai ? "OpenAI NLP" : "no OpenAI",
      h.mireye ? "Mireye" : "no Mireye",
      h.spread_up ? `engine :${h.spread_port}` : "engine down",
      h.elmfire_bin ? "ELMFIRE" : "no ELMFIRE bin",
    ];
    el.textContent = bits.join(" · ");
    el.className = "health " + (h.ok && h.openai && h.mireye ? "ok" : "bad");
    el.title = JSON.stringify(h);
  } catch (e) {
    el.textContent = "health failed";
    el.className = "health bad";
  }
}

async function renderWatch() {
  const recent = await (await fetch("/api/recent")).json();
  const recentHtml = (recent.cards || []).length
    ? `<h2>Recent asks</h2><table class="table"><thead><tr><th>Site</th><th>Action</th><th>Engine</th></tr></thead><tbody>${(recent.cards || [])
        .map(
          (c) => `<tr data-card="${c.card_id}">
        <td>${c.site?.name || ""}</td><td>${pill(c.action)}</td>
        <td class="muted">${c.spread_field_version || "no field"}</td></tr>`
        )
        .join("")}</tbody></table>`
    : "";
  app.innerHTML = `<h1>Watch board</h1><p class="muted">Policy bucket, engine ETA, and P(burn) at each community pin.</p>${recentHtml}<div id="board">Loading…</div>`;
  const data = await (await fetch("/api/sites")).json();
  const rows = (data.sites || []).sort((a, b) => ACTION_ORDER.indexOf(a.action) - ACTION_ORDER.indexOf(b.action));
  if (!rows.length) {
    $("#board").innerHTML = `<div class="empty">No sites in config/sites.yaml.</div>`;
    return;
  }
  $("#board").innerHTML = `<table class="table"><thead><tr>
    <th>Site</th><th>Action</th><th>ETA (σ)</th><th>P(burn 72)</th><th>Distance</th><th>Engine</th><th>Weather</th>
  </tr></thead><tbody>${rows
    .map(
      (s) => `<tr data-card="${s.last_card_id || ""}" data-site="${s.site_id}">
      <td><strong>${s.name}</strong><div class="muted">${s.site_id}</div></td>
      <td>${pill(s.action)}</td>
      <td>${etaCell(s)}</td>
      <td>${p72Cell(s)}</td>
      <td>${fmtKm(s.dist_perimeter_m)}<div class="muted">${s.incident_name || "no incident"}</div></td>
      <td class="muted">${s.engine || s.spread_field_version || "no field"}</td>
      <td>${s.red_flag ? "Red Flag" : "RF no"} ${chips(s.flags)}</td>
    </tr>`
    )
    .join("")}</tbody></table>`;
  $("#board").querySelectorAll("tr[data-card]").forEach((tr) => {
    tr.addEventListener("click", () => {
      if (tr.dataset.card) location.hash = `#/card/${tr.dataset.card}`;
      else location.hash = "#/ask";
    });
  });
  app.querySelectorAll("h2 + table tr[data-card], table tr[data-card]").forEach((tr) => {
    if (tr.closest("#board")) return;
    tr.addEventListener("click", () => {
      if (tr.dataset.card) location.hash = `#/card/${tr.dataset.card}`;
    });
  });
}

function p72Cell(s) {
  const p = s.p_burn_by_T || {};
  const v = p["72"] ?? p.t72;
  if (v === null || v === undefined) return `<span class="na">n/a</span>`;
  const n = Number(v);
  const note = n === 0 ? "unreached" : "engine";
  return `<div><strong>${n.toFixed(2)}</strong></div><div class="muted">${note}</div>`;
}

function etaCell(s) {
  if (s.eta_hours == null) {
    if (s.spread_field_version) return `<span class="na">outside field</span>`;
    return `<span class="na">no field</span>`;
  }
  const sig = s.eta_sigma_hours == null ? "" : ` ± ${Number(s.eta_sigma_hours).toFixed(0)} h`;
  return `<div>${Number(s.eta_hours).toFixed(0)} h${sig}</div><div class="muted">engine arrival</div>`;
}

function renderAsk(simulate) {
  const title = simulate ? "ELMFIRE simulator" : "Ask";
  const blurb = simulate
    ? "Name a place or pin a point. Policy picks the bucket. ELMFIRE playback is the case unfolding in time."
    : "Name a town near a forest, or pin lat/lng. NLP geocodes to a community pin. Policy picks the action. The agent tailors the playbook from Mireye and the engine.";
  app.innerHTML = `
    <h1>${title}</h1>
    <p class="muted">${blurb}</p>
    <div class="form">
      <textarea id="q" placeholder="Idyllwild, or a town near the forest">${simulate ? "Simulate ignition in the chaparral west of Idyllwild." : "Is Idyllwild in play if this ignites west of town?"}</textarea>
      <input id="name" class="wide" placeholder="Site name (optional)" value="" />
      <input id="lat" type="number" step="0.0001" placeholder="lat (optional)" />
      <input id="lng" type="number" step="0.0001" placeholder="lng (optional)" />
      <button type="button" class="ghost" id="showcase">Load Idyllwild case</button>
      <button id="go">${simulate ? "Run simulator" : "Ask"}</button>
    </div>
    <p class="muted">Leave lat/lng empty to parse the town. Load Idyllwild sets the community pin and a separate west-chaparral ignition. Honesty line is required in the playbook.</p>
    <div class="grid2">
      <div>
        <div id="map"></div>
        <div id="playback"></div>
        <div id="trace" class="panel trace" style="margin-top:12px">Waiting for tools…</div>
      </div>
      <div id="result"><div class="empty">Submit to run the agent.</div></div>
    </div>`;
  initMap(33.7461, -116.7139);
  $("#go").addEventListener("click", () => startAsk(simulate));
  $("#showcase").addEventListener("click", () => loadShowcase(simulate));
}

const SHOWCASE_FALLBACK = {
  id: "idyllwild_chaparral",
  name: "Idyllwild",
  place: "Idyllwild, California",
  lat: 33.7461,
  lng: -116.7139,
  ignition_lat: 33.744,
  ignition_lng: -116.732,
  simulate: true,
  buffer_km: 8,
  q: "Idyllwild is the community. If the chaparral west of town ignites, is the town in play?",
  honesty:
    "Community pin is Idyllwild (33.7461, -116.7139). Ignition is west in San Jacinto chaparral (33.7440, -116.7320). ELMFIRE playback is the engine case unfolding in time.",
};

let caseIgnition = { lat: SHOWCASE_FALLBACK.ignition_lat, lng: SHOWCASE_FALLBACK.ignition_lng };
let caseBuffer = 8;
let caseLoaded = false;

function readCoord(sel) {
  const raw = ($(sel).value || "").trim();
  if (!raw) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return n;
}

function usablePoint(lat, lng) {
  if (lat == null || lng == null) return false;
  if (Math.abs(lat) < 1e-6 && Math.abs(lng) < 1e-6) return false;
  return lat >= 24 && lat <= 50 && lng >= -126 && lng <= -66;
}

async function loadShowcase(simulate) {
  let s = SHOWCASE_FALLBACK;
  try {
    const r = await fetch("/api/showcase");
    if (r.ok) s = Object.assign({}, SHOWCASE_FALLBACK, await r.json());
  } catch (e) {
    /* keep fallback */
  }
  $("#lat").value = s.lat;
  $("#lng").value = s.lng;
  $("#name").value = s.name || "Idyllwild";
  const q = String(s.q || SHOWCASE_FALLBACK.q).replace(/^"|"$/g, "");
  $("#q").value = simulate
    ? `Simulate ignition in the chaparral west of ${s.place || s.name}. ${q}`
    : q;
  caseIgnition = { lat: Number(s.ignition_lat), lng: Number(s.ignition_lng) };
  caseBuffer = Number(s.buffer_km) || 8;
  caseLoaded = true;
  initMap(s.lat, s.lng);
  $("#result").innerHTML = `<div class="banner">${s.honesty || SHOWCASE_FALLBACK.honesty}</div>
    <div class="muted">Town ${Number(s.lat).toFixed(4)}, ${Number(s.lng).toFixed(4)} · ignition ${caseIgnition.lat.toFixed(4)}, ${caseIgnition.lng.toFixed(4)} · buffer ${caseBuffer} km</div>`;
}

function stopPlayback() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
}

function initMap(lat, lng) {
  stopPlayback();
  playbackOverlay = null;
  if (!usablePoint(lat, lng)) {
    lat = SHOWCASE_FALLBACK.lat;
    lng = SHOWCASE_FALLBACK.lng;
  }
  if (mapRef) {
    mapRef.remove();
    mapRef = null;
  }
  const el = $("#map");
  if (!el) return;
  mapRef = L.map("map").setView([lat, lng], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
  }).addTo(mapRef);
  L.circleMarker([lat, lng], { radius: 7, color: "#111", weight: 2, fillColor: "#f4f0e6", fillOpacity: 1 })
    .bindTooltip("community")
    .addTo(mapRef);
  setTimeout(() => mapRef && mapRef.invalidateSize(), 80);
}

function rasterAtHour(field, hour) {
  const grid = field.arrival_hours;
  if (!grid || !grid.length) return null;
  const h = grid.length;
  const w = grid[0].length;
  const src = document.createElement("canvas");
  src.width = w;
  src.height = h;
  const sctx = src.getContext("2d");
  const img = sctx.createImageData(w, h);
  let n = 0;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = grid[r][c];
      const i = (r * w + c) * 4;
      if (v === null || v === undefined || Number(v) > hour) {
        img.data[i + 3] = 0;
        continue;
      }
      n += 1;
      const age = hour - Number(v);
      const front = age <= 3;
      img.data[i] = front ? 255 : 160;
      img.data[i + 1] = front ? Math.round(220 - age * 20) : 40;
      img.data[i + 2] = front ? 40 : 10;
      img.data[i + 3] = front ? 230 : 200;
    }
  }
  sctx.putImageData(img, 0, 0);
  const scale = 6;
  const dst = document.createElement("canvas");
  dst.width = w * scale;
  dst.height = h * scale;
  const dctx = dst.getContext("2d");
  dctx.imageSmoothingEnabled = false;
  dctx.drawImage(src, 0, 0, dst.width, dst.height);
  dst.dataset.reached = String(n);
  return dst.toDataURL("image/png");
}

function setPlaybackHour(field, hour) {
  if (!mapRef || !field || field.west == null) return;
  const bounds = [
    [field.south, field.west],
    [field.north, field.east],
  ];
  const url = rasterAtHour(field, hour);
  if (!url) return;
  if (playbackOverlay) mapRef.removeLayer(playbackOverlay);
  playbackOverlay = L.imageOverlay(url, bounds, { opacity: 0.85, pane: "overlayPane" }).addTo(mapRef);
  const label = $("#hourLabel");
  if (label) label.textContent = `${Math.round(hour)} h`;
  const slider = $("#hour");
  if (slider && Number(slider.value) !== Math.round(hour)) slider.value = String(Math.round(hour));
  const reached = $("#reached");
  if (reached) {
    const bb = field.burned_bbox;
    reached.textContent = `${field.n_reached || "?"} cells reached · max η ${field.max_eta_hours == null ? "n/a" : Number(field.max_eta_hours).toFixed(1)} h`;
  }
}

function mountPlayback(field, host, { autoplay = false } = {}) {
  stopPlayback();
  const el = host || $("#playback");
  if (!el || !field || !field.arrival_hours) {
    if (el) el.innerHTML = "";
    return;
  }
  const maxH = Math.max(1, Math.ceil(Number(field.max_eta_hours) || Number(field.horizon_hours) || 72));
  playbackHour = 0;
  el.innerHTML = `
    <div class="playback">
      <div class="muted">Engine case unfolding in time.</div>
      <div class="play-row">
        <button type="button" id="play">Play</button>
        <input type="range" id="hour" min="0" max="${maxH}" step="1" value="0" />
        <span id="hourLabel">0 h</span>
        <span class="muted">max ${maxH} h</span>
      </div>
      <div id="reached" class="muted"></div>
    </div>`;
  const playBtn = $("#play", el);
  const apply = (h) => {
    playbackHour = h;
    setPlaybackHour(field, h);
  };
  const stopAtEnd = () => {
    stopPlayback();
    const btn = $("#play", el);
    if (btn) btn.textContent = "Play";
    else if (playBtn) playBtn.textContent = "Play";
  };
  const tick = () => {
    const step = Math.max(1, Math.round(maxH / 36));
    const next = playbackHour + step;
    if (next >= maxH) {
      apply(maxH);
      stopAtEnd();
      return;
    }
    apply(next);
  };
  $("#hour", el).addEventListener("input", (e) => apply(Number(e.target.value)));
  playBtn.addEventListener("click", () => {
    if (playbackTimer) {
      stopAtEnd();
      return;
    }
    if (playbackHour >= maxH) apply(0);
    playBtn.textContent = "Pause";
    playbackTimer = setInterval(tick, 350);
  });
  apply(0);
  const fit = field.burned_bbox
    ? [
        [field.burned_bbox.south, field.burned_bbox.west],
        [field.burned_bbox.north, field.burned_bbox.east],
      ]
    : [
        [field.south, field.west],
        [field.north, field.east],
      ];
  mapRef.fitBounds(fit, { padding: [28, 28], maxZoom: 13 });
  if (autoplay) {
    playBtn.textContent = "Pause";
    playbackTimer = setInterval(tick, 350);
  }
}

function drawReportOnMap(report, opts = {}) {
  const site = report.site || {};
  const lat = site.lat;
  const lng = site.lng;
  if (!mapRef) initMap(usablePoint(lat, lng) ? lat : SHOWCASE_FALLBACK.lat, usablePoint(lat, lng) ? lng : SHOWCASE_FALLBACK.lng);
  if (usablePoint(lat, lng)) {
    mapRef.setView([lat, lng], 12);
    L.circleMarker([lat, lng], { radius: 7, color: "#111", weight: 2, fillColor: "#f4f0e6", fillOpacity: 1 })
      .bindTooltip(site.name || "community")
      .addTo(mapRef);
  }
  (report.map?.hotspots || []).forEach((hs) => {
    L.circleMarker([hs.lat, hs.lng], { radius: 3, color: "#c0392b", fillOpacity: 0.8 }).addTo(mapRef);
  });
  (report.map?.perimeters || []).forEach((p) => {
    (p.rings || []).forEach((ring) => {
      const latlngs = ring.map(([lng0, lat0]) => [lat0, lng0]);
      L.polygon(latlngs, { color: "#c2185b", weight: 2, fill: false }).bindTooltip("operational, unofficial").addTo(mapRef);
    });
  });
  if (report.map?.ignition && usablePoint(report.map.ignition.lat, report.map.ignition.lng)) {
    L.circleMarker([report.map.ignition.lat, report.map.ignition.lng], {
      radius: 8,
      color: "#e67e22",
      fillColor: "#e67e22",
      fillOpacity: 0.9,
    })
      .bindTooltip("ignition")
      .addTo(mapRef);
  }
  const field = report.engine?.field;
  if (field && field.arrival_hours) {
    mountPlayback(field, $("#playback"), { autoplay: Boolean(opts.autoplay) });
  }
  setTimeout(() => mapRef && mapRef.invalidateSize(), 80);
}

async function startAsk(simulate) {
  const lat = readCoord("#lat");
  const lng = readCoord("#lng");
  const name = $("#name").value || "ask-mode-site";
  const q = $("#q").value;
  $("#go").disabled = true;
  $("#trace").innerHTML = "Connecting to live tools…";
  $("#result").innerHTML = `<div class="muted">Live OpenAI + Mireye + ELMFIRE ensemble (policy n=7). simulate_ignition is a real engine run — stay on this page.</div>`;
  const payload = { name, q, simulate };
  if (usablePoint(lat, lng)) {
    payload.lat = lat;
    payload.lng = lng;
    initMap(lat, lng);
  }
  const idyllwildCase =
    caseLoaded ||
    /idyllwild/i.test(`${q} ${name}`) ||
    (usablePoint(lat, lng) &&
      Math.abs(lat - SHOWCASE_FALLBACK.lat) < 0.02 &&
      Math.abs(lng - SHOWCASE_FALLBACK.lng) < 0.02);
  if (idyllwildCase && usablePoint(caseIgnition.lat, caseIgnition.lng)) {
    payload.ignition_lat = caseIgnition.lat;
    payload.ignition_lng = caseIgnition.lng;
    payload.buffer_km = caseBuffer;
  }
  const url = simulate ? "/api/simulate" : "/api/ask";
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    $("#result").innerHTML = `<div class="banner warn">HTTP ${resp.status}</div>`;
    $("#go").disabled = false;
    return;
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const traces = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop();
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.event === "tool") {
        traces.push(ev);
        $("#trace").innerHTML = traces
          .map((t) => {
            const cls = t.ok ? "ok" : "err";
            const back = t.backfill ? ' <span class="back">backfill</span>' : "";
            const extra = t.tool === "simulate_ignition" && t.result && t.result.engine ? ` · ${t.result.engine}` : "";
            return `<div><span class="${cls}">${t.ok ? "ok" : "err"}</span> ${t.tool} ${Math.round(t.latency_ms)}ms${back} ${t.error || ""}${extra}</div>`;
          })
          .join("");
        $("#trace").scrollTop = $("#trace").scrollHeight;
      }
      if (ev.event === "spread" && ev.field) {
        $("#result").innerHTML = `<div class="banner">ELMFIRE field ${ev.field.spread_field_version || ""} · ${ev.field.n_reached || "?"} reached cells. Playing arrival…</div>`;
        if (!mapRef) initMap((ev.site && ev.site.lat) || SHOWCASE_FALLBACK.lat, (ev.site && ev.site.lng) || SHOWCASE_FALLBACK.lng);
        if (ev.ignition && usablePoint(ev.ignition.lat, ev.ignition.lng)) {
          L.circleMarker([ev.ignition.lat, ev.ignition.lng], { radius: 8, color: "#e67e22", fillColor: "#e67e22", fillOpacity: 0.9 })
            .bindTooltip("ignition")
            .addTo(mapRef);
        }
        mountPlayback(ev.field, $("#playback"), { autoplay: true });
      }
      if (ev.event === "status") {
        $("#trace").innerHTML += `<div class="muted">${ev.message}</div>`;
      }
      if (ev.event === "complete") {
        lastReport = ev.report;
        paintLivePanel(ev.report);
      }
      if (ev.event === "error") {
        $("#result").innerHTML = `<div class="banner warn">${ev.message}</div>`;
      }
    }
  }
  $("#go").disabled = false;
}

function paintLivePanel(report) {
  lastReport = report;
  const card = report.action_card;
  if (!card) {
    $("#result").innerHTML = `<div class="banner warn">No ActionCard. Check the trace.</div>`;
    return;
  }
  window.__CARD_CACHE = window.__CARD_CACHE || {};
  window.__CARD_CACHE[card.card_id] = report;
  const honesty = (report.parse && report.parse.honesty) || "";
  const etaNull = card.eta_hours == null;
  $("#result").innerHTML = `
    ${honesty ? `<div class="banner">${honesty}</div>` : ""}
    ${report.simulate ? `<div class="banner warn">Simulated ignition west of town in chaparral.</div>` : ""}
    ${pill(card.action)}
    <div class="action-hero" style="color: var(--${card.action})">${(card.action || "").replaceAll("_", " ").toUpperCase()}</div>
    <div>${card.site.name} · ${Number(card.site.lat).toFixed(4)}, ${Number(card.site.lng).toFixed(4)}</div>
    <div class="muted">policy ${card.policy_version} · ${na(card.spread_field_version)}</div>
    <h2>Engine clock at the community</h2>
    ${
      etaNull
        ? `<p>Field ran. Arrival at the community pin is still outside the reached cells.</p>`
        : `<div class="clock">${Number(card.eta_hours).toFixed(0)} h <span class="muted">± ${card.eta_sigma_hours == null ? "n/a" : Number(card.eta_sigma_hours).toFixed(0)} h</span></div>`
    }
    ${pburnPanel(card, report)}
    ${playbookHtml(report)}
    <h2>Why</h2>
    <ul class="list">${(card.reasons || []).map((a) => `<li>${a}</li>`).join("")}</ul>
    <p><a href="#/card/${card.card_id}">Open full card</a></p>`;
  drawReportOnMap(report, { autoplay: true });
}

function paintResult(report) {
  paintLivePanel(report);
}

function pAt(p, h) {
  if (!p) return null;
  const v = p[h] ?? p[`t${h}`] ?? p[`T${h}`];
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtP(v) {
  if (v === null || v === undefined) return "n/a";
  return Number(v).toFixed(2);
}

function nearestFrontFromField(field, lat, lng) {
  const grid = field && field.arrival_hours;
  if (!grid || !grid.length || field.west == null || !usablePoint(lat, lng)) return null;
  const h = grid.length;
  const w = grid[0].length;
  let best = null;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const eta = grid[r][c];
      if (eta === null || eta === undefined || !Number.isFinite(Number(eta))) continue;
      const clat = field.north - ((r + 0.5) / h) * (field.north - field.south);
      const clng = field.west + ((c + 0.5) / w) * (field.east - field.west);
      const dlat = (clat - lat) * 111320;
      const dlng = (clng - lng) * 111320 * Math.cos((lat * Math.PI) / 180);
      const dist = Math.hypot(dlat, dlng);
      if (best === null || dist < best.dist_m) best = { dist_m: dist, eta_hours: Number(eta), lat: clat, lng: clng };
    }
  }
  return best;
}

function pburnPanel(card, report) {
  const p = card.p_burn_by_T;
  const rows = [
    ["24", pAt(p, "24")],
    ["48", pAt(p, "48")],
    ["72", pAt(p, "72")],
  ].filter(([, v]) => v !== null);
  if (!rows.length) return `<div class="pburn"><h2>P(burn) at the community pin</h2><p class="na">Engine did not return P(burn) for this pin.</p></div>`;
  const eng = report.engine || {};
  const nMembers = eng.n_members || eng.field?.n_members;
  const fieldMax = eng.p_burn_field_max || eng.field?.p_burn_field_max || {};
  const front =
    eng.front && eng.front.dist_m != null
      ? eng.front
      : nearestFrontFromField(eng.field, card.site?.lat, card.site?.lng);
  const nReached = eng.field?.n_reached;
  const memberNote =
    nMembers && Number(nMembers) > 1
      ? `Ensemble n=${nMembers}: fraction of members whose arrival at this pixel is ≤ T.`
      : `n=1 run: 1.00 if this pixel arrives by T, else 0.00.`;
  const bars = rows
    .map(([h, v]) => {
      const pct = Math.round(Number(v) * 100);
      const zero = Number(v) === 0;
      return `<div class="pburn-row">
        <div class="muted">by ${h} h</div>
        <div class="pburn-val">${Number(v).toFixed(2)}</div>
        <div class="bar"><span style="width:${pct}%"></span></div>
        <div class="muted">${zero ? "unreached (engine 0)" : `${pct} % of members`}</div>
      </div>`;
    })
    .join("");
  const max72 = fieldMax["72"] ?? fieldMax.t72;
  const frontLine =
    front && front.dist_m != null
      ? `<p>Nearest reached cell: <strong>${fmtKm(front.dist_m)}</strong> · arrival ${
          front.eta_hours == null ? "n/a" : `${Number(front.eta_hours).toFixed(1)} h`
        }${front.p_burn_72 == null ? "" : ` · P72 there ${fmtP(front.p_burn_72)}`}</p>`
      : "";
  return `<div class="pburn">
    <h2>P(burn) at the community pin</h2>
    <p class="muted">${memberNote} 0.00 is engine output, not a missing value.</p>
    ${bars}
    <p class="muted">Field max P(burn by 72 h) ${fmtP(max72)}${
      nReached == null ? "" : ` · ${nReached} cells reached`
    }. Sampled at ${Number(card.site.lat).toFixed(4)}, ${Number(card.site.lng).toFixed(4)} — not the ignition cell.</p>
    ${frontLine}
  </div>`;
}

function aspectsHtml(aspects, report) {
  if (!aspects || !Object.keys(aspects).length) {
    const mireye = (report && report.trace ? report.trace : []).find((t) => t.tool === "mireye_fetch");
    const err = mireye && mireye.error ? ` Live call: ${mireye.error}.` : "";
    return `<div class="na">Live Mireye returned no aspect fields.${err}</div>`;
  }
  return Object.entries(aspects)
    .map(([role, block]) => {
      const kvs = Object.entries(block.fields || {})
        .filter(([, meta]) => meta && meta.value !== null && meta.value !== undefined)
        .map(
          ([n, meta]) =>
            `<span>${n}</span><span>${meta.value}${meta.confidence ? ` <span class="muted">${meta.confidence}</span>` : ""}</span>`
        )
        .join("");
      return `<div class="aspect"><h3>${role} · ${block.name}</h3><div class="kv">${kvs}</div></div>`;
    })
    .join("");
}

function playbookHtml(report) {
  const steps = report.playbook || report.action_card?.recommended_actions || [];
  if (!steps.length) return "";
  return `<h2>Playbook (${report.action_card?.action || ""})</h2>
    <ul class="list">${steps.map((s) => `<li>${s}</li>`).join("")}</ul>`;
}

function responseHtml(rc) {
  if (!rc) return "";
  const water = (rc.water_sources || [])
    .map((w) => `<li>${w.type} ${w.name || ""} · availability ${w.availability || "unknown"} · ${w.distance_m == null ? "distance unknown" : fmtKm(w.distance_m)}</li>`)
    .join("");
  const routes = (rc.access_routes || [])
    .map((r) => `<li>${r.road_class || "road"} ${r.road_name || ""} · ${fmtKm(r.distance_m)} · ${r.usability}</li>`)
    .join("");
  const haz = (rc.hazmat_sites || [])
    .map((h) => `<li>${h.priority} ${h.type} ${h.name || ""} · ${fmtKm(h.distance_m)}</li>`)
    .join("");
  const fs = rc.fire_station || {};
  const src = fs.eta_source === "osm_network" ? "OSM road network" : fs.eta_source === "distance_proxy" ? "Distance / speed estimate (V1 fallback)" : "";
  return `<h2>Response dossier</h2>
    <div class="panel">
      <p>Agency: <strong>${na(rc.responsible_agency)}</strong></p>
      <p>Fire station: ${na(fs.name)} · ${fmtKm(fs.distance_m)} · ETA ${fs.eta_minutes_estimate == null ? '<span class="na">n/a</span>' : fs.eta_minutes_estimate + " min"} ${src ? `<span class="chip">${src}</span>` : ""}</p>
      <h3>Water</h3><ul class="list">${water || "<li>none within radius</li>"}</ul>
      <h3>Access</h3><ul class="list">${routes || "<li>none</li>"}</ul>
      <h3>Hazmat (≤ 5 km)</h3><ul class="list">${haz || "<li>none within 5 km</li>"}</ul>
    </div>`;
}

async function renderCard(cardId) {
  let report = (window.__CARD_CACHE && window.__CARD_CACHE[cardId]) || lastReport;
  if (!report || report.action_card?.card_id !== cardId) {
    try {
      report = await (await fetch(`/api/cards/${cardId}`)).json();
    } catch (e) {
      app.innerHTML = `<div class="banner warn">Card not found. Run Ask first.</div>`;
      return;
    }
  }
  lastReport = report;
  const card = report.action_card;
  const eng = report.engine || {};
  const etaNull = card.eta_hours == null;
  const hasField = Boolean(card.spread_field_version);
  const honesty = (report.parse && report.parse.honesty) || "";
  const notify = report.notify || {};
  app.innerHTML = `
    <div class="grid2">
      <div>
        <div class="panel">
          ${pill(card.action)}
          <div class="action-hero" style="color: var(--${card.action})">${(card.action || "").replaceAll("_", " ").toUpperCase()}</div>
          <div>${card.site.name} · ${card.site.lat.toFixed(4)}, ${card.site.lng.toFixed(4)}</div>
          <div class="muted">${card.generated_at} · policy ${card.policy_version}</div>
          ${honesty ? `<div class="banner">${honesty}</div>` : ""}
          ${card.flags?.includes("no_ros_high_sigma") ? `<div class="banner warn">Evacuate was suppressed because uncertainty is too high. Action is protect_asset.</div>` : ""}
          ${report.simulate ? `<div class="banner warn">Simulated ignition west of town in chaparral.</div>` : ""}
          ${notify.log_only ? `<div class="banner">notify_ops logged only (no Slack token). Channel ${na(notify.channel)}.</div>` : ""}
          ${playbookHtml(report)}
          <h2>Why this action</h2>
          <ul class="list">${(card.reasons || []).map((a) => `<li>${a}</li>`).join("")}</ul>
          <div>${chips(card.flags)}</div>
        </div>
        <div class="panel" style="margin-top:12px">
          <h2>Engine clock</h2>
          ${
            etaNull
              ? hasField
                ? `<p>Spread field <code>${card.spread_field_version}</code> ran. Arrival at this site is still outside the reached cells.</p>`
                : `<p class="na">no field</p>`
              : `<div class="clock">${Number(card.eta_hours).toFixed(0)} h <span class="muted">± ${card.eta_sigma_hours == null ? "n/a" : Number(card.eta_sigma_hours).toFixed(0)} h</span></div>
                 <div class="muted">arrival estimate with sigma</div>`
          }
          ${pburnPanel(card, report)}
          <div class="kv">
            <span>spread_field_version</span><span>${na(card.spread_field_version)}</span>
            <span>engine</span><span>${na(eng.engine)}</span>
            <span>policy</span><span>${card.policy_version}</span>
            <span>clock</span><span>engine ETA + distance</span>
          </div>
        </div>
        <h2>Playbook brief</h2>
        ${report.brief_replaced ? `<div class="banner">Brief failed validation and was replaced with the fallback.</div>` : ""}
        <div class="panel">${(report.brief || "").replaceAll("\n", "<br/>")}</div>
        ${responseHtml(report.response_card)}
      </div>
      <div>
        <div id="map"></div>
        <div id="playback"></div>
        <h2>Situation</h2>
        <div class="panel">
          <div class="kv">
            <span>Incident</span><span>${na(card.incident.incident_name)} ${card.incident.irwin_id ? `<a href="#/incidents/${encodeURIComponent(card.incident.irwin_id)}">${card.incident.irwin_id}</a>` : ""}</span>
            <span>Acres</span><span>${card.incident.acres == null ? '<span class="na">n/a</span>' : card.incident.acres}</span>
            <span>Containment</span><span>${card.incident.containment_pct == null ? '<span class="na">n/a</span>' : card.incident.containment_pct}</span>
            <span>Distance to perimeter</span><span>${fmtKm(card.incident.dist_perimeter_m)}</span>
            <span>Red Flag</span><span>${card.weather.red_flag ? "yes" : "no"}</span>
            <span>Wind</span><span>${card.weather.wind_speed_ms == null ? '<span class="na">n/a</span>' : `${card.weather.wind_speed_ms} m/s ${card.weather.wind_dir_cardinal || ""}`}</span>
            <span>RH</span><span>${card.weather.rh_pct == null ? '<span class="na">n/a</span>' : card.weather.rh_pct + " %"}</span>
          </div>
        </div>
        <h2>Mireye aspects (live API)</h2>
        <div class="panel">${aspectsHtml(report.aspects, report)}</div>
        <h2>Tool trace</h2>
        <div class="panel trace">${(report.trace || [])
          .map((t) => `<div><span class="${t.ok ? "ok" : "err"}">${t.ok ? "ok" : "err"}</span> ${t.tool} ${Math.round(t.latency_ms)}ms ${t.backfill ? '<span class="back">backfill</span>' : ""} ${t.error || ""}</div>`)
          .join("")}</div>
        <div class="footer">card ${card.card_id} · policy ${card.policy_version} · ${card.spread_field_version || "no spread"}</div>
      </div>
    </div>`;
  initMap(card.site.lat, card.site.lng);
  drawReportOnMap(report);
}

async function renderIncident(irwin) {
  try {
    const data = await (await fetch(`/api/incidents/${encodeURIComponent(irwin)}`)).json();
    app.innerHTML = `<h1>${na(data.incident_name)} <span class="muted">${irwin}</span></h1>
      <p class="muted">Hour playback of the engine field.</p>
      <div id="map"></div>
      <div id="playback"></div>
      <pre class="panel trace">${JSON.stringify(data.engine?.sample || {}, null, 2)}</pre>`;
    initMap(34, -118);
    drawReportOnMap({ site: { lat: 34, lng: -118, name: irwin }, engine: data.engine, map: data.map });
  } catch (e) {
    app.innerHTML = `<div class="banner warn">Incident not in this session.</div>`;
  }
}

async function renderAudit() {
  app.innerHTML = `<h1>Audit</h1><p class="muted">Today’s JSONL tool log. Reconstruct numbers from here.</p><div id="log">Loading…</div>`;
  const data = await (await fetch("/api/logs")).json();
  $("#log").innerHTML = `<div class="panel trace">${(data.lines || [])
    .slice(-80)
    .map((l) => {
      let o = {};
      try {
        o = JSON.parse(l);
      } catch (e) {
        return `<div>${l}</div>`;
      }
      return `<div>${o.ts || ""} ${o.kind || ""} ${o.tool || ""} ${o.site_id || ""} ${o.error || ""}</div>`;
    })
    .join("")}</div>`;
}

window.addEventListener("hashchange", route);
loadHealth();
route();
