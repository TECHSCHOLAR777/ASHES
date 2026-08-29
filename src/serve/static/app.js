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
      h.openai ? "OpenAI" : "no OpenAI",
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
  app.innerHTML = `<h1>Watch board</h1><p class="muted">Book of sites. Last card from this UI session (watch loop is separate).</p>${recentHtml}<div id="board">Loading…</div>`;
  const data = await (await fetch("/api/sites")).json();
  const rows = (data.sites || []).sort((a, b) => ACTION_ORDER.indexOf(a.action) - ACTION_ORDER.indexOf(b.action));
  if (!rows.length) {
    $("#board").innerHTML = `<div class="empty">No sites in config/sites.yaml.</div>`;
    return;
  }
  $("#board").innerHTML = `<table class="table"><thead><tr>
    <th>Site</th><th>Action</th><th>ETA (σ)</th><th>Distance</th><th>y_hat</th><th>Weather</th>
  </tr></thead><tbody>${rows
    .map(
      (s) => `<tr data-card="${s.last_card_id || ""}" data-site="${s.site_id}">
      <td><strong>${s.name}</strong><div class="muted">${s.site_id}</div></td>
      <td>${pill(s.action)}</td>
      <td>${etaCell(s)}</td>
      <td>${fmtKm(s.dist_perimeter_m)}<div class="muted">${s.incident_name || "no incident"}</div></td>
      <td>${s.y_hat == null ? '<span class="na">n/a</span>' : `${Number(s.y_hat).toFixed(2)} ± ${Number(s.sigma ?? 0).toFixed(2)}`}</td>
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

function etaCell(s) {
  if (s.eta_hours == null) {
    if (s.spread_field_version) return `<span class="na">outside field</span>`;
    return `<span class="na">no field</span>`;
  }
  const sig = s.eta_sigma_hours == null ? "" : ` ± ${Number(s.eta_sigma_hours).toFixed(0)} h`;
  return `<div>${Number(s.eta_hours).toFixed(0)} h${sig}</div><div class="muted">arrival estimate with sigma</div>`;
}

function renderAsk(simulate) {
  const title = simulate ? "ELMFIRE simulator" : "Ask";
  const blurb = simulate
    ? "Ignite at a point, fetch LANDFIRE, run the out-of-process engine, then Mireye aspects and the policy table."
    : "One lat/lng. The agent calls live tools. Policy picks the action. The model writes the brief from grounded facts.";
  app.innerHTML = `
    <h1>${title}</h1>
    <p class="muted">${blurb}</p>
    <div class="form">
      <input id="lat" type="number" step="0.0001" value="33.9806" />
      <input id="lng" type="number" step="0.0001" value="-117.3755" />
      <input id="name" class="wide" value="${simulate ? "Simulated ignition" : "Riverside Warehouse"}" />
      <textarea id="q">${simulate ? "Simulate ignition at this parcel and report arrival, aspects, and action." : "Is this warehouse at risk this fire week?"}</textarea>
      <button id="go">${simulate ? "Run simulator" : "Ask"}</button>
    </div>
    <div class="grid2">
      <div>
        <div id="map"></div>
        <div id="trace" class="panel trace" style="margin-top:12px">Waiting for tools…</div>
      </div>
      <div id="result"><div class="empty">Submit to run the agent.</div></div>
    </div>`;
  initMap(33.9806, -117.3755);
  $("#go").addEventListener("click", () => startAsk(simulate));
}

function initMap(lat, lng) {
  if (mapRef) {
    mapRef.remove();
    mapRef = null;
  }
  mapRef = L.map("map").setView([lat, lng], 10);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
  }).addTo(mapRef);
  L.circleMarker([lat, lng], { radius: 6, color: "#111", fillColor: "#111", fillOpacity: 1 }).addTo(mapRef);
}

function rasterCanvas(field, mode) {
  const grid = mode === "p72" ? field.p_burn_72 : field.arrival_hours;
  if (!grid || !grid.length) return null;
  const h = grid.length;
  const w = grid[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = grid[r][c];
      const i = (r * w + c) * 4;
      if (v === null || v === undefined) {
        img.data[i + 3] = 0;
        continue;
      }
      let t;
      if (mode === "p72") t = Math.max(0, Math.min(1, Number(v)));
      else t = 1 - Math.max(0, Math.min(1, Number(v) / 72));
      img.data[i] = Math.round(180 + t * 60);
      img.data[i + 1] = Math.round(40 + (1 - t) * 140);
      img.data[i + 2] = Math.round(20 + (1 - t) * 30);
      img.data[i + 3] = 170;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function drawReportOnMap(report) {
  const site = report.site || {};
  if (!mapRef) initMap(site.lat, site.lng);
  mapRef.setView([site.lat, site.lng], 11);
  L.circleMarker([site.lat, site.lng], { radius: 7, color: "#111", weight: 2, fillColor: "#f4f0e6", fillOpacity: 1 })
    .bindTooltip(site.name || "site")
    .addTo(mapRef);
  (report.map?.hotspots || []).forEach((hs) => {
    L.circleMarker([hs.lat, hs.lng], { radius: 3, color: "#c0392b", fillOpacity: 0.8 }).addTo(mapRef);
  });
  (report.map?.perimeters || []).forEach((p) => {
    (p.rings || []).forEach((ring) => {
      const latlngs = ring.map(([lng, lat]) => [lat, lng]);
      L.polygon(latlngs, { color: "#c2185b", weight: 2, fill: false }).bindTooltip("operational, unofficial").addTo(mapRef);
    });
  });
  const field = report.engine?.field;
  if (field && field.west != null) {
    const bounds = [
      [field.south, field.west],
      [field.north, field.east],
    ];
    const url = rasterCanvas(field, field.p_burn_72 ? "p72" : "arrival");
    if (url) {
      L.imageOverlay(url, bounds, { opacity: 0.6, pane: "overlayPane" }).addTo(mapRef);
      mapRef.fitBounds(bounds, { padding: [20, 20] });
    }
  }
  if (report.map?.ignition) {
    L.circleMarker([report.map.ignition.lat, report.map.ignition.lng], {
      radius: 8,
      color: "#e67e22",
      fillColor: "#e67e22",
      fillOpacity: 0.9,
    })
      .bindTooltip("ignition")
      .addTo(mapRef);
  }
}

async function startAsk(simulate) {
  const lat = Number($("#lat").value);
  const lng = Number($("#lng").value);
  const name = $("#name").value;
  const q = $("#q").value;
  $("#go").disabled = true;
  $("#trace").innerHTML = "Connecting…";
  $("#result").innerHTML = `<div class="muted">Agent is calling tools…</div>`;
  initMap(lat, lng);
  const url = simulate ? "/api/simulate" : "/api/ask";
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lng, name, q, simulate }),
  });
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
            return `<div><span class="${cls}">${t.ok ? "ok" : "err"}</span> ${t.tool} ${Math.round(t.latency_ms)}ms${back} ${t.error || ""}</div>`;
          })
          .join("");
      }
      if (ev.event === "status") {
        $("#trace").innerHTML += `<div class="muted">${ev.message}</div>`;
      }
      if (ev.event === "complete") {
        lastReport = ev.report;
        paintResult(ev.report);
        drawReportOnMap(ev.report);
      }
      if (ev.event === "error") {
        $("#result").innerHTML = `<div class="banner warn">${ev.message}</div>`;
      }
    }
  }
  $("#go").disabled = false;
}

function paintResult(report) {
  lastReport = report;
  const card = report.action_card;
  if (!card) {
    $("#result").innerHTML = `<div class="banner warn">No ActionCard. Check the trace.</div>`;
    return;
  }
  window.__CARD_CACHE = window.__CARD_CACHE || {};
  window.__CARD_CACHE[card.card_id] = report;
  location.hash = `#/card/${card.card_id}`;
}

function pburnBars(p) {
  if (!p) return "";
  const rows = [
    ["24", p["24"] ?? p.t24],
    ["48", p["48"] ?? p.t48],
    ["72", p["72"] ?? p.t72],
  ].filter(([, v]) => v !== null && v !== undefined);
  if (!rows.length) return "";
  return rows
    .map(([h, v]) => {
      const pct = Math.round(Number(v) * 100);
      return `<div>P(burn by ${h} h) ${Number(v).toFixed(2)}<div class="bar"><span style="width:${pct}%"></span></div></div>`;
    })
    .join("");
}

function aspectsHtml(aspects) {
  if (!aspects || !Object.keys(aspects).length) return `<div class="na">No Mireye aspects fetched.</div>`;
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
  app.innerHTML = `
    <div class="grid2">
      <div>
        <div class="panel">
          ${pill(card.action)}
          <div class="action-hero" style="color: var(--${card.action})">${(card.action || "").replaceAll("_", " ").toUpperCase()}</div>
          <div>${card.site.name} · ${card.site.lat.toFixed(4)}, ${card.site.lng.toFixed(4)}</div>
          <div class="muted">${card.generated_at}</div>
          ${card.flags?.includes("no_ros_high_sigma") ? `<div class="banner warn">Evacuate was suppressed because uncertainty is too high. Action is protect_asset.</div>` : ""}
          ${report.simulate ? `<div class="banner warn">Simulated ignition. Engine raster is not an official perimeter.</div>` : ""}
          ${card.model_version && card.model_version.includes("untrained") ? `<div class="banner">Model is untrained (${card.model_version}). Engine ETA / P(burn) still shown when a field exists.</div>` : ""}
          <h2>Recommended actions</h2>
          <ul class="list">${(card.recommended_actions || []).map((a) => `<li>${a}</li>`).join("")}</ul>
          <h2>Why this action</h2>
          <ul class="list">${(card.reasons || []).map((a) => `<li>${a}</li>`).join("")}</ul>
          <div>${chips(card.flags)}</div>
        </div>
        <div class="panel" style="margin-top:12px">
          <h2>Arrival estimate with sigma</h2>
          ${
            etaNull
              ? hasField
                ? `<p>Spread field <code>${card.spread_field_version}</code> ran. This site was not inside a reached cell, so no arrival estimate is shown.</p>`
                : `<p class="na">no field</p>`
              : `<div class="clock">${Number(card.eta_hours).toFixed(0)} h <span class="muted">± ${card.eta_sigma_hours == null ? "n/a" : Number(card.eta_sigma_hours).toFixed(0)} h</span></div>
                 <div class="muted">arrival estimate with sigma</div>`
          }
          ${pburnBars(card.p_burn_by_T)}
          <div class="kv">
            <span>Raw spread field / baseline</span><span>${card.baseline_y == null ? '<span class="na">n/a</span>' : Number(card.baseline_y).toFixed(2)}</span>
            <span>Calibrated y_hat</span><span>${card.y_hat == null ? '<span class="na">n/a</span>' : `${Number(card.y_hat).toFixed(2)} ± ${Number(card.sigma).toFixed(2)}`}</span>
            <span>spread_field_version</span><span>${na(card.spread_field_version)}</span>
            <span>engine</span><span>${na(eng.engine)}</span>
            <span>policy</span><span>${card.policy_version}</span>
            <span>model</span><span>${card.model_version}</span>
          </div>
        </div>
        <h2>Brief</h2>
        ${report.brief_replaced ? `<div class="banner">Brief failed validation and was replaced with the fallback.</div>` : ""}
        <div class="panel">${(report.brief || "").replaceAll("\n", "<br/>")}</div>
        ${responseHtml(report.response_card)}
      </div>
      <div>
        <div id="map"></div>
        <p class="muted">Arrival / P(burn 72) overlay is the engine raster, not an official perimeter.</p>
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
        <h2>Mireye aspects</h2>
        <div class="panel">${aspectsHtml(report.aspects)}</div>
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
      <p class="muted">One field per incident; nearby sites reuse it. Engine raster is not an official perimeter.</p>
      <div id="map"></div>
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
