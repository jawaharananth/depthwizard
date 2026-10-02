// Tool dock: every analysis tool one click (or one key) away, bottom centre.
//
// The tools themselves are the viewer's existing ones -- this dock only gives
// them a place on screen. Each item clicks the original control, so the
// behaviour, guards and readouts are exactly the tested ones. The readout
// (#measure-panel), the cross-section profile (#xsec) and the flood and sun
// sliders are moved into a card that rises above the dock while a tool is in
// use. Keys 1-9 and 0 follow the order on screen.

const I = {   // 24px stroke icons
  photo: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
  conf: '<path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="8"/>',
  slope: '<path d="M3 20h18L14 7l-4 6-3-3z"/>',
  measured: '<path d="M4 20 20 4"/><path d="m6 14 2 2M10 10l2 2M14 6l2 2"/>',
  height: '<path d="M12 3v18M8 7l4-4 4 4M8 17l4 4 4-4"/>',
  section: '<path d="M3 17c3-6 5-9 9-9s6 4 9 1"/><path d="M3 21h18"/>',
  sight: '<circle cx="5" cy="17" r="2"/><path d="M7 16 19 8"/><circle cx="19" cy="7" r="2"/>',
  area: '<path d="M4 6l7-3 9 5-3 11-11-2z"/>',
  flood: '<path d="M3 15c2 0 2-2 4.5-2S10 15 12 15s2-2 4.5-2 2.5 2 4.5 2"/><path d="M3 19c2 0 2-2 4.5-2S10 19 12 19s2-2 4.5-2 2.5 2 4.5 2"/><path d="M12 3v7"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  dsm: '<path d="M12 3v12M7 10l5 5 5-5M4 19h16"/>',
  report: '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h5"/>',
};
const svg = (k) => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${I[k]}</svg>`;

export function installToolbar() {
  const $ = (id) => document.getElementById(id);
  const css = document.createElement("style");
  css.textContent = `
  #dwt { position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%); z-index: 32;
         display: flex; gap: 4px; padding: 6px; border-radius: 16px; align-items: stretch;
         background: rgba(17,24,31,.86); border: 1px solid rgba(255,255,255,.09);
         box-shadow: 0 18px 40px -18px rgba(0,0,0,.6); backdrop-filter: blur(10px);
         font: 500 12px "Segoe UI", system-ui, sans-serif; color: #dbe4ea; max-width: calc(100vw - 260px); overflow-x: auto; }
  #dwt .grp { display: flex; gap: 2px; padding: 0 4px; }
  #dwt .grp + .grp { border-left: 1px solid rgba(255,255,255,.09); }
  #dwt button, #dwt a { appearance: none; border: 0; background: transparent; color: inherit; text-decoration: none;
         display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 58px; padding: 7px 8px 6px;
         border-radius: 10px; cursor: pointer; font: inherit; position: relative; }
  #dwt button:hover, #dwt a:hover { background: rgba(255,255,255,.07); }
  #dwt button.on { background: rgba(242,181,68,.16); color: #f2b544; }
  #dwt button:disabled { opacity: .35; cursor: default; }
  #dwt button:focus-visible, #dwt a:focus-visible { outline: 2px solid #5ec8b8; outline-offset: 1px; }
  #dwt kbd { position: absolute; top: 3px; right: 5px; font: 600 9px "Cascadia Mono", ui-monospace, monospace; opacity: .45; }
  #dwt-card { position: fixed; left: 50%; bottom: 92px; transform: translateX(-50%); z-index: 32; display: none;
         width: min(560px, calc(100vw - 32px)); padding: 12px 14px; border-radius: 14px;
         background: rgba(17,24,31,.9); border: 1px solid rgba(255,255,255,.09); color: #e8eef2;
         font: 13px/1.5 "Segoe UI", system-ui, sans-serif; backdrop-filter: blur(10px); }
  #dwt-card .row { display: flex; align-items: center; gap: 10px; margin: 6px 0 2px; }
  #dwt-card .row label { width: 92px; color: #9fb0bf; font-size: 12px; }
  #dwt-card input[type=range] { flex: 1; accent-color: #f2b544; }
  #dwt-card .readout { color: #c9d5de; font: 12px "Cascadia Mono", ui-monospace, monospace; white-space: pre-line; }
  #dwt-card #measure-panel { white-space: pre-line; }
  #dwt-card canvas#xsec { width: 100% !important; max-width: none !important; height: 120px; border-radius: 8px; margin-top: 8px; }
  #dwt-card .x { position: absolute; top: 6px; right: 8px; background: none; border: 0; color: #9fb0bf; font-size: 16px; cursor: pointer; }
  #dwt-card .btnrow { display: flex; gap: 8px; margin-top: 8px; }
  #dwt-card .btnrow button { border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); color: #e8eef2;
         border-radius: 8px; padding: 6px 10px; cursor: pointer; font: 600 12px "Segoe UI", system-ui; }
  #dwt-card .btnrow button.active { border-color: #f2b544; color: #f2b544; }
  #dw-demo-caption { bottom: 104px !important; }
  #dwx-rgb, #dwx-conf, #dwx-measured { display: none !important; }
  #dw-demo-btn { bottom: 20px !important; }
  @media (max-width: 900px) { #dwt { max-width: calc(100vw - 16px); } #dwt button, #dwt a { min-width: 44px; } #dwt span.lb { display: none; } }`;
  document.head.appendChild(css);

  // --- items: [key, label, icon, how to activate, how to read "on"] ---------
  const legacy = (id) => () => { const b = $(id); if (b && !b.disabled) b.click(); };
  const isOn = (id, cls = "active") => () => { const b = $(id); return !!(b && b.classList.contains(cls)); };
  const groups = [
    [["photo", "Photo", "photo", legacy("dwx-rgb"), isOn("dwx-rgb", "on")],
     ["conf", "± map", "conf", legacy("dwx-conf"), isOn("dwx-conf", "on")],
     ["slope", "Slope", "slope", legacy("btn-slope"), isOn("btn-slope")],
     ["measured", "Measured", "measured", legacy("dwx-measured"), isOn("dwx-measured", "on")]],
    [["height", "Height", "height", legacy("btn-measure"), isOn("btn-measure")],
     ["section", "Section", "section", legacy("btn-xsec"), isOn("btn-xsec")],
     ["sight", "Sight", "sight", legacy("btn-los"), isOn("btn-los")],
     ["area", "Area", "area", legacy("btn-area"), isOn("btn-area")]],
    [["flood", "Flood", "flood", () => openPanel("flood"), () => panel === "flood"],
     ["sun", "Sun", "sun", () => openPanel("sun"), () => panel === "sun"]],
    [["dsm", "DSM", "dsm", null, null],
     ["report", "Report", "report", legacy("btn-pdf"), null]],
  ];
  const bar = document.createElement("div");
  bar.id = "dwt";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "Analysis tools");
  const items = [];
  let n = 0;
  for (const g of groups) {
    const gr = document.createElement("div");
    gr.className = "grp";
    for (const [id, label, icon, act, on] of g) {
      const key = id === "dsm" || id === "report" ? null : String((++n) % 10);
      const el = document.createElement(id === "dsm" ? "a" : "button");
      el.id = "dwt-" + id;
      if (id === "dsm") { el.href = "#"; el.setAttribute("download", ""); }
      else el.type = "button";
      el.innerHTML = `${svg(icon)}<span class="lb">${label}</span>${key ? `<kbd>${key}</kbd>` : ""}`;
      el.title = label + (key ? `  (key ${key})` : "");
      if (act) el.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); act(); setTimeout(sync, 60); });
      gr.appendChild(el);
      items.push({ id, el, key, act, on });
    }
    bar.appendChild(gr);
  }
  document.body.appendChild(bar);

  // --- the card above the dock ----------------------------------------------
  const card = document.createElement("div");
  card.id = "dwt-card";
  card.setAttribute("role", "region");
  card.setAttribute("aria-label", "Tool readout");
  const close = document.createElement("button");
  close.className = "x"; close.type = "button"; close.textContent = "×"; close.title = "Close";
  card.appendChild(close);
  const body = document.createElement("div");
  card.appendChild(body);
  document.body.appendChild(card);
  let panel = null;    // "flood" | "sun" | null (tool readouts show without a panel)

  const mp = $("measure-panel"), xs = $("xsec");
  const floodRow = $("flood") && $("flood").closest(".slider-row");
  const sunRow = $("sunhr") && $("sunhr").closest(".slider-row");
  const vexRow = $("vexag") && $("vexag").closest(".slider-row");
  const mk = (row, readoutId, label) => {
    const wrap = document.createElement("div");
    if (row) {
      const r = document.createElement("div");
      r.className = "row";
      const l = document.createElement("label");
      l.textContent = label;
      const inp = row.querySelector("input");
      r.appendChild(l); r.appendChild(inp);
      wrap.appendChild(r);
    }
    const ro = $(readoutId);
    if (ro) { ro.classList.add("readout"); wrap.appendChild(ro); }
    return wrap;
  };
  const floodBox = document.createElement("div");
  floodBox.appendChild(mk(floodRow, "flood-readout", "Water level"));
  const fb = document.createElement("div");
  fb.className = "btnrow";
  const srcBtn = document.createElement("button");
  srcBtn.type = "button"; srcBtn.textContent = "Place water source";
  srcBtn.addEventListener("click", () => { legacy("btn-flood")(); setTimeout(sync, 60); });
  fb.appendChild(srcBtn);
  floodBox.appendChild(fb);
  const sunBox = document.createElement("div");
  sunBox.appendChild(mk(sunRow, "sun-readout", "Time of day"));
  sunBox.appendChild(mk(vexRow, "vexag-readout", "Height boost"));
  const toolBox = document.createElement("div");
  if (mp) toolBox.appendChild(mp);
  if (xs) toolBox.appendChild(xs);
  // All three stay in the page (the viewer finds the sliders by id at any
  // time); only their visibility changes.
  for (const b of [floodBox, sunBox, toolBox]) body.appendChild(b);
  floodBox.style.display = sunBox.style.display = "none";

  function openPanel(which) {
    panel = panel === which ? null : which;
    floodBox.style.display = panel === "flood" ? "block" : "none";
    sunBox.style.display = panel === "sun" ? "block" : "none";
    show();
  }
  const toolActive = () => ["btn-measure", "btn-xsec", "btn-los", "btn-area", "btn-gcp", "btn-flood"]
    .some((id) => $(id) && $(id).classList.contains("active"));
  const DEFAULT_TXT = /^Click a building to read its height\.?$/;
  function show() {
    const hasText = mp && mp.textContent.trim() && !DEFAULT_TXT.test(mp.textContent.trim());
    const hasX = xs && xs.style.display === "block";
    toolBox.style.display = (hasText || hasX || toolActive()) ? "block" : "none";
    card.style.display = (panel || toolActive() || hasText || hasX) ? "block" : "none";
  }
  close.addEventListener("click", () => {
    panel = null;
    // turn off whichever tool is armed, through its own button
    for (const id of ["btn-measure", "btn-xsec", "btn-los", "btn-area"]) {
      const b = $(id); if (b && b.classList.contains("active")) b.click();
    }
    if (mp) mp.textContent = "";
    if (xs) xs.style.display = "none";
    floodBox.style.display = sunBox.style.display = "none";
    card.style.display = "none"; sync();
  });

  function sync() {
    for (const it of items) {
      if (it.on) it.el.classList.toggle("on", !!it.on());
      if (it.id === "slope") it.el.disabled = !!($("btn-slope") && $("btn-slope").disabled);
      if (it.id === "conf") it.el.disabled = !!($("dwx-conf") && $("dwx-conf").disabled);
    }
    const dsm = $("btn-dsm");
    const d = items.find((x) => x.id === "dsm");
    if (dsm && dsm.href) { d.el.href = dsm.href; d.el.style.display = ""; d.el.title = (dsm.textContent || "Download DSM").trim() + " (GeoTIFF)"; }
    else d.el.style.display = "none";
    srcBtn.classList.toggle("active", !!($("btn-flood") && $("btn-flood").classList.contains("active")));
    show();
  }
  // follow the original controls, whoever changes them
  const mo = new MutationObserver(() => sync());
  for (const id of ["btn-measure", "btn-xsec", "btn-los", "btn-area", "btn-slope", "btn-flood", "btn-gcp", "dwx-rgb", "dwx-conf", "dwx-measured"]) {
    const el = $(id); if (el) mo.observe(el, { attributes: true, attributeFilter: ["class", "disabled"] });
  }
  if (mp) mo.observe(mp, { childList: true, characterData: true, subtree: true });
  if (xs) mo.observe(xs, { attributes: true, attributeFilter: ["style"] });
  // the DSM link appears once the server has answered
  const wait = setInterval(() => { if ($("btn-dsm")) { sync(); clearInterval(wait); } }, 500);
  setTimeout(() => clearInterval(wait), 20000);

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    const it = items.find((x) => x.key === e.key);
    if (it && it.act && !it.el.disabled) { it.act(); setTimeout(sync, 60); e.preventDefault(); }
  });
  sync();
  return { sync };
}
