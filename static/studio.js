/* ============================================================================
   Invitation Promo Studio · client flow (4 steps)
   1) Choose Template  → browse by category / search / preview the sample video
   2) Upload Photos    → upload product shots, apply the template's camera work
   3) Choose Format    → the client picks the format (Vertical 9:16 / Square 1:1).
                         The template's camera work is kept; the framing is
                         re-composed for the chosen format (not cropped after).
   4) Generate Video   → create / preview / download
   Rendering uses the card-pan pipeline (detect pieces → plan moves → render).
   All user-facing copy is English: this studio is shown to EU/US customers.
   ========================================================================== */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const SESSION_KEY = "ips.project";

const ST = {
  project: null,
  photos: [],        // [{path, file}]
  labels: [],        // Collection Showcase: design name per photo (optional)
  shots: [],         // Collection Showcase: per-photo camera plan ("detail" | "full" | "rotate" | "")
  focuses: [],       // Collection Showcase: per-photo detail focus bbox [x0,y0,x1,y1] or null
  skus: [],          // TOP 5 Countdown: style SKU per photo (optional, drawn beside the rank number)
  nums: [],          // TOP 5 Countdown: rank number per photo (optional, blank = auto countdown n..01)
  _recogDone: {},     // paths already sent for auto-label recognition (avoid duplicate calls)
  _recogPend: {},     // paths currently being recognized (show "Recognizing…" placeholder)
  template: null,    // selected template object
  ratio: "9:16",     // output format
  busy: false,
  rendered: false,
};

/* The multi-photo template ("Collection Showcase"): 2-8 photos, one per design. */
function isMultiTpl() {
  return !!(ST.template && ST.template.pipeline === "multi_suite");
}

/* The scrolling-gallery template: 2-8 photos, no names, no text overlay. */
function isScrollTpl() {
  return !!(ST.template && ST.template.pipeline === "scroll_gallery");
}
/* The grid-showcase template: 4-8 photos, optional cover title. */
function isFourGridTpl() {
  return !!(ST.template && ST.template.pipeline === "four_grid");
}

/* The fast-cut template: 3-12 photos, static beat-cut montage. */
function isFastCutTpl() {
  return !!(ST.template && ST.template.pipeline === "fast_cut");
}

/* The cinematic template: 6-12 photos, beat-synced whoosh + rotation/pan mix. */
function isCinematicTpl() {
  return !!(ST.template && ST.template.pipeline === "cinematic_cut");
}

/* The three-screen template: 5-12 photos, stacked bands opening + per-photo move. */
function isThreeScreenTpl() {
  return !!(ST.template && ST.template.pipeline === "three_screen");
}

/* The TOP 5 Countdown template: 3-8 product photos, maroon rank number + style SKU. */
function isTop5Tpl() {
  return !!(ST.template && ST.template.pipeline === "top5");
}

/* The New Collection carousel: 2-8 photos, side cards smaller -> full size at centre. */
function isCarouselTpl() {
  return !!(ST.template && ST.template.pipeline === "carousel");
}

/* ---------------- tiny helpers ---------------- */

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  let j = {};
  try { j = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok || j.error) throw new Error(j.error || ("Request failed (" + r.status + ")"));
  return j;
}

function toast(msg, ms = 3000) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(t._t);
  t._t = setTimeout(() => (t.hidden = true), ms);
}

const fmt1 = (n) => (Math.round(n * 10) / 10).toString();

function pollJob(jid, onTick, maxMs = 26 * 60 * 1000) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const timer = setInterval(async () => {
      let job = null;
      try { ({ job } = await api("/api/job?id=" + encodeURIComponent(jid))); } catch (e) { /* transient */ }
      if (!job) return;
      if (job.status === "running") {
        if (Date.now() - t0 > maxMs) {
          clearInterval(timer);
          reject(new Error("This step took too long and was stopped. Please try again."));
          return;
        }
        onTick && onTick(job);
        return;
      }
      clearInterval(timer);
      if (job.status === "error") reject(new Error(job.error || "The task failed"));
      else resolve(job.result || {});
    }, 1500);
  });
}

function photoUrl(p) {
  const file = p.replace(/\\/g, "/").split("/").pop();
  return "/media/" + encodeURIComponent(ST.project.id) + "/uploads/" + encodeURIComponent(file);
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* Template copy may use either "label" (new, English) or "zh" (legacy). */
function moveLabel(m) { return m && (m.label || m.zh) ? (m.label || m.zh) : ""; }
function movesText(t) {
  return (t.moves || []).map(moveLabel).filter(Boolean).join(" · ");
}

/* ---------------- step navigation ---------------- */

function goStep(n) {
  [1, 2, 3, 4, 5].forEach((i) => { const el = $("#step" + i); if (el) el.hidden = i !== n; });
  $$(".rail-item").forEach((el) => {
    const k = +el.dataset.step;
    el.classList.toggle("on", k === n);
    el.classList.toggle("done", k < n);
  });
  renderAside();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------------- session / project ---------------- */

async function openSession() {
  let prev = null;
  try { prev = sessionStorage.getItem(SESSION_KEY); } catch (e) {}
  if (prev) {
    try { sessionStorage.removeItem(SESSION_KEY); } catch (e) {}
    api("/api/session/discard", { id: prev }).catch(() => {});
  }
  const { project } = await api("/api/project", { name: "Studio session" });
  try { sessionStorage.setItem(SESSION_KEY, project.id); } catch (e) {}
  ST.project = project;
}

async function startOver() {
  ST.photos = [];
  ST.template = null;
  ST.ratio = RATIOS[0].id;
  ST.rendered = false;
  $("#result").hidden = true;
  $("#renderErr").hidden = true;
  $("#renderBox").hidden = false;
  $("#s4Title").textContent = "Creating Your Video";
  $("#tplPick").innerHTML = "";
  renderPhotos();
  renderTplChips();
  renderGrid();
  updateTplSelection();
  renderRatios();
  goStep(1);
  await openSession();
}

/* ============================================================================
   STEP 1 · template library
   ========================================================================== */

let TPL_ALL = { categories: [], templates: [] };
let TPL_CAT = "all";
let TPL_Q = "";

async function loadTemplates() {
  try {
    TPL_ALL = await api("/api/templates");
  } catch (e) {
    TPL_ALL = { categories: [], templates: [] };
  }
  renderTplChips();
  renderGrid();
  // Only one template in the library: select it, and hide the category/search bar
  // (nothing to filter). Two or more → the bar comes back on its own.
  const list = TPL_ALL.templates || [];
  const bar = document.querySelector(".tpl-bar");
  if (bar) bar.style.display = list.length <= 1 ? "none" : "";
  if (list.length === 1 && (!ST.template || ST.template.id !== list[0].id)) {
    chooseTemplate(list[0].id);
  }
}

function renderTplChips() {
  const cats = TPL_ALL.categories && TPL_ALL.categories.length
    ? TPL_ALL.categories
    : [{ id: "all", name: "All" }];
  $("#tplChips").innerHTML = cats.map((c) =>
    '<button type="button" class="chip' + (c.id === TPL_CAT ? " on" : "") +
    '" data-cat="' + esc(c.id) + '">' + esc(c.name) + "</button>"
  ).join("");
}

function filteredTemplates() {
  const q = TPL_Q.trim().toLowerCase();
  return (TPL_ALL.templates || []).filter((t) => {
    if (TPL_CAT !== "all" && t.category !== TPL_CAT) return false;
    if (!q) return true;
    const hay = (t.name + " " + (t.desc || "") + " " + styleName(t) + " " +
      movesText(t)).toLowerCase();
    return hay.indexOf(q) >= 0;
  });
}

/* Template style label shown as a tag on the template card. */
function styleName(t) {
  const s = (TPL_ALL.styles || []).find((x) => x.id === t.style);
  return s ? s.name : "";
}

function renderGrid() {
  const list = filteredTemplates();
  const grid = $("#tplGrid");
  grid.classList.toggle("single", list.length === 1);
  if (!list.length) {
    grid.innerHTML = '<div class="tpl-empty">No template matches that. Try another category or search term.</div>';
    return;
  }
  grid.innerHTML = list.map((t) => {
    const sel = ST.template && ST.template.id === t.id;
    return (
      '<article class="tpl-card' + (sel ? " sel" : "") + '" data-id="' + esc(t.id) + '">' +
        '<div class="tpl-thumb" data-act="preview" data-id="' + esc(t.id) + '">' +
          '<video src="' + esc(t.cover) + '"' + (t.poster ? ' poster="' + esc(t.poster) + '"' : '') + ' muted loop playsinline preload="metadata"></video>' +
          '<button class="tpl-play" data-act="preview" data-id="' + esc(t.id) + '" type="button" aria-label="Preview">▷</button>' +
          (sel ? '<span class="tpl-check">✓ Selected</span>' : "") +
        '</div>' +
        '<div class="tpl-body">' +
          '<div class="tpl-head">' +
            '<h4>' + esc(t.name) + "</h4>" +
            (styleName(t) ? '<span class="tpl-style s-' + esc(t.style || "") + '">' + esc(styleName(t)) + "</span>" : "") +
            '<span class="tpl-moves">' + esc(movesText(t)) + "</span>" +
          "</div>" +
          '<p class="tpl-desc">' + esc(t.desc || "") + "</p>" +
          // 说明区不再放「Preview」按钮（用户 2026-10-14）：封面本身就是预览——
          // 悬停即静音播放，点封面可放大看（灯箱）。这里只留「选择」这一个动作。
          '<div class="tpl-actions">' +
            '<button class="tpl-btn solid" data-act="choose" data-id="' + esc(t.id) + '" type="button">' + (sel ? "Selected" : "Select") + "</button>" +
          "</div>" +
        "</div>" +
      "</article>"
    );
  }).join("");
}

function chooseTemplate(id) {
  const t = (TPL_ALL.templates || []).find((x) => x.id === id);
  if (!t) return;
  const changed = !ST.template || ST.template.id !== t.id;
  ST.template = t;
  // A template carries the format it is framed for; adopting it when the
  // selection actually changes keeps the client's own pick from being reset.
  if (changed) ST.ratio = defaultRatio();
  renderGrid();
  updateTplSelection();
  renderRatios();
}

function updateTplSelection() {
  const ok = !!ST.template;
  $("#btnToStep2").disabled = !ok;
  $("#step1Hint").textContent = ok
    ? "Selected: " + ST.template.name
    : "Pick a template first.";
}

function openTplPreview(id) {
  const t = (TPL_ALL.templates || []).find((x) => x.id === id);
  if (!t) return;
  const v = $("#tplLbVideo");
  try { v.pause(); } catch (e) {}
  v.removeAttribute("src");
  try { v.load(); } catch (e) {}
  v.src = t.cover;
  // Opened by a click, so the browser allows playback with sound (the sample has music).
  // Should autoplay-with-audio ever be refused, fall back to a muted play rather than nothing.
  v.play().catch(() => { v.muted = true; v.play().catch(() => {}); });
  $("#tplLbCap").textContent = t.name + (movesText(t) ? " · " + movesText(t) : "");
  $("#tplLightbox").hidden = false;
}

function closeTplPreview() {
  const v = $("#tplLbVideo");
  try { v.pause(); } catch (e) {}
  v.removeAttribute("src");
  try { v.load(); } catch (e) {}
  $("#tplLightbox").hidden = true;
}

/* ============================================================================
   STEP 2 · upload photos (apply the template's camera work)
   ========================================================================== */

async function gotoStep2() {
  if (!ST.template) { toast("Pick a template first."); return; }
  const t = ST.template;
  const isCut = t.pipeline === "card_pan";
  const isMulti = t.pipeline === "multi_suite";
  const isScroll = t.pipeline === "scroll_gallery";
  const isFour = isFourGridTpl();
  const isFast = isFastCutTpl();
  const isCin = isCinematicTpl();
  const isThree = isThreeScreenTpl();
  const isTop5 = isTop5Tpl();
  const isCar = isCarouselTpl();
  $("#tplPick").innerHTML =
    '<div class="pick">' +
      '<div class="pick-cover"><video src="' + esc(t.cover) + '" muted loop playsinline></video></div>' +
      '<div class="pick-info">' +
        '<div class="pick-name">' + esc(t.name) +
          (styleName(t) ? ' <span class="tpl-style s-' + esc(t.style || "") + '">' + esc(styleName(t)) + "</span>" : "") +
        "</div>" +
        '<div class="pick-moves">' + esc(movesText(t)) + "</div>" +
        '<div class="pick-note">' + (isCut
          ? "Upload 1\u20138 product photos \u2014 every photo gets its own detected detail close-ups, stitched into one video. More photos make a longer video."
          : isMulti
          ? "Upload 2\u20138 collection photos \u2014 one photo per design. You can name each design under its photo."
          : isScroll
          ? "Upload 6\u20138 wedding stationery photos and watch them scroll in two opposing columns\u2014a clean, text-free showcase of your coordinated wedding details."
          : isFour
          ? "Upload 4\u20138 wedding stationery photos. The first four open as a clean 2\u00D72 grid, then highlight each piece with quick cuts to showcase a coordinated wedding look."
          : isFast
          ? "Upload 5\u201312 photos. Every photo plays three times in a beat-synced cut."
          : isCin
          ? "Upload 6\u201312 photos. The cut lands on the beat \u2014 a few shots rotate, the rest pan in alternating directions."
        : isThree
        ? "Upload 5\u201312 wedding photos. The opening reveals three stacked bands, then every photo gets its own camera move."
        : isTop5
        ? "Upload 3\u20138 product photos. A title card opens, then each photo gets its own camera move with a maroon rank number and its style SKU \u2014 fill in the SKU and rank under each photo."
        : isCar
        ? "Upload 2\u20138 collection photos. Each design slides to the centre at full size while the others wait smaller at the edges \u2014 fill in the title lines and your website above."
        : "Upload your product photos and this camera-work sequence is applied automatically.") + "</div>" +
      "</div>" +
    "</div>";
  const pv = $("#tplPick").querySelector("video");
  if (pv) pv.play().catch(() => {});
  const tw = $("#t5TitleWrap");            // opening title fields: TOP 5 only
  if (tw) tw.hidden = !isTop5;
  const cw = $("#carFieldsWrap");          // carousel text fields: New Collection only
  if (cw) cw.hidden = !isCar;
  goStep(2);
}

const SHOTS = [
  { v: "detail", t: "Detail" },
  { v: "full",   t: "Full" },
  { v: "rotate", t: "Rotate" },
];

function renderPhotos() {
  const multi = isMultiTpl();
  const top5 = isTop5Tpl();
  // keep ST.labels / ST.shots / ST.focuses aligned with ST.photos no matter how the list changed
  while (ST.labels.length < ST.photos.length) ST.labels.push("");
  ST.labels.length = ST.photos.length;
  while (ST.shots.length < ST.photos.length) ST.shots.push("");
  ST.shots.length = ST.photos.length;
  while (ST.focuses.length < ST.photos.length) ST.focuses.push(null);
  ST.focuses.length = ST.photos.length;
  while (ST.skus.length < ST.photos.length) ST.skus.push("");
  ST.skus.length = ST.photos.length;
  while (ST.nums.length < ST.photos.length) ST.nums.push("");
  ST.nums.length = ST.photos.length;
  const n = ST.photos.length;
  $("#photos").innerHTML = ST.photos.map((p, i) => {
    const shotSel = multi
      ? '<div class="shot-sel" title="Camera plan for this photo">' +
        SHOTS.map((s) =>
          '<button type="button" class="shot-opt' + (ST.shots[i] === s.v ? " on" : "") +
          '" data-act="shot" data-i="' + i + '" data-shot="' + s.v + '">' + s.t + "</button>"
        ).join("") + "</div>"
      : "";
    return '<div class="photo" data-i="' + i + '" draggable="true">' +
      '<div class="thumb"><img src="' + photoUrl(p.path) + '" alt="" loading="lazy">' +
        '<span class="ord">' + (i + 1) + "</span>" +
        '<span class="grip" draggable="true" data-act="grip" data-i="' + i + '" title="Drag to reorder" aria-label="Drag to reorder">\u283F</span>' +
        '<div class="hov">' +
          '<button class="ic" data-act="move-left" data-i="' + i + '" title="Move left" aria-label="Move left"' + (i === 0 ? " disabled" : "") + '>&#8592;</button>' +
          '<button class="ic" data-act="remove" data-i="' + i + '" title="Remove" aria-label="Remove">&times;</button>' +
          '<button class="ic" data-act="move-right" data-i="' + i + '" title="Move right" aria-label="Move right"' + (i === n - 1 ? " disabled" : "") + '>&#8594;</button>' +
        "</div>" +
      "</div>" +
      (multi
        ? '<input class="name-input" data-act="label" data-i="' + i + '" type="text" maxlength="40" ' +
          'placeholder="' + (ST._recogPend[p.path] ? "Recognizing design name\u2026" : "Design name (optional)") + '" value="' + esc(ST.labels[i] || "") + '">' +
          shotSel
        : top5
        ? '<div class="top5-fields">' +
          '<input class="name-input" data-act="sku" data-i="' + i + '" type="text" maxlength="40" ' +
            'placeholder="Style SKU (e.g. WSNT003)" value="' + esc(ST.skus[i] || "") + '">' +
          '<input class="name-input rank-input" data-act="num" data-i="' + i + '" type="text" maxlength="3" ' +
            'placeholder="Rank (blank = auto)" inputmode="numeric" value="' + esc(ST.nums[i] || "") + '">' +
          '</div>'
        : "") +
    "</div>";
  }).join("");
  renderAside();
  updateGenBtn();
}

function setShot(i, v) {
  if (isNaN(i)) return;
  // click the active one again → clear back to "auto" (let the template decide)
  ST.shots[i] = (ST.shots[i] === v) ? "" : v;
  renderPhotos();
}

function updateGenBtn() {
  const multi = isMultiTpl();
  const scroll = isScrollTpl();
  const four = isFourGridTpl();
  const fastcut = isFastCutTpl();
  const cin = isCinematicTpl();
  const three = isThreeScreenTpl();
  const top5 = isTop5Tpl();
  const car = isCarouselTpl();
  const n = ST.photos.length;
  // Per-template minimums must match the server endpoints:
  //   multi-suite (Collection)  → 2  (server.py /api/multi-suite: len(srcs) < 2)
  //   scroll-gallery (Colour)    → 6  (server.py /api/scroll-gallery: len(srcs) < 6)
  //   four-grid (Grid)           → 4  (server.py /api/four-grid: len(srcs) < 4)
  //   fast-cut (Fast Cut)        → 5  (server.py /api/fast-cut: len(srcs) < 5)
  //   cinematic-cut (Cinematic)  → 6  (server.py /api/mix-cut: len(srcs) < 6)
  //   three-screen (Three-Screen)→ 5  (server.py /api/three-screen: len(srcs) < 5)
  //   carousel (New Collection)  → 2  (server.py /api/carousel: len(srcs) < 2)
  const min = multi ? 2 : scroll ? 6 : four ? 4 : fastcut ? 5 : cin ? 6 : three ? 5 : top5 ? 3 : car ? 2 : 1;
  $("#btnToStep3").disabled = n === 0 || ST.busy || n < min;
  let eta = "";
  if (n) {
    eta = n + (n > 1 ? " photos" : " photo") +
      " \u00B7 using the \u201C" + (ST.template ? ST.template.name : "") + "\u201D template";
    if (multi && n < 2) eta += " \u00B7 add one more photo to continue";
    if (scroll && n < 6) eta += " \u00B7 add at least 6 photos to continue";
    if (four && n < 4) eta += " \u00B7 add at least 4 photos to continue";
    if (fastcut && n < 5) eta += " \u00B7 add at least 5 photos to continue";
    if (cin && n < 6) eta += " \u00B7 add at least 6 photos to continue";
    if (three && n < 5) eta += " \u00B7 add at least 5 photos to continue";
    if (top5 && n < 3) eta += " \u00B7 add at least 3 photos to continue";
    if (car && n < 2) eta += " \u00B7 add one more photo to continue";
    if ((multi || scroll || four) && n >= 8) eta += " \u00B7 maximum reached";
    if (fastcut && n >= 12) eta += " \u00B7 maximum reached";
    if (cin && n >= 12) eta += " \u00B7 maximum reached";
    if (three && n >= 12) eta += " \u00B7 maximum reached";
    if (top5 && n >= 8) eta += " \u00B7 maximum reached";
    // Fast Cut: every photo plays three times at one beat (0.592s) each —
    // show the auto-growing duration right here so the count/length link is clear.
    if (fastcut) eta += " \u00B7 about " + fmt1(n * 3 * 0.592) + " seconds";
    // Cinematic Cut: duration scales with photo count (4.8 + 0.56×N s), cut to the beat.
    if (cin && n >= 6) eta += " \u00B7 about " + fmt1(4.8 + 0.56 * n) + " seconds, cut to the beat";
  }
  $("#eta").textContent = eta;
}

/* Upload pipeline: shrink each image in the browser (longest edge <= 2000px)
   before upload so multi-megabyte phone photos don't saturate the network, and
   push several files concurrently while revealing each thumbnail as it lands —
   instead of stalling silently until the whole batch finishes. */
async function pool(items, limit, worker) {
  let i = 0;
  const runners = [];
  const next = async () => {
    while (i < items.length) {
      const idx = i++;
      await worker(items[idx], idx);
    }
  };
  const n = Math.min(limit, items.length);
  for (let k = 0; k < n; k++) runners.push(next());
  await Promise.all(runners);
}

function compressImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const maxDim = 2000;
      let w = img.width, h = img.height;
      const scale = Math.min(1, maxDim / Math.max(w, h));
      w = Math.max(1, Math.round(w * scale));
      h = Math.max(1, Math.round(h * scale));
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const ctx = cv.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      const isPng = file.type === "image/png";
      const mime = isPng ? "image/png" : "image/jpeg";
      const q = isPng ? undefined : 0.92;
      cv.toBlob((blob) => {
        if (!blob) { reject(new Error("image compress failed")); return; }
        resolve({ blob: blob, ext: isPng ? "png" : "jpg" });
      }, mime, q);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("cannot decode image")); };
    img.src = url;
  });
}

async function uploadBlob(name, blob, ext) {
  const stem = (name || "photo").replace(/\.[^.]+$/, "");
  const fname = stem + "." + ext;
  const url = "/api/upload?project=" + encodeURIComponent(ST.project.id) +
    "&name=" + encodeURIComponent(fname) + "&kind=upload";
  const r = await fetch(url, { method: "POST", body: blob });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || "Upload failed");
  ST.project = j.project;
  return j.entry;
}

function showUprog(text) {
  const e = document.getElementById("uprog");
  if (e) { e.textContent = text; e.hidden = false; }
}
function hideUprog() {
  const e = document.getElementById("uprog");
  if (e) e.hidden = true;
}

async function uploadOne(file) {
  const c = await compressImage(file);
  return uploadBlob(file.name, c.blob, c.ext);
}

async function addFiles(files) {
  const list = Array.from(files || []).filter((f) => /^image\//.test(f.type));
  if (!list.length) { toast("Please choose JPG or PNG images."); return; }
  const cap = (isFastCutTpl() || isThreeScreenTpl()) ? 12
    : (isMultiTpl() || isScrollTpl() || isFourGridTpl() || isTop5Tpl()) ? 8 : 99;
  ST.busy = true;
  updateGenBtn();
  let accepted = ST.photos.length, skipped = 0, done = 0;
  const total = list.length;
  const worker = async (f) => {
    if (accepted >= cap) { skipped++; return; }
    accepted++;
    try {
      showUprog("Preparing " + (done + skipped + 1) + " / " + total + "...");
      const e = await uploadOne(f);
      ST.photos.push({ path: e.path, file: e.file });
      ST.labels.push("");
      done++;
      showUprog("Uploading " + done + " / " + total + "...");
      renderPhotos();
    } catch (err) {
      accepted--;
      toast("Could not upload " + f.name + ": " + err.message);
    }
  };
  await pool(list, 4, worker);
  ST.busy = false;
  hideUprog();
  renderPhotos();
  updateGenBtn();
  if (skipped) toast("You can add up to " + cap + " photos for this template - the rest were skipped.");
  if (isMultiTpl()) recognizeLabels();
}

/* Auto-fill design names for the multi-photo template: after upload, ask the
   backend to read each photo (printed text first, then a semantic tag) and drop
   the result into the matching name input — without overwriting what the user
   has already typed. */
async function recognizeLabels() {
  if (!isMultiTpl()) return;
  const need = [];
  ST.photos.forEach((p, i) => {
    if (!ST.labels[i] && !ST._recogDone[p.path]) {
      need.push({ i: i, path: p.path });
      ST._recogDone[p.path] = true; // reserve so we don't fire duplicate requests
    }
  });
  if (!need.length) return;
  // mark pending → re-render so empty inputs show a "Recognizing…" placeholder
  // (visual model takes ~10s per batch; without the hint the empty box reads
  //  as "the feature isn't live" — user feedback 2026-10-15)
  need.forEach((x) => { ST._recogPend[x.path] = true; });
  renderPhotos();
  try {
    const { job } = await api("/api/recognize-labels", { srcs: need.map((x) => x.path) });
    const res = await pollJob(job, () => {});
    const labels = (res && res.labels) || [];
    const shots = (res && res.shots) || [];
    const focuses = (res && res.focuses) || [];
    labels.forEach((lab, k) => {
      const it = need[k];
      if (!it) return;
      if (lab && !ST.labels[it.i]) {
        ST.labels[it.i] = lab;
        const inp = document.querySelector('.name-input[data-i="' + it.i + '"]');
        if (inp) inp.value = lab;
      }
      // shot plan + detail focus: only auto-fill empty slots (don't clobber manual edits)
      if (shots[k] && !ST.shots[it.i]) ST.shots[it.i] = shots[k];
      if (focuses[k] && !ST.focuses[it.i]) ST.focuses[it.i] = focuses[k];
    });
  } catch (e) {
    need.forEach((x) => { delete ST._recogDone[x.path]; }); // allow retry on failure
  } finally {
    need.forEach((x) => { delete ST._recogPend[x.path]; });
    renderPhotos();
  }
}

function removePhoto(i) {
  ST.photos.splice(i, 1);
  ST.labels.splice(i, 1);
  renderPhotos();
  updateGenBtn();
}

/* Reorder uploaded photos. `from` and `to` are array indices; every parallel
   array (labels / shots / focuses) is shifted in lock-step so the name inputs
   and camera-plan choices stay glued to their image. Reordering matters most for
   Grid Showcase, where the FIRST 4 photos become the 2×2 grid (and their
   position decides which slot each lands in). */
function reorderPhoto(from, to) {
  if (from === to || from < 0 || to < 0 || from >= ST.photos.length || to >= ST.photos.length) return;
  const move = (arr) => { const it = arr.splice(from, 1)[0]; arr.splice(to, 0, it); };
  move(ST.photos); move(ST.labels); move(ST.shots); move(ST.focuses);
  renderPhotos();
}

/* Move a photo one slot left/right (touch-friendly fallback for drag). */
function movePhoto(i, dir) {
  reorderPhoto(i, i + dir);
}

/* ============================================================================
   STEP 3 · choose format
   The camera work comes from the template; the format is the client's choice.
   The format is NOT a crop of the finished video — the pipeline re-composes the
   framing for the chosen aspect ratio (source of truth: core/card_pan.py RATIOS).
   ========================================================================== */

const RATIOS = [
  { id: "9:16", api: "9x16", shape: "r9-16", name: "Vertical · 9:16",
    note: "Ideal for Reels, Stories, and TikTok." },
  { id: "1:1",  api: "1x1",  shape: "r1-1",  name: "Square · 1:1",
    note: "Ideal for square social posts and product showcases." },
];

/* Format the template is framed for (templates.json → recommended_ratio). */
function defaultRatio() {
  const r = ST.template && ST.template.recommended_ratio;
  return RATIOS.some((x) => x.id === r) ? r : RATIOS[0].id;
}

/* UI format id → pipeline format id ("9:16" → "9x16"). */
function ratioApi(id) {
  const r = RATIOS.find((x) => x.id === id);
  return (r || RATIOS[0]).api;
}

/* step 2 → step 3 */
function gotoStep3() {
  if (!ST.photos.length) { toast("Add a photo first."); return; }
  if (!RATIOS.some((x) => x.id === ST.ratio)) ST.ratio = defaultRatio();
  renderRatios();
  goStep(3);
}

/* step 3 → step 4 */
function gotoStep4() {
  // Keep the format the user picked in step 3; only fall back to the template
  // default when the current pick is somehow invalid (fixes 1:1 reverting to 9:16).
  if (!RATIOS.some((x) => x.id === ST.ratio)) ST.ratio = defaultRatio();
  renderStep4();
  goStep(4);
}

function renderRatios() {
  $("#ratios").innerHTML = RATIOS.map((r) =>
    '<button type="button" class="ratio' + (r.id === ST.ratio ? " on" : "") +
    '" data-ratio="' + esc(r.id) + '" aria-pressed="' + (r.id === ST.ratio) + '">' +
      '<span class="rshape ' + r.shape + '" aria-hidden="true"></span>' +
      '<span class="rname">' + esc(r.name) + (r.id === defaultRatio() ? " \u00B7 Recommended" : "") + "</span>" +
      '<span class="rdesc">' + esc(r.note) + "</span>" +
    "</button>"
  ).join("");
  const tag = $("#ratioKeep");
  if (tag) {
    tag.textContent = ST.template
      ? "Your photo is re-framed for the format you pick \u2014 nothing is cropped afterwards. " +
        "The template\u2019s camera work is applied either way."
      : "";
  }
}

/* step 4 · captions & voiceover toggle state */
function capOn() {
  return !!($("#capOn") && $("#capOn").checked);
}

function voPayload() {
  if (!$("#voOn") || !$("#voOn").checked) return null;
  return {
    on: true,
    selling: (($("#voSelling") && $("#voSelling").value.trim()) || ""),
    cta: (($("#voCta") && $("#voCta").value) || "view"),
  };
}

function renderStep4() {
  const t = ST.template;
  const n = ST.photos.length;
  // captions fields depend on the template's pipeline
  const pipe = t ? t.pipeline : "";
  const cin = pipe === "cinematic_cut";
  const three = pipe === "three_screen";
  // Cinematic Cut: length scales with photo count (4.8 + 0.56×N s) — show it live.
  // Three-Screen: opener 8 beats + per-photo 3 beats (last 4) at 0.5923s/beat.
  const secs = cin ? Math.round((4.8 + 0.56 * n) * 10) / 10
                   : three ? Math.round((3 * n + 9) * 0.5923 * 10) / 10
                   : ((t && t.target_secs) || 15);
  const info = $("#voTplInfo");
  if (info) {
    info.textContent = (t ? t.name + " \u00B7 " : "") +
      n + (n === 1 ? " photo" : " photos") + " \u00B7 " +
      secs + "-second video \u00B7 " + (ST.ratio || "9:16");
  }
  // Cinematic Cut now offers captions & voiceover like the other templates —
  // the standard step-4 heading/lede applies to it too (set for every template).
  const eb = document.querySelector("#step4 .eyebrow");
  const h2 = document.querySelector("#step4 h2");
  const ld = document.querySelector("#step4 .lede");
  if (eb && h2 && ld) {
    eb.textContent = "04 / Captions & voiceover";
    h2.textContent = "Add Captions & Voiceover";
    ld.textContent = "Introduce your product with on-screen text, an AI voiceover, or both.";
  }
  const fm = $("#capFieldsMulti"); if (fm) fm.hidden = pipe !== "multi_suite";
  const fg = $("#capFieldsGrid"); if (fg) fg.hidden = pipe !== "four_grid";
  const fl = $("#capFieldsLine"); if (fl) fl.hidden = !(pipe === "card_pan" || pipe === "scroll_gallery");
  const ff = $("#capFieldsFast"); if (ff) ff.hidden = pipe !== "fast_cut";
  const fc2 = $("#capFieldsCin"); if (fc2) fc2.hidden = !cin;
  const ft = $("#capFieldsThree"); if (ft) ft.hidden = !three;
  const cf = $("#capFields"); if (cf) cf.hidden = !capOn();
  const vf = $("#voFields"); if (vf) vf.hidden = !($("#voOn") && $("#voOn").checked);
  const bf = $("#bgmFields"); if (bf) bf.hidden = false;   // 2026-10-27 BGM picker: all templates
  loadBgmList();                                            // lazy-load track list once
  prefillFastCaption();
  if (cin && capOn()) {
    const el = $("#cinText");
    if (el && el.value.trim() === "") el.value = "the details, the day";
  }
}

/* When the user turns on "Add Captions" for Fast Cut, surface the default
   italic caption text so it's visible/editable right away (instead of an
   empty field with only a placeholder). Only fills when the field is empty,
   so any text the user types is never overwritten. */
function prefillFastCaption() {
  if (!isFastCutTpl() || !capOn()) return;
  const el = $("#fcText");
  if (el && el.value.trim() === "") el.value = "the details, the day";
}

/* ============================================================================
   STEP 4 · create + export
   ========================================================================== */

function setResultVideo(url) {
  const v = $("#resultVideo");
  const box = $("#videoErr");
  box.hidden = true;
  box.textContent = "";
  v.hidden = false;
  v.onerror = null;
  try { v.pause(); } catch (e) {}
  v.removeAttribute("src");
  try { v.load(); } catch (e) {}
  v.src = url;
  // If the browser refuses to play it inline, say so plainly and hand over the file
  // instead of leaving an empty black frame on screen.
  v.onerror = () => {
    v.hidden = true;
    box.hidden = false;
    box.append(
      document.createTextNode("Your browser could not play this video inline. Download it below, or "),
      (() => { const a = document.createElement("a"); a.href = url; a.target = "_blank"; a.rel = "noopener"; a.textContent = "open the video in a new tab"; return a; })(),
      document.createTextNode(".")
    );
  };
}

/* Font picked in step 4 (shared by every template's on-screen text).
   Returns the fonts.py registry key; empty/select missing → default italic. */
function capFontKey() {
  const el = $("#capFont");
  return (el && el.value) || "italic";
}

/* ---- Background music (2026-10-27): template default / library pick / upload ---- */
let BGM_LOADED = false;
async function loadBgmList() {
  if (BGM_LOADED) return;
  BGM_LOADED = true;             // set first: a failed load shouldn't retry on every keystroke
  const sel = $("#bgmSel");
  if (!sel) return;
  try {
    const j = await api("/api/bgm");
    const keep = sel.value;
    // 2026-10-27 产品口径：客户只能选「模版默认 / 无 BGM / 自己上传的曲」，
    // 曲库曲目不再露出（服务端 _bgm_pick 也只接受 user_ 前缀）。
    (j.tracks || [])
      .filter((tr) => (tr.license || "").indexOf("User upload") === 0)
      .forEach((tr) => {
        const o = document.createElement("option");
        o.value = tr.name;
        o.textContent = (tr.credit || tr.name) + " \u00B7 my upload";
        sel.appendChild(o);
      });
    if (keep) sel.value = keep;  // restore after re-load (e.g. right after upload)
  } catch (e) { /* list stays "Template default" — server falls back anyway */ }
}
function bgmPayload() {
  const el = $("#bgmSel");
  const v = el && el.value;
  return v ? v : null;
}
/* 把步骤 4 的 BGM 选择写进生成请求体：
   "" → 不传（服务端用模版固定曲）；"__none__" → silent（无 BGM）；其余 → 上传曲名。 */
function applyBgm(body) {
  const v = bgmPayload();
  if (v === "__none__") { body.silent = true; return; }
  if (v) body.bgm = v;
}
async function uploadBgm(file) {
  const hint = $("#bgmHint");
  const show = (t) => { if (hint) { hint.textContent = t; hint.hidden = false; } };
  if (!file) return;
  if (file.size > 30 * 1024 * 1024) { show("Audio too large — please keep it under 30MB."); return; }
  show("Uploading “" + file.name + "”…");
  const b64 = await new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(String(fr.result).split(",")[1] || "");
    fr.onerror = () => rej(new Error("read failed"));
    fr.readAsDataURL(file);
  });
  try {
    const j = await api("/api/bgm/upload", { name: file.name, data_b64: b64 });
    BGM_LOADED = false;          // force re-fetch so the new track shows up
    await loadBgmList();
    const sel = $("#bgmSel");
    if (sel && j.name) sel.value = j.name;
    show("Added — “" + (j.label || file.name) + "” is selected for this video.");
  } catch (e) {
    show("Upload failed: " + (e.message || "unknown error"));
  }
}

async function generate() {
  if (!ST.template || !ST.photos.length) { toast("Pick a template and add a photo first."); return; }
  if (ST.busy) return;
  ST.busy = true;
  ST.rendered = false;
  goStep(5);
  $("#renderErr").hidden = true;
  $("#result").hidden = true;
  $("#renderBox").hidden = false;
  $("#s4Title").textContent = "Creating Your Video";
  $("#s4Lede").textContent = "Your video is being created. Please keep this tab open until it\u2019s ready.";
  const t0 = Date.now();

  // Route by template pipeline: card_pan (1-8 photos, detected pieces per photo) or cammove.
  const t = ST.template;
  const srcs = ST.photos.map((p) => p.path);
  let job;
  try {
    if (t.pipeline === "card_pan") {
      const body = {
        project: ST.project.id,
        // 2026-10-27 multi-photo: send every photo — each gets its own detected
        // detail shots and the clips are stitched (photo 1 keeps the hook shot,
        // the last keeps the closing pull-back).
        srcs: srcs,
        src: srcs[0],
        // Format chosen in step 3 → pipeline format id. The server (and the plan
        // builder behind it) re-composes the framing for this format.
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      if (t.plan) body.plan = t.plan;
      if (t.target_secs) body.target_secs = t.target_secs;
      // step 4 toggles: this template has no native text → captions = burned line
      // (skipped when voiceover is on: its subtitles already are on-screen text).
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      if (capOn() && !vo) {
        body.captions = { on: true, text: (($("#capLine") && $("#capLine").value.trim()) || ""),
                          font: capFontKey() };
      }
      setRender(2, "Reading the pieces in your photo, then creating your video in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/card-pan", body));
    } else if (t.pipeline === "multi_suite") {
      // Collection Showcase: every photo is one design; names + shot plan are optional.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        // Captions off → don't send design names either, so the recognized
        // content from step 2 (the per-photo nameplates) stays out of the video.
        labels: capOn() ? ST.labels.slice(0, srcs.length) : [],
        shots: ST.shots.slice(0, srcs.length),
        focuses: ST.focuses.slice(0, srcs.length),
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      // Captions off → empty title = no opening title (the old "Our Collection"
      // default only applies when captions are on and the field is left blank).
      const capOnNow = capOn();
      body.title = capOnNow ? (($("#colTitle") && $("#colTitle").value.trim()) || "") : "";
      if (capOnNow) body.font = capFontKey();
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      setRender(2, "Building your collection video in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/multi-suite", body));
    } else if (t.pipeline === "scroll_gallery") {
      // Scrolling Gallery: every photo is shown; first half left (up), rest right (down).
      // No names / shots — pure scrolling showcase.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      if (capOn() && !vo) {
        body.captions = { on: true, text: (($("#capLine") && $("#capLine").value.trim()) || ""),
                          font: capFontKey() };
      }
      setRender(2, "Building your scrolling gallery in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/scroll-gallery", body));
    } else if (t.pipeline === "four_grid") {
      // Grid Showcase: first four photos open as a 2x2 grid, then every photo quick-cuts.
      // Cover title is optional (Add Captions toggle in step 4).
      const body = {
        project: ST.project.id,
        srcs: srcs,
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      if (capOn()) {
        body.show_text = true;
        body.title = (($("#fgTitle") && $("#fgTitle").value.trim()) || "");
        body.subtitle = (($("#fgSub") && $("#fgSub").value.trim()) || "");
        body.font = capFontKey();
      } else {
        body.show_text = false;
      }
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      setRender(2, "Building your grid video in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/four-grid", body));
    } else if (t.pipeline === "fast_cut") {
      // Fast Cut: every photo plays three times in a beat-synced quick cut;
      // duration grows with the photo count (3N × 0.592s). Caption is optional
      // (Add Captions toggle in step 4); blank field → server default text.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      if (capOn()) {
        body.show_text = true;
        body.text = (($("#fcText") && $("#fcText").value.trim()) || "");
        body.font = capFontKey();
      } else {
        body.show_text = false;
      }
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      setRender(2, "Building your fast-cut video in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/fast-cut", body));
    } else if (t.pipeline === "cinematic_cut") {
      // Cinematic Cut: 6–12 photos, beat-synced cut — a zoom-whoosh opens the
      // second shot exactly on the beat, a few shots rotate, the rest pan.
      // Caption & voiceover are optional (step-4 toggles), same as Fast Cut.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      if (capOn()) {
        body.show_text = true;
        body.text = (($("#cinText") && $("#cinText").value.trim()) || "");
        body.font = capFontKey();
      } else {
        body.show_text = false;
      }
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      setRender(2, "Building your cinematic cut in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/mix-cut", body));
    } else if (t.pipeline === "three_screen") {
      // Three-Screen Story: 5-12 photos → three stacked bands opening + per-photo move.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      if (capOn()) {
        body.captions = { on: true, text: (($("#tsText") && $("#tsText").value.trim()) || ""),
                          font: capFontKey() };
      }
      const vo = voPayload();
      if (vo) body.voiceover = vo;
      setRender(2, "Building your three-screen story in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/three-screen", body));
    } else if (t.pipeline === "top5") {
      // TOP 5 Countdown: 3-8 product photos, a maroon rank number + style SKU
      // beside it on every shot. SKU/rank are optional per photo; blank rank = auto
      // countdown (n, n-1, … 01 from first to last photo).
      const body = {
        project: ST.project.id,
        srcs: srcs,
        skus: ST.skus.slice(0, srcs.length),
        nums: ST.nums.slice(0, srcs.length),
        title_main: ($("#t5Title") && $("#t5Title").value.trim()) || "TOP 5",
        title_sub: ($("#t5Sub") ? $("#t5Sub").value.trim() : ""),
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      setRender(2, "Building your TOP 5 countdown in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/top5", body));
    } else if (t.pipeline === "carousel") {
      // New Collection: 2-8 photos -> side-small / centre-full card carousel.
      // Title lines and footer text are editable in step 2.
      const body = {
        project: ST.project.id,
        srcs: srcs,
        title1: ($("#carTitle1") && $("#carTitle1").value.trim()) || "Our Wedding Invitations",
        title2: ($("#carTitle2") && $("#carTitle2").value.trim()) || "NEW COLLECTION",
        footer: ($("#carFooter") ? $("#carFooter").value.trim() : ""),
        ratio: ratioApi(ST.ratio),
        fps: 30,
      };
      setRender(2, "Building your collection carousel in " + ST.ratio + "\u2026");
      applyBgm(body);
      ({ job } = await api("/api/carousel", body));
    } else {
      const body = { project: ST.project.id, ratio: ST.ratio, moves: t.seq || [] };
      if (srcs.length >= 2) body.sources = srcs;
      else body.src = srcs[0];
      setRender(2, "Applying the camera work of \u201C" + esc(t.name) + "\u201D…");
      ({ job } = await api("/api/cameramove/render", body));
    }
    const res = await pollJob(job, (j) => setRender(j.progress || 5, j.msg || "Creating…"));
    const secs = res.secs || 0;
    const name = "invitation-promo-" + fmt1(secs) + "s.mp4";
    const video = res.video || (res.file ? "/media/" + ST.project.id + "/cameramove/" + res.file.split("/").pop() : "");
    setResultVideo(video);
    $("#btnDownload").href = video + (video.indexOf("?") >= 0 ? "&" : "?") + "dl=" + encodeURIComponent(name);
    $("#btnDownload").setAttribute("download", name);
    // Below the video: duration · format · template (+ optional add-ons).
    $("#resultMeta").textContent = fmt1(secs) + " seconds \u00B7 " + ST.ratio + " \u00B7 " + t.name +
      (res.captions ? " \u00B7 Captions" : "") +
      (res.voiceover ? " \u00B7 Voiceover" : "");
    $("#renderBox").hidden = true;
    $("#s4Title").textContent = "Your video is ready";
    $("#s4Lede").textContent = "Preview your video, then download it to share.";
    ST.rendered = true;
    renderAside();
    $("#result").hidden = false;
  } catch (e) {
    $("#renderBox").hidden = true;
    $("#renderErr").hidden = false;
    $("#renderErr").textContent = "Your video could not be created: " + e.message + " — please try again, or use a different photo.";
  } finally {
    ST.busy = false;
  }
}

function setRender(pct, msg) {
  const f = $("#renderFill");
  if (f) f.style.width = Math.max(2, Math.min(100, pct || 0)) + "%";
  const s = $("#renderStatus");
  if (s) s.textContent = msg || "";
}

/* ---------------- right summary ---------------- */

function renderAside() {
  const ul = $("#asideList");
  if (!ul) return;
  const rows = [];
  rows.push(ST.template ? "<li><b>Template</b> " + esc(ST.template.name) + "</li>" : "<li>No template yet</li>");
  rows.push("<li><b>Photos</b> " + ST.photos.length + "</li>");
  rows.push("<li><b>Format</b> " + ST.ratio + "</li>");
  if (ST.rendered) rows.push('<li class="ok">Video ready ✓</li>');
  ul.innerHTML = rows.join("");
}

/* ---------------- wiring ---------------- */

function bind() {
  // template category chips
  $("#tplChips").addEventListener("click", (e) => {
    const el = e.target.closest("[data-cat]");
    if (!el) return;
    TPL_CAT = el.dataset.cat;
    renderTplChips();
    renderGrid();
  });
  // search
  $("#tplSearch").addEventListener("input", (e) => { TPL_Q = e.target.value; renderGrid(); });
  // template card actions (preview / select)
  $("#tplGrid").addEventListener("click", (e) => {
    const el = e.target.closest("[data-act]");
    if (!el) return;
    const id = el.dataset.id;
    if (el.dataset.act === "choose") chooseTemplate(id);
    else if (el.dataset.act === "preview") openTplPreview(id);
  });
  // hover a card → its sample plays; leave → it rewinds
  $("#tplGrid").addEventListener("mouseover", (e) => {
    const thumb = e.target.closest(".tpl-thumb");
    if (!thumb) return;
    const v = thumb.querySelector("video");
    if (v) v.play().catch(() => {});
  });
  $("#tplGrid").addEventListener("mouseout", (e) => {
    const thumb = e.target.closest(".tpl-thumb");
    if (!thumb || thumb.contains(e.relatedTarget)) return;
    const v = thumb.querySelector("video");
    if (v) { try { v.pause(); v.currentTime = 0; } catch (err) {} }
  });
  // template preview lightbox
  $("#tplLbClose").addEventListener("click", closeTplPreview);
  $("#tplLightbox").addEventListener("click", (e) => {
    if (e.target.id === "tplLightbox") closeTplPreview();
  });

  // step navigation
  $$(".rail-item").forEach((el) => {
    el.addEventListener("click", () => {
      if (ST.busy) return;
      const target = +el.dataset.step;
      const cur = [1, 2, 3, 4, 5].find((i) => { const s = $("#step" + i); return s && !s.hidden; });
      if (cur && target < cur) goStep(target);
    });
  });

  // step 1 → 2
  $("#btnToStep2").addEventListener("click", gotoStep2);
  // step 2 back / → step 3
  $("#btnBack1").addEventListener("click", () => goStep(1));
  $("#btnToStep3").addEventListener("click", gotoStep3);
  // step 3 format / back / → step 4
  $("#ratios").addEventListener("click", (e) => {
    const el = e.target.closest("[data-ratio]");
    if (!el) return;
    ST.ratio = el.dataset.ratio;
    renderRatios();
    renderAside();
  });
  $("#btnBack2").addEventListener("click", () => goStep(2));
  $("#btnToStep4").addEventListener("click", gotoStep4);
  // step 4 captions & voiceover / back / create
  $("#btnBack2b").addEventListener("click", () => goStep(2));
  $("#btnToStep5").addEventListener("click", generate);

  // upload / drag & drop
  const drop = $("#drop");
  const input = $("#fileInput");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
  });
  // step 4 · captions / voiceover toggles reveal their fields
  const capOnEl = $("#capOn");
  if (capOnEl) capOnEl.addEventListener("change", renderStep4);
  const voOn = $("#voOn");
  if (voOn) {
    voOn.addEventListener("change", renderStep4);
  }
  // step 4 · background music upload (2026-10-27)
  const bgmFile = $("#bgmFile");
  if (bgmFile) bgmFile.addEventListener("change", () => {
    uploadBgm(bgmFile.files && bgmFile.files[0]);
    bgmFile.value = "";
  });

  // photo grid (reorder + remove + design-name inputs for the collection template)
  $("#photos").addEventListener("click", (e) => {
    const el = e.target.closest("[data-act]");
    if (!el || el.disabled) return;
    if (el.dataset.act === "remove") removePhoto(+el.dataset.i);
    else if (el.dataset.act === "shot") setShot(+el.dataset.i, el.dataset.shot);
    else if (el.dataset.act === "move-left") movePhoto(+el.dataset.i, -1);
    else if (el.dataset.act === "move-right") movePhoto(+el.dataset.i, +1);
  });
  $("#photos").addEventListener("input", (e) => {
    const el = e.target.closest("[data-act]");
    if (!el) return;
    const i = parseInt(el.dataset.i, 10);
    if (isNaN(i)) return;
    if (el.dataset.act === "label") ST.labels[i] = el.value;
    else if (el.dataset.act === "sku") ST.skus[i] = el.value;
    else if (el.dataset.act === "num") ST.nums[i] = el.value;
  });

  // Drag-to-reorder: grab any photo card (or its ⠿ handle) and drop it onto
  // another slot. The whole card is draggable so reordering feels natural;
  // the order sent to the renderer is exactly ST.photos order, so this controls
  // which images land in the 2×2 grid, the scroll columns, or the cut sequence.
  let dragFrom = null;
  $("#photos").addEventListener("dragstart", (e) => {
    // never hijack text selection inside name inputs / buttons
    if (e.target.closest("input, select, textarea, button")) { e.preventDefault(); return; }
    const ph = e.target.closest(".photo");
    if (!ph) { e.preventDefault(); return; }
    dragFrom = +ph.dataset.i;
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = "move";
      try { e.dataTransfer.setData("text/plain", String(dragFrom)); } catch (_) {}
    }
    ph.classList.add("dragging");
  });
  $("#photos").addEventListener("dragover", (e) => {
    const ph = e.target.closest(".photo");
    if (!ph || dragFrom === null) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
    document.querySelectorAll(".photo.dropzone").forEach((x) => x.classList.remove("dropzone"));
    ph.classList.add("dropzone");
  });
  $("#photos").addEventListener("drop", (e) => {
    const ph = e.target.closest(".photo");
    if (!ph || dragFrom === null) return;
    e.preventDefault();
    const to = +ph.dataset.i;
    const from = dragFrom;
    dragFrom = null;
    document.querySelectorAll(".photo.dropzone").forEach((x) => x.classList.remove("dropzone"));
    if (to !== from) reorderPhoto(from, to);
  });
  $("#photos").addEventListener("dragend", () => {
    dragFrom = null;
    document.querySelectorAll(".photo.dragging").forEach((x) => x.classList.remove("dragging"));
    document.querySelectorAll(".photo.dropzone").forEach((x) => x.classList.remove("dropzone"));
  });

  // make another
  $("#btnAgain").addEventListener("click", () => startOver().catch((e) => toast(e.message)));

  // setup modal
  $("#btnSetup").addEventListener("click", openSetup);
  $("#btnSetupClose").addEventListener("click", () => ($("#setupModal").hidden = true));
  $("#btnCfgCancel").addEventListener("click", () => ($("#setupModal").hidden = true));
  $("#btnCfgSave").addEventListener("click", () => saveSetup().catch((e) => toast(e.message)));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("#setupModal").hidden = true;
      closeTplPreview();
    }
  });
}

async function openSetup() {
  try {
    const { config } = await api("/api/config");
    const sb = config.storyboard_api || {};
    $("#cfgKey").value = sb.api_key || "";
    $("#cfgEp").value = sb.endpoint || "";
  } catch (e) {}
  $("#setupModal").hidden = false;
}

async function saveSetup() {
  const { config } = await api("/api/config");
  const sb = Object.assign({}, config.storyboard_api || {});
  sb.api_key = $("#cfgKey").value.trim();
  sb.endpoint = $("#cfgEp").value.trim();
  await api("/api/config", { storyboard_api: sb });
  $("#setupModal").hidden = true;
  toast("Saved.");
}

async function boot() {
  bind();
  goStep(1);
  try { await openSession(); } catch (e) { toast("Could not start a session: " + e.message, 6000); return; }
  renderPhotos();
  updateGenBtn();
  await loadTemplates();   // may auto-select the only template → sets the format
  renderRatios();
}

boot();

window.__studio = { ST, RATIOS, goStep, addFiles, generate, gotoStep2, gotoStep3, startOver, openTplPreview, ratioApi };
