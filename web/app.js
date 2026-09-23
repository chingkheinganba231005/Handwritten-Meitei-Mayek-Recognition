import { grayFromRGBA, preprocess, toDatasetScale, view } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };

let config = null;
let session = null;
let mode = "draw";           // "draw" | "upload"
let uploaded = null;         // ImageBitmap of the uploaded photo

// ---------- model ----------

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status(`${label} ${Math.round((100 * got) / total)}%`);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function load() {
  config = await (await fetch("config.json")).json();
  $("model-name").textContent = config.model_name || "the network";
  if (config.test_accuracy) $("accuracy").textContent = `${(100 * config.test_accuracy).toFixed(2)}%`;
  if (config.repo_url) $("repo").href = config.repo_url;
  const ortBase = new URL(config.ort_base || "ort/", location.href).href;
  await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = ortBase + "ort.wasm.min.js";
    s.onload = resolve;
    s.onerror = () => reject(new Error("could not load ONNX Runtime Web"));
    document.head.appendChild(s);
  });
  ort.env.wasm.wasmPaths = ortBase;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const bytes = await fetchWithProgress(config.model_url || "model.onnx",
    `Downloading the model${config.model_mb ? ` (${Math.round(config.model_mb)} MB, only the first time)` : ""}…`);
  status("Starting the model…");
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  status("Ready. Draw a character.");
  $("recognise").disabled = false;
}

// ---------- drawing ----------

const pad = $("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
let drawing = false;
let strokes = 0;

function resetPad() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  strokes = 0;
}

function pos(e) {
  const r = pad.getBoundingClientRect();
  return [((e.clientX - r.left) * pad.width) / r.width, ((e.clientY - r.top) * pad.height) / r.height];
}

pad.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = Number($("pen").value);
  const [x, y] = pos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.01, y);
  ctx.stroke();
});
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  e.preventDefault();
  const [x, y] = pos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});
const endStroke = () => {
  if (!drawing) return;
  drawing = false;
  strokes++;
  if ($("live").checked && session) recognise();
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("clear").addEventListener("click", () => { resetPad(); clearResult(); });

// ---------- upload ----------

$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploaded = await createImageBitmap(file);
  const prev = $("upload-preview");
  const scale = Math.min(1, 400 / Math.max(uploaded.width, uploaded.height));
  prev.width = Math.round(uploaded.width * scale);
  prev.height = Math.round(uploaded.height * scale);
  const pctx = prev.getContext("2d");
  pctx.fillStyle = "#fff";
  pctx.fillRect(0, 0, prev.width, prev.height);
  pctx.drawImage(uploaded, 0, 0, prev.width, prev.height);
  prev.hidden = false;
  if (session) recognise();
});

for (const tab of document.querySelectorAll("[data-mode]")) {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    for (const t of document.querySelectorAll("[data-mode]")) t.setAttribute("aria-selected", t === tab);
    $("draw-panel").hidden = mode !== "draw";
    $("upload-panel").hidden = mode !== "upload";
    clearResult();
  });
}

function grayOf(source, w, h) {
  // composite onto white paper, then greyscale as PIL does
  const c = new OffscreenCanvas(w, h);
  const cx = c.getContext("2d");
  cx.fillStyle = "#fff";
  cx.fillRect(0, 0, w, h);
  cx.drawImage(source, 0, 0, w, h);
  return grayFromRGBA(cx.getImageData(0, 0, w, h).data, w, h);
}

// ---------- recognition ----------

function softmax(logits, offset, n) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) m = Math.max(m, logits[offset + i]);
  const e = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) { e[i] = Math.exp(logits[offset + i] - m); s += e[i]; }
  return e.map((v) => v / s);
}

function clearResult() {
  $("top").innerHTML = "";
  $("glyph").textContent = "";
  $("label").textContent = "";
  $("seen").getContext("2d").clearRect(0, 0, 128, 128);
}

async function recognise() {
  let gray;
  if (mode === "draw") {
    if (!strokes) { status("Draw a character first."); return; }
    gray = grayOf(pad, pad.width, pad.height);
  } else {
    if (!uploaded) { status("Choose an image first."); return; }
    const scale = Math.min(1, 1024 / Math.max(uploaded.width, uploaded.height));
    gray = grayOf(uploaded, Math.round(uploaded.width * scale), Math.round(uploaded.height * scale));
  }
  const t0 = performance.now();
  const small = toDatasetScale(gray, mode === "upload", config.stroke_ratio);
  const { ink, meta } = preprocess(small, config.img);
  showSeen(ink);

  const size = config.img;
  const views = $("tta").checked ? config.views : ["id"];
  const base = Float32Array.from(ink, (v) => v / 255);
  const batch = new Float32Array(views.length * size * size);
  views.forEach((v, k) => {
    const x = view(base, size, v);
    for (let i = 0; i < x.length; i++) batch[k * size * size + i] = (x[i] - config.mean) / config.std;
  });
  const feeds = { image: new ort.Tensor("float32", batch, [views.length, 1, size, size]) };
  if (config.meta) {
    const f = [Math.log(Math.max(meta[0], 1)), Math.log(Math.max(meta[1], 1)),
               Math.log(Math.max(meta[2], 1)), Math.log(Math.max(meta[3], 1)), meta[4]];
    const z = new Float32Array(views.length * 5);
    for (let k = 0; k < views.length; k++)
      for (let j = 0; j < 5; j++) z[k * 5 + j] = (f[j] - config.meta_mean[j]) / config.meta_std[j];
    feeds.meta = new ort.Tensor("float32", z, [views.length, 5]);
  }
  const out = await session.run(feeds);
  const logits = out.logits.data;
  const n = config.classes.length;
  const probs = new Float64Array(n);
  for (let k = 0; k < views.length; k++) {
    const p = softmax(logits, k * n, n);
    for (let i = 0; i < n; i++) probs[i] += p[i] / views.length;
  }
  showResult(probs);
  status(`Done in ${Math.round(performance.now() - t0)} ms, on your device.`);
}

function showSeen(ink) {
  const c = $("seen").getContext("2d");
  const im = c.createImageData(128, 128);
  for (let i = 0; i < ink.length; i++) {
    const v = 255 - ink[i];
    im.data.set([v, v, v, 255], 4 * i);
  }
  c.putImageData(im, 0, 0);
}

function showResult(probs) {
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
  const best = config.classes[order[0]];
  $("glyph").textContent = best.char || "?";
  $("label").textContent = `class ${best.id} · ${best.name} · ${(100 * probs[order[0]]).toFixed(1)}%`
    + (best.char ? "" : " (not yet matched to a Unicode character)");
  $("top").innerHTML = "";
  for (const i of order) {
    const c = config.classes[i];
    const li = document.createElement("li");
    li.innerHTML = `<span class="g">${c.char || "?"}</span><span class="n">${c.id} · ${c.name}</span>`
      + `<span class="bar"><span style="width:${(100 * probs[i]).toFixed(1)}%"></span></span>`
      + `<span class="p">${(100 * probs[i]).toFixed(1)}%</span>`;
    $("top").appendChild(li);
  }
}

$("recognise").addEventListener("click", () => session && recognise());
$("pen").addEventListener("input", () => { $("pen-value").textContent = $("pen").value; });

resetPad();
load().catch((err) => { status(`Could not load the model: ${err.message}`); console.error(err); });

// exposed for automated checks
window.mayek = { preprocess, toDatasetScale, view, grayFromRGBA, ready: () => !!session,
                 run: async (feeds) => session.run(feeds), config: () => config };
