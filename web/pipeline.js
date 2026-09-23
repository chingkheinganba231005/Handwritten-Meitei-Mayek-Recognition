// Image preprocessing, ported from mayek/preprocess.py. Kept free of any
// browser API so it can be checked against the Python version.
//
// Images are {w, h, data} with data a typed array in row-major order.

const FLT_EPSILON = 1.1920928955078125e-7;
const MARGIN = 1.16;

export function grayFromRGBA(rgba, w, h) {
  // Same weights and rounding as PIL's convert("L").
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;
  }
  return { w, h, data: out };
}

function otsuThreshold(data) {
  // cv2.threshold(..., THRESH_OTSU) for uint8 data
  const hist = new Float64Array(256);
  for (const v of data) hist[v]++;
  const scale = 1 / data.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * hist[i];
  mu *= scale;
  let q1 = 0, mu1 = 0, maxSigma = 0, maxVal = 0;
  for (let i = 0; i < 256; i++) {
    const p = hist[i] * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
  }
  return maxVal;
}

export function inkMask(img) {
  // -> {g: Float64Array grey with dark ink on light paper, ink: Uint8Array}
  const t = otsuThreshold(img.data);
  const n = img.data.length;
  const ink = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) { if (img.data[i] <= t) { ink[i] = 1; count++; } }
  const g = new Float64Array(n);
  const flip = count / n > 0.5;
  for (let i = 0; i < n; i++) {
    g[i] = flip ? 255 - img.data[i] : img.data[i];
    if (flip) ink[i] = 1 - ink[i];
  }
  return { g, ink, w: img.w, h: img.h };
}

function sorted(values) {
  return Float64Array.from(values).sort();
}

function median(values) {
  const s = sorted(values);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentile(values, q) {
  const s = sorted(values);
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

export function resizeLinear(src, sw, sh, dw, dh) {
  // cv2.resize(..., INTER_LINEAR) on float data
  const fx = new Float64Array(dw), sx = new Int32Array(dw);
  const sxScale = sw / dw, syScale = sh / dh;
  const axis = (d, scale, size, pos, frac) => {
    let f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    f -= s;
    if (s < 0) { f = 0; s = 0; }
    if (s >= size - 1) { f = 0; s = size - 1; }
    pos[d] = s; frac[d] = f;
  };
  for (let x = 0; x < dw; x++) axis(x, sxScale, sw, sx, fx);
  const fy = new Float64Array(dh), sy = new Int32Array(dh);
  for (let y = 0; y < dh; y++) axis(y, syScale, sh, sy, fy);
  const rows = new Float64Array(sh * dw);  // horizontal pass
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      const a = src[y * sw + sx[x]];
      const b = sx[x] + 1 < sw ? src[y * sw + sx[x] + 1] : a;
      rows[y * dw + x] = a * (1 - fx[x]) + b * fx[x];
    }
  }
  const out = new Float64Array(dw * dh);  // vertical pass
  for (let y = 0; y < dh; y++) {
    const r0 = sy[y], r1 = Math.min(sy[y] + 1, sh - 1);
    for (let x = 0; x < dw; x++) {
      out[y * dw + x] = rows[r0 * dw + x] * (1 - fy[y]) + rows[r1 * dw + x] * fy[y];
    }
  }
  return out;
}

function areaWeights(sn, dn) {
  // for each output index, the (input index, weight) pairs of an exact box filter
  const scale = sn / dn, out = [];
  for (let d = 0; d < dn; d++) {
    const a = d * scale, b = (d + 1) * scale, taps = [];
    for (let s = Math.floor(a); s < Math.ceil(b) && s < sn; s++) {
      const w = Math.min(b, s + 1) - Math.max(a, s);
      if (w > 1e-9) taps.push([s, w / scale]);
    }
    out.push(taps);
  }
  return out;
}

export function resizeArea(src, sw, sh, dw, dh) {
  // cv2.resize(..., INTER_AREA) when shrinking
  const wx = areaWeights(sw, dw), wy = areaWeights(sh, dh);
  const rows = new Float64Array(sh * dw);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < dw; x++) {
      let v = 0;
      for (const [s, w] of wx[x]) v += src[y * sw + s] * w;
      rows[y * dw + x] = v;
    }
  const out = new Float64Array(dw * dh);
  for (let y = 0; y < dh; y++)
    for (let x = 0; x < dw; x++) {
      let v = 0;
      for (const [s, w] of wy[y]) v += rows[s * dw + x] * w;
      out[y * dw + x] = v;
    }
  return out;
}

export function preprocess(img, size = 128) {
  // mayek.preprocess.preprocess, ink channel only -> {ink: Uint8Array size*size, meta: [H, W, bh, bw, ink fraction]}
  const { w: W, h: H } = img;
  const { g: grey, ink } = inkMask(img);
  const paperVals = [], inkVals = [];
  for (let i = 0; i < ink.length; i++) (ink[i] ? inkVals : paperVals).push(grey[i]);
  const paper = paperVals.length ? median(paperVals) : 255;
  const dark = inkVals.length ? percentile(inkVals, 5) : 0;
  const denom = Math.max(paper - dark, 1);
  let y1 = H, y2 = 0, x1 = W, x2 = 0;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (ink[y * W + x]) { y1 = Math.min(y1, y); y2 = Math.max(y2, y + 1); x1 = Math.min(x1, x); x2 = Math.max(x2, x + 1); }
  if (!inkVals.length) { y1 = 0; y2 = H; x1 = 0; x2 = W; }
  const bh = y2 - y1, bw = x2 - x1;
  const side = Math.floor(Math.max(bh, bw) * MARGIN) + 2;
  const canvas = new Float64Array(side * side);
  const oy = (side - bh) >> 1, ox = (side - bw) >> 1;
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++) {
      const v = (paper - grey[(y1 + y) * W + x1 + x]) / denom;
      canvas[(oy + y) * side + ox + x] = Math.min(1, Math.max(0, v));
    }
  const big = side > size ? resizeArea(canvas, side, side, size, size) : resizeLinear(canvas, side, side, size, size);
  const out = new Uint8Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = Math.floor(Math.min(1, Math.max(0, big[i])) * 255 + 0.5);
  return { ink: out, meta: [H, W, bh, bw, inkVals.length / ink.length] };
}

// ---- demo-only rescaling (mayek.preprocess.to_dataset_scale) ----

function distanceTransform(mask, w, h) {
  // cv2.distanceTransform(DIST_L2, 3): 3x3 chamfer with a = 0.955, b = 1.3693
  const a = 0.955, b = 1.3693, INF = 1e9;
  const d = new Float64Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? INF : 0;
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? INF : d[y * w + x]);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!d[i]) continue;
      d[i] = Math.min(d[i], at(x - 1, y) + a, at(x, y - 1) + a, at(x - 1, y - 1) + b, at(x + 1, y - 1) + b);
    }
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (!d[i]) continue;
      d[i] = Math.min(d[i], at(x + 1, y) + a, at(x, y + 1) + a, at(x + 1, y + 1) + b, at(x - 1, y + 1) + b);
    }
  return d;
}

function thin(mask, w, h) {
  // Zhang-Suen thinning
  const m = Uint8Array.from(mask);
  const px = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : m[y * w + x]);
  let changed = true;
  while (changed) {
    changed = false;
    for (let step = 0; step < 2; step++) {
      const drop = [];
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          if (!m[y * w + x]) continue;
          const p = [px(x, y - 1), px(x + 1, y - 1), px(x + 1, y), px(x + 1, y + 1),
                     px(x, y + 1), px(x - 1, y + 1), px(x - 1, y), px(x - 1, y - 1)];
          const B = p.reduce((s, v) => s + v, 0);
          if (B < 2 || B > 6) continue;
          let A = 0;
          for (let k = 0; k < 8; k++) if (!p[k] && p[(k + 1) % 8]) A++;
          if (A !== 1) continue;
          if (step === 0 ? p[0] * p[2] * p[4] || p[2] * p[4] * p[6] : p[0] * p[2] * p[6] || p[0] * p[4] * p[6]) continue;
          drop.push(y * w + x);
        }
      for (const i of drop) m[i] = 0;
      if (drop.length) changed = true;
    }
  }
  return m;
}

export function strokeWidth(mask, w, h) {
  const d = distanceTransform(mask, w, h);
  const s = thin(mask, w, h);
  const vals = [];
  for (let i = 0; i < s.length; i++) if (s[i]) vals.push(2 * d[i]);
  return vals.length ? median(vals) : 0;
}

function ellipseKernel(k) {
  // cv2.getStructuringElement(MORPH_ELLIPSE, (k, k))
  const r = k >> 1, c = k >> 1, rows = [];
  for (let i = 0; i < k; i++) {
    const dy = i - r;
    const dx = Math.round(c * Math.sqrt((r * r - dy * dy) / (r * r || 1)));
    rows.push([Math.max(c - dx, 0), Math.min(c + dx + 1, k)]);
  }
  return rows;
}

function dilate(mask, w, h, k) {
  const rows = ellipseKernel(k), r = k >> 1, out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let i = 0; i < k; i++) {
        const yy = y + i - r;
        if (yy < 0 || yy >= h) continue;
        for (let j = rows[i][0]; j < rows[i][1]; j++) {
          const xx = x + j - r;
          if (xx >= 0 && xx < w) out[yy * w + xx] = 1;
        }
      }
    }
  return out;
}

export function toDatasetScale(img, crop = false, strokeRatio = 0.1, side = 24) {
  // any character image -> a small scan like the TUMMHCD ones
  if (Math.max(img.w, img.h) <= 2 * side) return img;
  let { g, ink, w, h } = inkMask(img);
  let any = false;
  for (const v of ink) if (v) { any = true; break; }
  if (!any) {
    const small = resizeArea(Float64Array.from(img.data), w, h, side, side);
    return { w: side, h: side, data: Uint8Array.from(small, (v) => Math.floor(v + 0.5)) };
  }
  if (crop) {
    let y1 = h, y2 = 0, x1 = w, x2 = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (ink[y * w + x]) { y1 = Math.min(y1, y); y2 = Math.max(y2, y); x1 = Math.min(x1, x); x2 = Math.max(x2, x); }
    const pad = Math.floor(0.2 * Math.max(y2 - y1, x2 - x1)) + 1;
    const Y1 = Math.max(y1 - pad, 0), Y2 = Math.min(y2 + pad + 1, h), X1 = Math.max(x1 - pad, 0), X2 = Math.min(x2 + pad + 1, w);
    const cw = X2 - X1, chh = Y2 - Y1, g2 = new Float64Array(cw * chh), ink2 = new Uint8Array(cw * chh);
    for (let y = 0; y < chh; y++)
      for (let x = 0; x < cw; x++) { g2[y * cw + x] = g[(Y1 + y) * w + X1 + x]; ink2[y * cw + x] = ink[(Y1 + y) * w + X1 + x]; }
    g = g2; ink = ink2; w = cw; h = chh;
  }
  const n = Math.max(w, h);
  const paperVals = [];
  for (let i = 0; i < ink.length; i++) if (!ink[i]) paperVals.push(g[i]);
  const paper = paperVals.length ? median(paperVals) : 255;
  let frame = new Float64Array(n * n).fill(paper);
  const mask = new Uint8Array(n * n);
  const oy = (n - h) >> 1, ox = (n - w) >> 1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) { frame[(oy + y) * n + ox + x] = g[y * w + x]; mask[(oy + y) * n + ox + x] = ink[y * w + x]; }

  const target = strokeRatio * n;
  const have = strokeWidth(mask, n, n);
  if (have && have < target) {
    const k = Math.round(target - have) | 1;
    const grown = dilate(mask, n, n, k);
    const inkVals = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) inkVals.push(frame[i]);
    const dark = percentile(inkVals, 5);
    for (let i = 0; i < frame.length; i++) if (grown[i] && !mask[i]) frame[i] = dark;
  }
  const small = resizeArea(frame, n, n, side, side);
  return { w: side, h: side, data: Uint8Array.from(small, (v) => Math.min(255, Math.max(0, Math.floor(v + 0.5)))) };
}

export function view(x, size, v) {
  // test-time view, as F.affine_grid + F.grid_sample(bilinear, zeros, align_corners=False)
  if (v === "id") return x;
  let s = 1, tx = 0, ty = 0;
  if (v.startsWith("s")) s = parseFloat(v.slice(1));
  else if (v.startsWith("dx")) tx = (-parseFloat(v.slice(2)) * 2) / size;
  else if (v.startsWith("dy")) ty = (-parseFloat(v.slice(2)) * 2) / size;
  const out = new Float32Array(size * size);
  const get = (xx, yy) => (xx < 0 || yy < 0 || xx >= size || yy >= size ? 0 : x[yy * size + xx]);
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      const gx = s * ((2 * j + 1) / size - 1) + tx, gy = s * ((2 * i + 1) / size - 1) + ty;
      const px = ((gx + 1) * size - 1) / 2, py = ((gy + 1) * size - 1) / 2;
      const x0 = Math.floor(px), y0 = Math.floor(py), fx = px - x0, fy = py - y0;
      out[i * size + j] = get(x0, y0) * (1 - fx) * (1 - fy) + get(x0 + 1, y0) * fx * (1 - fy)
        + get(x0, y0 + 1) * (1 - fx) * fy + get(x0 + 1, y0 + 1) * fx * fy;
    }
  return out;
}
