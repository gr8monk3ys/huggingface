// Dataset Explorer -- client-side port of the former core.py.
//
// Static (sdk: static), so it never sleeps and does not consume a CPU Basic
// slot. Everything it needs is public and CORS-enabled: the datasets-server
// REST API serves configs, splits and rows without a token.
//
// The Python version streamed rows through the `datasets` library and drew
// charts with matplotlib. This does the same work against the HTTP API and
// draws the charts as SVG, which removes the server rather than replacing it.

const API = "https://datasets-server.huggingface.co";

const SAMPLE_SIZES = [20, 50, 100];
const MAX_PANELS = 4;
const MAX_CATEGORICAL_CARDINALITY = 20;
const TOP_VALUES_SHOWN = 5;
const UNIQUE_VALUES_BEFORE_LISTING = 10;
const SAMPLE_ROWS_SHOWN = 10;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
const num = (n) => (typeof n === "number" ? n.toLocaleString() : n);

// --- the API ---------------------------------------------------------------

async function api(path, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`${API}/${path}?${qs}`);
  const body = await res.json().catch(() => ({}));

  // The datasets-server reports problems in the body, not only the status:
  // a renamed dataset, a script-based dataset, a gated one. Those messages are
  // genuinely useful, so surface them instead of a generic failure.
  const failed = (body.failed || [])[0];
  const message =
    body.error || failed?.error?.error || (res.ok ? null : `HTTP ${res.status}`);
  if (message) throw new Error(message);
  return body;
}

const loadSplits = (dataset) => api("splits", { dataset });
const loadRows = (dataset, config, split, length) =>
  api("rows", { dataset, config, split, offset: 0, length });

// --- describing the sample -------------------------------------------------

function featureType(feature) {
  const t = feature.type || {};
  return t.dtype || t._type || "unknown";
}

function describe(features, rows) {
  const values = (name) => rows.map((r) => r.row[name]);

  return features.map((f) => {
    const name = f.name;
    const dtype = featureType(f);
    const raw = values(name);
    const present = raw.filter((v) => v !== null && v !== undefined && v !== "");
    const nonNullPct = rows.length ? (present.length / rows.length) * 100 : 0;

    const col = { name, dtype, nonNull: present.length, nonNullPct, rows: rows.length };

    const numeric = present.every((v) => typeof v === "number");
    if (numeric && present.length) {
      const nums = present.slice().sort((a, b) => a - b);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const variance =
        nums.reduce((acc, v) => acc + (v - mean) ** 2, 0) / Math.max(nums.length - 1, 1);
      Object.assign(col, {
        kind: "numeric",
        min: nums[0],
        max: nums[nums.length - 1],
        mean,
        std: Math.sqrt(variance),
        values: present,
      });
      return col;
    }

    const counts = new Map();
    present.forEach((v) => {
      const key = typeof v === "object" ? JSON.stringify(v) : String(v);
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    Object.assign(col, {
      kind: "categorical",
      unique: counts.size,
      top: counts.size <= UNIQUE_VALUES_BEFORE_LISTING ? sorted.slice(0, TOP_VALUES_SHOWN) : null,
      counts: sorted,
    });
    return col;
  });
}

// --- charts, drawn rather than imported ------------------------------------

function svgFrame(inner, w, h) {
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}"
            preserveAspectRatio="xMidYMid meet" role="img">${inner}</svg>`;
}

function histogram(col) {
  const W = 320, H = 150, PAD_L = 34, PAD_B = 26, PAD_T = 10, PAD_R = 8;
  const bins = 14;
  const { min, max, values } = col;
  const span = max - min || 1;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => {
    const i = Math.min(bins - 1, Math.floor(((v - min) / span) * bins));
    counts[i] += 1;
  });
  const peak = Math.max(...counts, 1);
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_B - PAD_T;
  const bw = plotW / bins;

  const bars = counts
    .map((c, i) => {
      const h = (c / peak) * plotH;
      return `<rect x="${(PAD_L + i * bw + 1).toFixed(1)}" y="${(PAD_T + plotH - h).toFixed(1)}"
        width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="var(--accent)" rx="1"/>`;
    })
    .join("");

  const axis = `
    <line x1="${PAD_L}" y1="${PAD_T + plotH}" x2="${W - PAD_R}" y2="${PAD_T + plotH}" stroke="var(--rule)"/>
    <text x="${PAD_L}" y="${H - 8}" fill="var(--faint)" font-size="9">${fmtNum(min)}</text>
    <text x="${W - PAD_R}" y="${H - 8}" fill="var(--faint)" font-size="9" text-anchor="end">${fmtNum(max)}</text>
    <text x="${PAD_L - 5}" y="${PAD_T + 8}" fill="var(--faint)" font-size="9" text-anchor="end">${peak}</text>
    <text x="${PAD_L - 5}" y="${PAD_T + plotH}" fill="var(--faint)" font-size="9" text-anchor="end">0</text>`;

  return svgFrame(bars + axis, W, H);
}

function fmtNum(v) {
  if (!isFinite(v)) return "-";
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toFixed(2);
}

function barChart(col) {
  const items = col.counts.slice(0, 8);
  const W = 320, ROW = 20, PAD_T = 8, LABEL_W = 96;
  const H = PAD_T * 2 + items.length * ROW;
  const peak = Math.max(...items.map((i) => i[1]), 1);
  const plotW = W - LABEL_W - 34;

  const rows = items
    .map(([label, count], i) => {
      const y = PAD_T + i * ROW;
      const w = (count / peak) * plotW;
      const short = label.length > 16 ? label.slice(0, 15) + "…" : label;
      return `
        <text x="${LABEL_W - 6}" y="${y + 13}" fill="var(--faint)" font-size="10"
              text-anchor="end">${esc(short)}</text>
        <rect x="${LABEL_W}" y="${y + 4}" width="${w.toFixed(1)}" height="12"
              fill="var(--accent)" rx="1"/>
        <text x="${LABEL_W + w + 5}" y="${y + 14}" fill="var(--faint)" font-size="9">${count}</text>`;
    })
    .join("");

  return svgFrame(rows, W, H);
}

function chartFor(col) {
  if (col.kind === "numeric" && col.values.length > 1) return histogram(col);
  if (col.kind === "categorical" && col.unique <= MAX_CATEGORICAL_CARDINALITY && col.unique > 1)
    return barChart(col);
  return null;
}

// --- rendering -------------------------------------------------------------

function renderStats(cols, meta) {
  const summary = `
    <div class="tiles">
      <div class="tile"><div class="tile-n">${num(meta.total)}</div><div class="tile-l">rows in split</div></div>
      <div class="tile"><div class="tile-n">${num(meta.sampled)}</div><div class="tile-l">rows sampled</div></div>
      <div class="tile"><div class="tile-n">${cols.length}</div><div class="tile-l">columns</div></div>
    </div>`;

  const list = cols
    .map((c) => {
      const detail =
        c.kind === "numeric"
          ? `<span>range ${fmtNum(c.min)} &ndash; ${fmtNum(c.max)}</span>
             <span>mean ${fmtNum(c.mean)}</span><span>std ${fmtNum(c.std)}</span>`
          : `<span>${num(c.unique)} unique</span>` +
            (c.top ? `<span>top: ${esc(c.top.map(([v, n]) => `${v} (${n})`).join(", "))}</span>` : "");
      return `
        <div class="col">
          <div class="col-head">
            <span class="col-name">${esc(c.name)}</span>
            <span class="col-type">${esc(c.dtype)}</span>
          </div>
          <div class="col-meta">
            <span>${c.nonNullPct.toFixed(1)}% non-null</span>${detail}
          </div>
        </div>`;
    })
    .join("");

  $("stats").innerHTML = summary + `<div class="cols">${list}</div>`;
}

function renderCharts(cols) {
  const drawable = cols.map((c) => [c, chartFor(c)]).filter(([, s]) => s).slice(0, MAX_PANELS);
  if (!drawable.length) {
    $("charts-wrap").hidden = true;
    return;
  }
  $("charts").innerHTML = drawable
    .map(
      ([c, svg]) => `
      <figure class="chart">
        <figcaption>${esc(c.name)}<span>${c.kind === "numeric" ? "distribution" : "value counts"}</span></figcaption>
        ${svg}
      </figure>`
    )
    .join("");
  $("charts-wrap").hidden = false;
}

function renderTable(features, rows) {
  const names = features.map((f) => f.name);
  const head = names.map((n) => `<th>${esc(n)}</th>`).join("");
  const body = rows
    .slice(0, SAMPLE_ROWS_SHOWN)
    .map((r) => {
      const cells = names
        .map((n) => {
          let v = r.row[n];
          if (v === null || v === undefined) v = "";
          if (typeof v === "object") v = JSON.stringify(v);
          v = String(v);
          const short = v.length > 140 ? v.slice(0, 139) + "…" : v;
          return `<td title="${esc(v.slice(0, 400))}">${esc(short)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  $("sample").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  $("sample-wrap").hidden = false;
}

function status(message, kind = "info") {
  $("status").innerHTML = message ? `<p class="${kind}">${esc(message)}</p>` : "";
}

function clearResults() {
  $("stats").innerHTML = "";
  $("charts-wrap").hidden = true;
  $("sample-wrap").hidden = true;
}

// --- flow ------------------------------------------------------------------

let splitIndex = [];

async function discover() {
  const dataset = $("dataset").value.trim();
  if (!dataset) return;
  clearResults();
  status("Looking up configs and splits…");
  $("go").disabled = true;
  try {
    const { splits } = await loadSplits(dataset);
    splitIndex = splits;
    const configs = [...new Set(splits.map((s) => s.config))];
    fill($("config"), configs);
    onConfigChange();
    status(`${configs.length} config${configs.length === 1 ? "" : "s"} available.`);
    $("go").disabled = false;
  } catch (err) {
    splitIndex = [];
    fill($("config"), []);
    fill($("split"), []);
    status(err.message, "error");
  }
}

function onConfigChange() {
  const config = $("config").value;
  fill($("split"), splitIndex.filter((s) => s.config === config).map((s) => s.split));
}

function fill(select, options) {
  select.innerHTML = "";
  options.forEach((o) => select.add(new Option(o, o)));
  select.disabled = !options.length;
}

async function explore(event) {
  event?.preventDefault();
  const dataset = $("dataset").value.trim();
  const config = $("config").value;
  const split = $("split").value;
  const length = Number($("size").value);
  if (!dataset || !config || !split) {
    status("Pick a dataset, config and split first.", "error");
    return;
  }

  $("go").disabled = true;
  clearResults();
  status(`Fetching ${length} rows…`);
  try {
    const data = await loadRows(dataset, config, split, length);
    const rows = data.rows || [];
    if (!rows.length) {
      status("That split returned no rows.", "error");
      return;
    }
    const cols = describe(data.features, rows);
    renderStats(cols, { total: data.num_rows_total, sampled: rows.length });
    renderCharts(cols);
    renderTable(data.features, rows);
    status("");
  } catch (err) {
    status(err.message, "error");
  } finally {
    $("go").disabled = false;
  }
}

function buildPresets() {
  $("presets").innerHTML = PRESETS.map(
    (p) => `<button type="button" class="preset" data-id="${esc(p.id)}">${esc(p.label)}</button>`
  ).join("");
  $("presets").addEventListener("click", (e) => {
    const btn = e.target.closest(".preset");
    if (!btn) return;
    $("dataset").value = btn.dataset.id;
    discover();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  SAMPLE_SIZES.forEach((n) => $("size").add(new Option(`${n} rows`, n)));
  $("size").value = 50;
  buildPresets();
  $("dataset").addEventListener("change", discover);
  $("config").addEventListener("change", onConfigChange);
  $("form").addEventListener("submit", explore);

  // Open on a real dataset rather than an empty shell.
  $("dataset").value = PRESETS[0].id;
  discover().then(() => explore());
});
