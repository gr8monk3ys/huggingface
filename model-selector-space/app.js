// Model Selector -- client-side port of the former core.py.
//
// This Space is static (sdk: static) so it is always awake and does not consume
// one of the three concurrent Space slots the free tier allows. It can be
// static because everything it does is a public, CORS-enabled Hub API call plus
// local ranking -- no token, no server.
//
// The live query and the curated fallback behave exactly as the Python did:
// try the Hub, fall back to the curated table, and always say which happened.

const HUB_API = "https://huggingface.co/api/models";
const LIVE_QUERY_LIMIT = 8;
const LIVE_RESULTS_SHOWN = 5;
const CURATED_RESULTS_SHOWN = 4;
const ANY_SIZE = "Any size";
const BEST_QUALITY = "Best Quality";

// --- ranking -------------------------------------------------------------

// Parse "7B", "67M", "1.5B" to millions. 0 for unparseable input, never a guess.
function parseSize(sizeStr) {
  if (!sizeStr) return 0;
  const s = String(sizeStr).trim().toUpperCase().replace(/,/g, "");
  const match = s.match(/([0-9]*\.?[0-9]+)\s*([BMK]?)/);
  if (!match) return 0;
  const val = parseFloat(match[1]);
  const unit = match[2];
  if (unit === "B") return val * 1000;
  if (unit === "K") return val / 1000;
  return val; // 'M' or unspecified -> already millions
}

function rankCurated(models, sizePref, priority) {
  const range = DATA.sizePreferences[sizePref] || DATA.sizePreferences[ANY_SIZE];
  let out = models.slice();
  if (sizePref !== ANY_SIZE) {
    out = out.filter((m) => {
      const size = parseSize(m.size);
      return size >= range.min && size <= range.max;
    });
  }
  if (priority === "Smallest/Fastest") {
    out.sort((a, b) => parseSize(a.size) - parseSize(b.size));
  } else if (priority === BEST_QUALITY) {
    out.sort((a, b) => parseSize(b.size) - parseSize(a.size));
  }
  // "Most Popular" keeps the curated order.
  return out;
}

// --- the live Hub query --------------------------------------------------

async function fetchLiveModels(taskId, limit = LIVE_QUERY_LIMIT) {
  const url =
    `${HUB_API}?filter=${encodeURIComponent(taskId)}` +
    `&sort=downloads&direction=-1&limit=${limit}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const rows = await res.json();
    const out = rows
      .map((m) => ({
        name: m.id || m.modelId,
        downloads: Number(m.downloads) || 0,
        likes: Number(m.likes) || 0,
      }))
      .filter((m) => m.name);
    return out.length ? out : null;
  } catch {
    // Offline, rate-limited, or blocked: the curated list is the fallback, and
    // the caller tells the user that is what they are looking at.
    return null;
  }
}

// --- the entry point, mirroring core.recommend ---------------------------

async function recommend(task, sizePref, priority) {
  const info = DATA.tasks[task];
  if (!info) throw new Error("Please select a task.");

  const live = await fetchLiveModels(info.id);
  if (live) {
    let ordered = live;
    if (priority === BEST_QUALITY) {
      ordered = live.slice().sort((a, b) => b.likes - a.likes);
    }
    const models = ordered.slice(0, LIVE_RESULTS_SHOWN);
    return {
      task,
      description: info.description,
      source: "live",
      models,
      codeExample: codeExample(task, models[0].name),
      // Size is a curated-list concept; live results rank by popularity.
      sizeFilterIgnored: sizePref !== ANY_SIZE,
    };
  }

  const curated = rankCurated(info.top_models, sizePref, priority);
  if (!curated.length) {
    throw new Error("No models match your size preference. Try 'Any size'.");
  }
  const models = curated.slice(0, CURATED_RESULTS_SHOWN);
  return {
    task,
    description: info.description,
    source: "curated",
    models,
    codeExample: codeExample(task, models[0].name),
    sizeFilterIgnored: false,
  };
}

function codeExample(task, modelName) {
  if (!modelName) return "";
  const tpl = DATA.codeTemplates[task] || DATA.genericTemplate;
  return tpl.split("__MODEL__").join(modelName);
}

// --- rendering -----------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

function renderResult(result) {
  const parts = [
    `<h2>Recommendations for: ${esc(result.task)}</h2>`,
    `<p class="muted">${esc(result.description)}</p>`,
  ];

  if (result.source === "live") {
    parts.push(`<p class="source">Live from the HuggingFace Hub, sorted by downloads.</p>`);
    if (result.sizeFilterIgnored) {
      parts.push(
        `<p class="note">Size filtering applies to the curated fallback; live results are ranked by popularity.</p>`
      );
    }
  } else {
    parts.push(`<p class="source">Curated picks (live Hub query unavailable right now).</p>`);
  }

  result.models.forEach((m, i) => {
    const detail =
      result.source === "live"
        ? `<span><strong>Downloads:</strong> ${m.downloads.toLocaleString()}</span>
           <span><strong>Likes:</strong> ${m.likes.toLocaleString()}</span>`
        : `<span><strong>Size:</strong> ${esc(m.size)} parameters</span>
           <span><strong>License:</strong> ${esc(m.license)}</span>`;
    parts.push(`
      <article class="model">
        <h3>${i + 1}. ${esc(m.name)}</h3>
        <div class="meta">${detail}</div>
        <a href="https://huggingface.co/${esc(m.name)}" target="_blank" rel="noopener">
          View on HuggingFace &rarr;
        </a>
      </article>`);
  });

  $("results").innerHTML = parts.join("\n");
  $("code").textContent = result.codeExample.replace(/^```python\n|\n```$/g, "");
  $("code-wrap").hidden = false;
}

function renderError(message) {
  $("results").innerHTML = `<p class="error">${esc(message)}</p>`;
  $("code-wrap").hidden = true;
}

// --- wiring --------------------------------------------------------------

function populate() {
  const taskSel = $("task");
  Object.keys(DATA.tasks).forEach((t) => taskSel.add(new Option(t, t)));
  const sizeSel = $("size");
  Object.keys(DATA.sizePreferences).forEach((s) => sizeSel.add(new Option(s, s)));
  sizeSel.value = ANY_SIZE;
  ["Most Popular", "Best Quality", "Smallest/Fastest"].forEach((p) =>
    $("priority").add(new Option(p, p))
  );
  showTaskInfo();
}

function showTaskInfo() {
  const info = DATA.tasks[$("task").value];
  $("task-info").innerHTML = info
    ? `<strong>${esc(info.description)}</strong><br>Common uses: ${info.use_cases
        .map(esc)
        .join(", ")}`
    : "";
}

async function onSubmit(event) {
  event.preventDefault();
  const btn = $("go");
  btn.disabled = true;
  $("results").innerHTML = `<p class="muted">Querying the Hub&hellip;</p>`;
  try {
    renderResult(await recommend($("task").value, $("size").value, $("priority").value));
  } catch (err) {
    renderError(err.message || String(err));
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  populate();
  $("task").addEventListener("change", showTaskInfo);
  $("form").addEventListener("submit", onSubmit);
});
