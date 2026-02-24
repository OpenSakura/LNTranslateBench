/* global window, document, fetch */

const state = {
  // Global Data
  index: null,
  summary: null,
  chapterCache: new Map(), // key: `${judge}|${sample}` => matches[]

  // UI State
  page: "overview", // 'overview' | 'details'
  judge: "combined",
  heatmapMetric: "expected",
  mobileDetailsCol: "chapters", // 'chapters' | 'matches' | 'result'

  // Selection State
  selectedChapter: null,
  selectedMatch: null,
};

const el = {
  // Global
  banner: document.getElementById("banner"),
  judgeSelect: document.getElementById("judgeSelect"),
  reloadBtn: document.getElementById("reloadBtn"),

  // Tabs/Switcher
  pageSwitcher: document.getElementById("pageSwitcher"),
  pageOverview: document.getElementById("pageOverview"),
  pageDetails: document.getElementById("pageDetails"),

  // Mobile navigation
  mobileNav: document.getElementById("mobileNav"),
  detailsSubNav: document.getElementById("detailsSubNav"),

  // Overview Page
  cardJudge: document.getElementById("cardJudge"),
  cardModels: document.getElementById("cardModels"),
  cardMatches: document.getElementById("cardMatches"),
  cardLoadedAt: document.getElementById("cardLoadedAt"),
  leaderboardBody: document.getElementById("leaderboardBody"),
  barPlot: document.getElementById("barPlot"),
  heatmap: document.getElementById("heatmap"),
  heatmapMetric: document.getElementById("heatmapMetric"),
  tooltip: document.getElementById("tooltip"),

  // Details Page
  chapterSearch: document.getElementById("chapterSearch"),
  chapterList: document.getElementById("chapterList"),
  matchCount: document.getElementById("matchCount"),
  matchList: document.getElementById("matchList"),
  matchDetail: document.getElementById("matchDetail"),
};

/* --- Helpers --- */

function setBanner(message, kind = "error") {
  if (!message) {
    el.banner.classList.add("hidden");
    el.banner.textContent = "";
    return;
  }
  el.banner.classList.remove("hidden");
  el.banner.textContent = message;
  el.banner.dataset.kind = kind;
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    // noop
  }
  if (!res.ok) {
    const msg = data && data.error ? data.error : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorForValue(v) {
  const x = clamp01(v);
  // Sakura diverging: warm rose → neutral → deep violet
  const rose = [0xbe, 0x12, 0x3c]; // #be123c
  const neutral = [0xf0, 0xf0, 0xf0]; // #f0f0f0
  const violet = [0x4c, 0x1d, 0x95]; // #4c1d95

  let rgb;
  if (x < 0.5) {
    const t = x / 0.5;
    rgb = [lerp(rose[0], neutral[0], t), lerp(rose[1], neutral[1], t), lerp(rose[2], neutral[2], t)];
  } else {
    const t = (x - 0.5) / 0.5;
    rgb = [lerp(neutral[0], violet[0], t), lerp(neutral[1], violet[1], t), lerp(neutral[2], violet[2], t)];
  }

  const [r, g, b] = rgb.map((n) => Math.round(n));
  return `rgb(${r}, ${g}, ${b})`;
}

function textColorForValue(v) {
  const x = clamp01(v);
  // Use luminance to decide text color
  const rose = [0xbe, 0x12, 0x3c];
  const neutral = [0xf0, 0xf0, 0xf0];
  const violet = [0x4c, 0x1d, 0x95];

  let rgb;
  if (x < 0.5) {
    const t = x / 0.5;
    rgb = [lerp(rose[0], neutral[0], t), lerp(rose[1], neutral[1], t), lerp(rose[2], neutral[2], t)];
  } else {
    const t = (x - 0.5) / 0.5;
    rgb = [lerp(neutral[0], violet[0], t), lerp(neutral[1], violet[1], t), lerp(neutral[2], violet[2], t)];
  }

  const [r, g, b] = rgb.map((n) => Math.round(n));
  // Relative luminance
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.55 ? 'rgba(0, 0, 0, 0.75)' : 'rgba(255, 255, 255, 0.9)';
}

function setPage(pageName) {
  state.page = pageName;
  if (pageName === "overview") {
    el.pageSwitcher.classList.remove("left");
    el.pageSwitcher.innerHTML = `<span class="arrow">»</span> <span class="label">Detailed View</span>`;
    el.pageOverview.classList.remove("hidden");
    el.pageDetails.classList.add("hidden");
    el.detailsSubNav.classList.add("hidden");
  } else {
    el.pageSwitcher.classList.add("left");
    el.pageSwitcher.innerHTML = `<span class="arrow">«</span> <span class="label">Overview</span>`;
    el.pageOverview.classList.add("hidden");
    el.pageDetails.classList.remove("hidden");
    // Show sub-nav on mobile when in details view
    if (isMobile()) {
      el.detailsSubNav.classList.remove("hidden");
      setMobileDetailsCol(state.mobileDetailsCol || "chapters");
    }
  }
  // Update mobile nav active state
  updateMobileNavActive(pageName);
}

function isMobile() {
  return window.innerWidth <= 768;
}

function syncTopbarHeight() {
  const h = document.querySelector(".topbar")?.offsetHeight || 0;
  document.documentElement.style.setProperty("--topbar-h", h + "px");
}

function updateMobileNavActive(pageName) {
  if (!el.mobileNav) return;
  el.mobileNav.querySelectorAll(".mobile-nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.page === pageName);
  });
}

function setMobileDetailsCol(col) {
  state.mobileDetailsCol = col;
  const grid = document.querySelector(".details-grid");
  if (!grid) return;

  // Remove all mobile-col-* classes
  grid.classList.remove("mobile-col-chapters", "mobile-col-matches", "mobile-col-result");

  if (isMobile()) {
    grid.classList.add(`mobile-col-${col}`);
  }

  // Update sub-nav active state
  if (el.detailsSubNav) {
    el.detailsSubNav.querySelectorAll(".subnav-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.col === col);
    });
  }
}

/* --- Overview Rendering --- */

function populateJudges(index) {
  const judges = index.judges || [];
  el.judgeSelect.innerHTML = "";

  const combined = document.createElement("option");
  combined.value = "combined";
  combined.textContent = "Combined (all judges)";
  el.judgeSelect.appendChild(combined);

  for (const j of judges) {
    const opt = document.createElement("option");
    opt.value = j.model;
    const m = typeof j.matches === "number" ? j.matches : 0;
    opt.textContent = `${j.model} (${m} matches)`;
    el.judgeSelect.appendChild(opt);
  }
  el.judgeSelect.value = state.judge;
}

function renderCards(index, summary) {
  el.cardJudge.textContent = summary.judge || state.judge;
  el.cardModels.textContent = String(summary.num_models ?? "—");
  el.cardMatches.textContent = String(summary.num_matches ?? "—");
  el.cardLoadedAt.textContent = index.loaded_at || "—";
}

function renderLeaderboard(summary) {
  const rows = summary.ratings || [];
  el.leaderboardBody.innerHTML = "";

  const maxElo = rows.length ? Math.max(...rows.map((r) => r.elo)) : 1;
  const minElo = rows.length ? Math.min(...rows.map((r) => r.elo)) : 0;
  const eloRange = maxElo - minElo || 1;

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");

    const tdRank = document.createElement("td");
    tdRank.className = "col-rank";
    if (idx < 3) {
      const medal = document.createElement("span");
      medal.className = `rank-medal ${["gold", "silver", "bronze"][idx]}`;
      medal.textContent = String(idx + 1);
      tdRank.appendChild(medal);
    } else {
      const normal = document.createElement("span");
      normal.className = "rank-normal";
      normal.textContent = String(idx + 1);
      tdRank.appendChild(normal);
    }
    tr.appendChild(tdRank);

    const tdModel = document.createElement("td");
    const modelSpan = document.createElement("span");
    modelSpan.className = "model-name";
    modelSpan.textContent = r.model || "—";
    tdModel.appendChild(modelSpan);
    tr.appendChild(tdModel);

    const tdElo = document.createElement("td");
    tdElo.className = "col-num";
    const eloSpan = document.createElement("span");
    eloSpan.className = "elo-value";
    if (idx === 0) eloSpan.classList.add("elo-gradient");
    eloSpan.textContent = fmt(r.elo, 2);
    tdElo.appendChild(eloSpan);

    // Inline Elo bar
    const eloBar = document.createElement("span");
    eloBar.className = "elo-bar-inline";
    const pct = ((r.elo - minElo) / eloRange) * 100;
    eloBar.style.width = `${Math.max(4, pct * 0.5)}px`;
    eloBar.style.background = idx < 3
      ? `linear-gradient(90deg, ${["#fbbf24", "#94a3b8", "#d97706"][idx]}, transparent)`
      : `linear-gradient(90deg, var(--sakura-400), transparent)`;
    eloBar.style.opacity = idx < 3 ? "0.7" : "0.3";
    tdElo.appendChild(eloBar);
    tr.appendChild(tdElo);

    const tdGames = document.createElement("td");
    tdGames.className = "col-num";
    const gamesSpan = document.createElement("span");
    gamesSpan.className = "games-count";
    gamesSpan.textContent = String(r.games ?? 0);
    tdGames.appendChild(gamesSpan);
    tr.appendChild(tdGames);

    tr.style.animationDelay = `${idx * 40}ms`;
    el.leaderboardBody.appendChild(tr);
  });
}

function renderBarPlot(summary) {
  const ratings = summary.ratings || [];
  el.barPlot.innerHTML = "";

  if (!ratings.length) {
    const div = document.createElement("div");
    div.className = "muted";
    div.textContent = "No data to plot.";
    el.barPlot.appendChild(div);
    return;
  }

  const n = ratings.length;
  const barW = 32;
  const gap = 6;
  const padding = { l: 50, r: 16, t: 30, b: 110 };
  const width = padding.l + padding.r + n * (barW + gap) - gap;
  const height = 340;
  const innerH = height - padding.t - padding.b;

  const maxElo = Math.max(...ratings.map((r) => r.elo));
  const minElo = Math.min(...ratings.map((r) => r.elo));
  const range = maxElo - minElo || 1;

  // Add some padding to the range for grid lines
  const niceMin = Math.floor(minElo / 50) * 50;
  const niceMax = Math.ceil(maxElo / 50) * 50;
  const niceRange = niceMax - niceMin || 1;

  function barColor(rank, total) {
    const t = total > 1 ? rank / (total - 1) : 0;
    // Sakura gradient: vibrant pink → soft lavender → muted rose
    if (t < 0.5) {
      const s = t * 2;
      const r = Math.round(236 + (167 - 236) * s);
      const g = Math.round(72 + (139 - 72) * s);
      const b = Math.round(153 + (250 - 153) * s);
      return `rgb(${r},${g},${b})`;
    } else {
      const s = (t - 0.5) * 2;
      const r = Math.round(167 + (120 - 167) * s);
      const g = Math.round(139 + (113 - 139) * s);
      const b = Math.round(250 + (200 - 250) * s);
      return `rgb(${r},${g},${b})`;
    }
  }

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));
  svg.style.minWidth = `${Math.max(width, 400)}px`;

  // Add defs for gradients and filters
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");

  // Bar glow filter
  const filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
  filter.setAttribute("id", "barGlow");
  filter.setAttribute("x", "-20%");
  filter.setAttribute("y", "-20%");
  filter.setAttribute("width", "140%");
  filter.setAttribute("height", "140%");
  const feGaussian = document.createElementNS("http://www.w3.org/2000/svg", "feGaussianBlur");
  feGaussian.setAttribute("stdDeviation", "3");
  feGaussian.setAttribute("result", "blur");
  filter.appendChild(feGaussian);
  const feMerge = document.createElementNS("http://www.w3.org/2000/svg", "feMerge");
  const feMergeNode1 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
  feMergeNode1.setAttribute("in", "blur");
  feMerge.appendChild(feMergeNode1);
  const feMergeNode2 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
  feMergeNode2.setAttribute("in", "SourceGraphic");
  feMerge.appendChild(feMergeNode2);
  filter.appendChild(feMerge);
  defs.appendChild(filter);
  svg.appendChild(defs);

  // Draw horizontal grid lines
  const gridSteps = 5;
  for (let g = 0; g <= gridSteps; g++) {
    const val = niceMin + (niceRange * g) / gridSteps;
    const gy = padding.t + innerH - ((val - niceMin) / niceRange) * innerH;

    const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gridLine.setAttribute("x1", String(padding.l - 6));
    gridLine.setAttribute("x2", String(width - padding.r));
    gridLine.setAttribute("y1", String(gy));
    gridLine.setAttribute("y2", String(gy));
    gridLine.setAttribute("class", "chart-grid-line");
    gridLine.setAttribute("stroke", "rgba(255,255,255,0.06)");
    gridLine.setAttribute("stroke-dasharray", "4 4");
    svg.appendChild(gridLine);

    // Grid label
    const gridLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    gridLabel.setAttribute("x", String(padding.l - 10));
    gridLabel.setAttribute("y", String(gy + 4));
    gridLabel.setAttribute("fill", "currentColor");
    gridLabel.setAttribute("font-size", "10");
    gridLabel.setAttribute("text-anchor", "end");
    gridLabel.setAttribute("opacity", "0.4");
    gridLabel.textContent = Math.round(val);
    svg.appendChild(gridLabel);
  }

  for (let i = 0; i < n; i++) {
    const r = ratings[i];
    const x = padding.l + i * (barW + gap);
    const barH = ((r.elo - niceMin) / niceRange) * innerH;
    const y = padding.t + innerH - barH;

    // Create gradient for each bar
    const gradId = `barGrad${i}`;
    const grad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
    grad.setAttribute("id", gradId);
    grad.setAttribute("x1", "0");
    grad.setAttribute("y1", "0");
    grad.setAttribute("x2", "0");
    grad.setAttribute("y2", "1");
    const stop1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    stop1.setAttribute("offset", "0%");
    stop1.setAttribute("stop-color", barColor(i, n));
    stop1.setAttribute("stop-opacity", "1");
    grad.appendChild(stop1);
    const stop2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    stop2.setAttribute("offset", "100%");
    stop2.setAttribute("stop-color", barColor(i, n));
    stop2.setAttribute("stop-opacity", "0.5");
    grad.appendChild(stop2);
    defs.appendChild(grad);

    // Bar shadow for depth
    const shadow = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    shadow.setAttribute("x", String(x + 2));
    shadow.setAttribute("y", String(y + 3));
    shadow.setAttribute("width", String(barW));
    shadow.setAttribute("height", String(barH));
    shadow.setAttribute("rx", "6");
    shadow.setAttribute("fill", "rgba(0,0,0,0.12)");
    svg.appendChild(shadow);

    // Main bar with gradient
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(barW));
    rect.setAttribute("height", String(barH));
    rect.setAttribute("rx", "6");
    rect.setAttribute("fill", `url(#${gradId})`);
    if (i === 0) rect.setAttribute("filter", "url(#barGlow)");

    // Animate bar entrance
    const stagger = Math.min(0.02, 0.4 / n);
    const anim = document.createElementNS("http://www.w3.org/2000/svg", "animate");
    anim.setAttribute("attributeName", "height");
    anim.setAttribute("from", "0");
    anim.setAttribute("to", String(barH));
    anim.setAttribute("dur", "0.6s");
    anim.setAttribute("fill", "freeze");
    anim.setAttribute("begin", `${i * stagger}s`);
    anim.setAttribute("calcMode", "spline");
    anim.setAttribute("keySplines", "0.16 1 0.3 1");
    rect.appendChild(anim);

    const animY = document.createElementNS("http://www.w3.org/2000/svg", "animate");
    animY.setAttribute("attributeName", "y");
    animY.setAttribute("from", String(padding.t + innerH));
    animY.setAttribute("to", String(y));
    animY.setAttribute("dur", "0.6s");
    animY.setAttribute("fill", "freeze");
    animY.setAttribute("begin", `${i * stagger}s`);
    animY.setAttribute("calcMode", "spline");
    animY.setAttribute("keySplines", "0.16 1 0.3 1");
    rect.appendChild(animY);

    svg.appendChild(rect);

    // Score label above bar
    const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
    score.setAttribute("x", String(x + barW / 2));
    score.setAttribute("y", String(y - 8));
    score.setAttribute("fill", "currentColor");
    score.setAttribute("font-size", "11");
    score.setAttribute("font-weight", "700");
    score.setAttribute("text-anchor", "middle");
    score.setAttribute("opacity", i === 0 ? "1" : "0.8");
    score.textContent = fmt(r.elo, 0);
    svg.appendChild(score);

    // Model label (rotated)
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(x + barW / 2));
    label.setAttribute("y", String(padding.t + innerH + 12));
    label.setAttribute("fill", "currentColor");
    label.setAttribute("font-size", "11");
    label.setAttribute("font-weight", "500");
    label.setAttribute("text-anchor", "end");
    label.setAttribute("transform", `rotate(-45, ${x + barW / 2}, ${padding.t + innerH + 12})`);
    label.textContent = r.model;
    svg.appendChild(label);
  }

  el.barPlot.appendChild(svg);
}

function cellTooltipContent(pairwise, i, j, metric) {
  const a = pairwise.models[i];
  const b = pairwise.models[j];
  const expected = pairwise.expected?.[i]?.[j];
  const observed = pairwise.observed?.[i]?.[j];
  const games = pairwise.games?.[i]?.[j];
  const wins = pairwise.wins?.[i]?.[j];
  const ties = pairwise.ties?.[i]?.[j];
  const lines = [];
  lines.push(`${a}  vs  ${b}`);
  if (metric === "expected") {
    lines.push(`Elo expected: ${expected === null || expected === undefined ? "—" : fmt(expected, 2)}`);
  } else {
    lines.push(`Observed: ${observed === null || observed === undefined ? "—" : fmt(observed, 2)}`);
  }
  if (games !== null && games !== undefined) {
    const w = wins ?? 0;
    const t = ties ?? 0;
    const l = games - w - t;
    lines.push(`W-L-T: ${w}-${l}-${t} (n=${games})`);
  }
  if (expected !== null && expected !== undefined && observed !== null && observed !== undefined) {
    lines.push(`Δ(observed-expected): ${(observed - expected).toFixed(2)}`);
  }
  return lines.join("\n");
}

function showTooltip(text, x, y) {
  el.tooltip.classList.remove("hidden");
  el.tooltip.textContent = text;
  const pad = 14;
  const maxX = window.innerWidth - el.tooltip.offsetWidth - pad;
  const maxY = window.innerHeight - el.tooltip.offsetHeight - pad;
  el.tooltip.style.left = `${Math.max(pad, Math.min(maxX, x + 14))}px`;
  el.tooltip.style.top = `${Math.max(pad, Math.min(maxY, y + 14))}px`;
}

function hideTooltip() {
  el.tooltip.classList.add("hidden");
}

function renderHeatmap(summary) {
  const pairwise = summary.pairwise;
  if (!pairwise || !pairwise.models || !pairwise.models.length) {
    el.heatmap.innerHTML = `<div class="muted">No data to plot.</div>`;
    return;
  }

  const metric = state.heatmapMetric;
  const matrix = metric === "observed" ? pairwise.observed : pairwise.expected;
  const models = pairwise.models;
  const n = models.length;

  el.heatmap.innerHTML = "";
  el.heatmap.style.gridTemplateColumns = `var(--label-width) repeat(${n}, var(--cell-size))`;
  el.heatmap.style.gridTemplateRows = `var(--top-label-height) repeat(${n}, var(--cell-size))`;

  const corner = document.createElement("div");
  corner.className = "hm-corner";
  el.heatmap.appendChild(corner);

  for (let j = 0; j < n; j++) {
    const d = document.createElement("div");
    d.className = "hm-col-label";
    const s = document.createElement("span");
    s.textContent = models[j];
    s.title = models[j];
    d.appendChild(s);
    el.heatmap.appendChild(d);
  }

  for (let i = 0; i < n; i++) {
    const rl = document.createElement("div");
    rl.className = "hm-row-label";
    rl.textContent = models[i];
    rl.title = models[i];
    el.heatmap.appendChild(rl);

    for (let j = 0; j < n; j++) {
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.dataset.i = String(i);
      cell.dataset.j = String(j);

      if (i === j) {
        cell.classList.add("diag");
        el.heatmap.appendChild(cell);
        continue;
      }

      const v = matrix?.[i]?.[j];
      if (v === null || v === undefined) {
        cell.classList.add("empty");
        cell.textContent = "—";
      } else {
        cell.style.background = colorForValue(v);
        cell.style.color = textColorForValue(v);
        cell.textContent = fmt(v, 2);
      }
      el.heatmap.appendChild(cell);
    }
  }

  el.heatmap.onmousemove = (ev) => {
    const target = ev.target.closest(".hm-cell");
    if (!target) return;
    if (target.classList.contains("diag")) {
      hideTooltip();
      return;
    }
    const i = Number(target.dataset.i);
    const j = Number(target.dataset.j);
    if (!Number.isFinite(i) || !Number.isFinite(j)) return;
    showTooltip(cellTooltipContent(pairwise, i, j, metric), ev.clientX, ev.clientY);
  };

  el.heatmap.onmouseleave = () => hideTooltip();
}

/* --- Details Rendering --- */

function chapterMatchCount(chapter, judge) {
  if (judge === "combined") return chapter.total_matches || 0;
  return (chapter.by_judge && chapter.by_judge[judge]) || 0;
}

function renderChapterList() {
  const chapters = (state.index?.chapters || []).slice();
  const query = el.chapterSearch.value.trim().toLowerCase();
  const filtered = query ? chapters.filter((c) => c.sample.toLowerCase().includes(query)) : chapters;

  el.chapterList.innerHTML = "";

  if (filtered.length === 0) {
    el.chapterList.innerHTML = `<div class="muted p-sm">No chapters found.</div>`;
    return;
  }

  for (const ch of filtered) {
    const sample = ch.sample;
    const count = chapterMatchCount(ch, state.judge);

    const item = document.createElement("div");
    item.className = "list-item";
    if (state.selectedChapter === sample) {
      item.classList.add("active");
    }

    const name = document.createElement("div");
    name.textContent = sample;
    name.style.fontWeight = "500";
    name.style.whiteSpace = "nowrap";
    name.style.overflow = "hidden";
    name.style.textOverflow = "ellipsis";
    name.style.fontSize = "13px";

    const meta = document.createElement("div");
    meta.className = "badge";
    meta.textContent = `${count}`;
    meta.style.minWidth = "28px";
    meta.style.flexShrink = "0";

    item.appendChild(name);
    item.appendChild(meta);

    item.onclick = () => selectChapter(sample);
    el.chapterList.appendChild(item);
  }
}

async function selectChapter(sample) {
  state.selectedChapter = sample;
  state.selectedMatch = null;
  renderChapterList(); // update active state

  el.matchList.innerHTML = `<div class="muted p-sm">Loading matches...</div>`;
  el.matchDetail.innerHTML = `<div class="placeholder-text">Select a match to view details.</div>`;
  el.matchCount.textContent = "";

  // Auto-navigate to matches column on mobile
  if (isMobile()) {
    setMobileDetailsCol("matches");
  }

  const matches = await fetchChapterMatches(sample);
  renderMatchList(matches);
}

async function fetchChapterMatches(sample) {
  const key = `${state.judge}|${sample}`;
  if (state.chapterCache.has(key)) {
    return state.chapterCache.get(key);
  }

  try {
    const data = await fetchJSON(`/api/chapter?sample=${encodeURIComponent(sample)}&judge=${encodeURIComponent(state.judge)}`);
    const matches = data.matches || [];
    state.chapterCache.set(key, matches);
    return matches;
  } catch (e) {
    el.matchList.innerHTML = `<div class="muted p-sm text-danger">Error: ${e.message}</div>`;
    return [];
  }
}

function renderMatchList(matches) {
  el.matchList.innerHTML = "";
  el.matchCount.textContent = `${matches.length} matches`;

  if (matches.length === 0) {
    el.matchList.innerHTML = `<div class="muted p-sm">No matches found for this judge.</div>`;
    return;
  }

  matches.forEach((m, idx) => {
    const item = document.createElement("div");
    item.className = "list-item";
    const matchId = `${m.model_1}-vs-${m.model_2}-${idx}`;
    m._ui_id = matchId;

    if (state.selectedMatch === matchId) {
      item.classList.add("active");
    }

    const content = document.createElement("div");
    content.style.minWidth = "0";
    content.style.display = "grid";
    content.style.gap = "4px";

    const vsRow = document.createElement("div");
    vsRow.className = "match-vs";

    const model1 = document.createElement("span");
    model1.textContent = m.model_1;
    model1.style.color = (m.winner_model === m.model_1) ? "var(--success)" : "var(--text)";

    const sep = document.createElement("span");
    sep.className = "vs-separator";
    sep.textContent = "vs";

    const model2 = document.createElement("span");
    model2.textContent = m.model_2;
    model2.style.color = (m.winner_model === m.model_2) ? "var(--success)" : "var(--text)";

    vsRow.appendChild(model1);
    vsRow.appendChild(sep);
    vsRow.appendChild(model2);
    content.appendChild(vsRow);

    const sub = document.createElement("div");
    sub.className = "meta";

    const winnerDisplay = m.winner_model ? m.winner_model : "Tie";
    sub.textContent = `Winner: ${winnerDisplay}`;

    content.appendChild(sub);
    item.appendChild(content);

    // Winner badge
    const badge = document.createElement("span");
    badge.className = "winner-indicator";
    if (m.winner_model) {
      badge.classList.add(m.winner_model === m.model_1 ? "win-a" : "win-b");
      badge.textContent = "W";
    } else {
      badge.classList.add("result-tie");
      badge.textContent = "T";
    }
    item.appendChild(badge);

    item.onclick = () => selectMatch(m);
    el.matchList.appendChild(item);
  });
}

function selectMatch(match) {
  state.selectedMatch = match._ui_id;
  // Re-render list to update selection
  const matches = state.chapterCache.get(`${state.judge}|${state.selectedChapter}`);
  if (matches) renderMatchList(matches);
  renderMatchDetail(match);

  // Auto-navigate to result column on mobile
  if (isMobile()) {
    setMobileDetailsCol("result");
  }
}

function renderMatchDetail(match) {
  el.matchDetail.innerHTML = "";

  const header = document.createElement("div");
  header.className = "detail-header";

  const h2 = document.createElement("h2");
  h2.style.fontSize = "17px";
  h2.style.marginBottom = "8px";
  h2.style.fontWeight = "700";
  h2.style.letterSpacing = "-0.02em";

  const m1Span = document.createElement("span");
  m1Span.textContent = match.model_1;
  m1Span.style.color = (match.winner_model === match.model_1) ? "var(--success)" : "var(--text)";

  const vsSpan = document.createElement("span");
  vsSpan.textContent = "  vs  ";
  vsSpan.style.color = "var(--muted)";
  vsSpan.style.fontWeight = "400";
  vsSpan.style.fontSize = "14px";

  const m2Span = document.createElement("span");
  m2Span.textContent = match.model_2;
  m2Span.style.color = (match.winner_model === match.model_2) ? "var(--success)" : "var(--text)";

  h2.appendChild(m1Span);
  h2.appendChild(vsSpan);
  h2.appendChild(m2Span);
  header.appendChild(h2);

  const meta = document.createElement("div");
  meta.className = "detail-meta";

  const judgeMeta = document.createElement("span");
  judgeMeta.className = "detail-meta-item";
  judgeMeta.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> <strong>${match.judge_model}</strong>`;
  meta.appendChild(judgeMeta);

  const winnerMeta = document.createElement("span");
  winnerMeta.className = "detail-meta-item";
  const winnerPill = document.createElement("span");
  winnerPill.className = `pill ${match.winner_model ? 'win' : 'tie'}`;
  winnerPill.textContent = match.winner_model || 'Tie';
  const winLabel = document.createElement("span");
  winLabel.textContent = "Winner: ";
  winnerMeta.appendChild(winLabel);
  winnerMeta.appendChild(winnerPill);
  meta.appendChild(winnerMeta);

  if (match.decision?.confidence) {
    const confMeta = document.createElement("span");
    confMeta.className = "detail-meta-item";
    confMeta.innerHTML = `Confidence: <strong>${fmt(match.decision.confidence, 2)}</strong>`;
    meta.appendChild(confMeta);
  }

  header.appendChild(meta);
  el.matchDetail.appendChild(header);

  // Decision Info
  const body = document.createElement("div");
  body.className = "chapter-body";
  body.style.border = "none";
  body.style.padding = "0";

  const decision = match.decision || {};

  if (decision.final_summary) {
    const summaryBlock = document.createElement("div");
    summaryBlock.style.padding = "14px";
    summaryBlock.style.borderRadius = "var(--radius-sm)";
    summaryBlock.style.background = "var(--accent-soft)";
    summaryBlock.style.border = "1px solid rgba(236, 72, 153, 0.15)";
    summaryBlock.style.fontSize = "13px";
    summaryBlock.style.lineHeight = "1.6";

    const summaryLabel = document.createElement("div");
    summaryLabel.style.fontSize = "11px";
    summaryLabel.style.fontWeight = "600";
    summaryLabel.style.textTransform = "uppercase";
    summaryLabel.style.letterSpacing = "0.06em";
    summaryLabel.style.color = "var(--sakura-400)";
    summaryLabel.style.marginBottom = "6px";
    summaryLabel.textContent = "Final Summary";

    const summaryText = document.createElement("div");
    summaryText.textContent = decision.final_summary;

    summaryBlock.appendChild(summaryLabel);
    summaryBlock.appendChild(summaryText);
    body.appendChild(summaryBlock);
  }

  if (Array.isArray(decision.key_differences) && decision.key_differences.length) {
    const kvd = document.createElement("div");
    kvd.style.padding = "14px";
    kvd.style.borderRadius = "var(--radius-sm)";
    kvd.style.background = "var(--panel)";
    kvd.style.border = "1px solid var(--border)";

    const kdLabel = document.createElement("div");
    kdLabel.style.fontSize = "11px";
    kdLabel.style.fontWeight = "600";
    kdLabel.style.textTransform = "uppercase";
    kdLabel.style.letterSpacing = "0.06em";
    kdLabel.style.color = "var(--accent2)";
    kdLabel.style.marginBottom = "8px";
    kdLabel.textContent = "Key Differences";

    kvd.appendChild(kdLabel);

    const ul = document.createElement("ul");
    ul.style.margin = "0";
    ul.style.paddingLeft = "18px";
    ul.style.display = "grid";
    ul.style.gap = "6px";
    ul.style.fontSize = "13px";
    ul.style.lineHeight = "1.5";
    for (const item of decision.key_differences) {
      const li = document.createElement("li");
      li.textContent = String(item);
      ul.appendChild(li);
    }
    kvd.appendChild(ul);
    body.appendChild(kvd);
  }

  if (decision.scores) {
    const scoresTable = renderScoresTable(decision.scores, match.presentation);
    if (scoresTable) {
      body.appendChild(scoresTable);
    }
  }

  // Text details
  const src = renderTextDetails("Source", match.source_file);
  if (src) body.appendChild(src);
  const tf = match.translation_files || {};
  const tA = renderTextDetails("Translation A", tf.A);
  if (tA) body.appendChild(tA);
  const tB = renderTextDetails("Translation B", tf.B);
  if (tB) body.appendChild(tB);

  el.matchDetail.appendChild(body);
}

function renderScoresTable(scores, presentation) {
  if (!scores || typeof scores !== "object") return null;

  const aLabel = presentation?.A ? `A (${presentation.A})` : "A";
  const bLabel = presentation?.B ? `B (${presentation.B})` : "B";

  const wrap = document.createElement("div");
  wrap.className = "scores";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");

  ["Category", aLabel, bLabel, "Notes"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  Object.keys(scores)
    .sort()
    .forEach((k) => {
      const row = document.createElement("tr");
      const cat = document.createElement("td");
      cat.textContent = k;
      cat.style.fontWeight = "500";
      row.appendChild(cat);

      const details = scores[k] || {};
      const aVal = details.A !== undefined ? Number(details.A) : null;
      const bVal = details.B !== undefined ? Number(details.B) : null;

      const a = document.createElement("td");
      a.className = "col-num";
      a.textContent = aVal !== null ? String(details.A) : "—";
      if (aVal !== null && bVal !== null) {
        if (aVal > bVal) a.classList.add("score-better");
        else if (aVal < bVal) a.classList.add("score-worse");
        else a.classList.add("score-equal");
      }
      row.appendChild(a);

      const b = document.createElement("td");
      b.className = "col-num";
      b.textContent = bVal !== null ? String(details.B) : "—";
      if (aVal !== null && bVal !== null) {
        if (bVal > aVal) b.classList.add("score-better");
        else if (bVal < aVal) b.classList.add("score-worse");
        else b.classList.add("score-equal");
      }
      row.appendChild(b);

      const notes = document.createElement("td");
      notes.textContent = details.notes !== undefined ? String(details.notes) : "";
      notes.style.fontSize = "12px";
      notes.style.color = "var(--text-secondary)";
      row.appendChild(notes);

      tbody.appendChild(row);
    });

  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

function renderTextDetails(label, path) {
  if (!path) return null;

  const d = document.createElement("details");
  d.className = "match match-text";
  const s = document.createElement("summary");

  const left = document.createElement("div");
  left.className = "match-title";
  const strong = document.createElement("strong");
  strong.textContent = label;
  left.appendChild(strong);

  const pill = document.createElement("span");
  pill.className = "pill";
  pill.textContent = path;
  left.appendChild(pill);

  s.appendChild(left);
  d.appendChild(s);

  const body = document.createElement("div");
  body.className = "match-body";
  const preWrap = document.createElement("div");
  preWrap.className = "pre";
  const pre = document.createElement("pre");
  pre.textContent = "Open to load…";
  preWrap.appendChild(pre);
  body.appendChild(preWrap);
  d.appendChild(body);

  let loaded = false;
  d.addEventListener("toggle", async () => {
    if (!d.open || loaded) return;
    loaded = true;
    try {
      const resp = await fetchJSON(`/api/text?path=${encodeURIComponent(path)}`);
      const note = resp.truncated ? "\n\n[truncated]" : "";
      pre.textContent = (resp.text || "") + note;
    } catch (e) {
      pre.textContent = `Error: ${e.message}`;
    }
  });

  return d;
}

/* --- Initialization --- */

async function loadIndexAndSummary() {
  setBanner("");
  const index = await fetchJSON("/api/index");
  state.index = index;
  populateJudges(index);

  if (state.page === "details") {
    renderChapterList();
  }

  await loadSummary();
}

async function loadSummary() {
  setBanner("");
  const summary = await fetchJSON(`/api/summary?judge=${encodeURIComponent(state.judge)}`);
  state.summary = summary;
  renderCards(state.index, summary);
  renderLeaderboard(summary);
  renderBarPlot(summary);
  renderHeatmap(summary);
}

function invalidateChapterCache() {
  state.chapterCache.clear();
  state.selectedChapter = null;
  state.selectedMatch = null;
  el.matchList.innerHTML = "";
  el.matchDetail.innerHTML = `<div class="placeholder-text">Select a chapter and a match to view details.</div>`;
  if (state.page === "details") {
    renderChapterList();
  }
}

async function onReload() {
  setBanner("");
  el.reloadBtn.disabled = true;
  el.reloadBtn.textContent = "Reloading…";
  try {
    await fetchJSON("/api/reload");
    invalidateChapterCache();
    await loadIndexAndSummary();
  } catch (e) {
    setBanner(e.message);
  } finally {
    el.reloadBtn.disabled = false;
    el.reloadBtn.textContent = "Reload";
  }
}

function wireEvents() {
  el.judgeSelect.addEventListener("change", async () => {
    state.judge = el.judgeSelect.value || "combined";
    invalidateChapterCache();
    await loadSummary();
  });

  el.heatmapMetric.addEventListener("change", () => {
    state.heatmapMetric = el.heatmapMetric.value;
    if (state.summary) renderHeatmap(state.summary);
  });

  el.reloadBtn.addEventListener("click", onReload);

  el.pageSwitcher.addEventListener("click", () => {
    // Remove pulse animation on first click
    el.pageSwitcher.classList.remove("pulse");
    if (state.page === "overview") {
      setPage("details");
      if (el.chapterList.children.length === 0) {
        renderChapterList();
      }
    } else {
      setPage("overview");
    }
  });

  el.chapterSearch.addEventListener("input", () => {
    renderChapterList();
  });

  // Mobile bottom navigation
  if (el.mobileNav) {
    el.mobileNav.addEventListener("click", (e) => {
      const btn = e.target.closest(".mobile-nav-item");
      if (!btn) return;
      const page = btn.dataset.page;
      if (page === state.page) return;

      setPage(page);
      if (page === "details" && el.chapterList.children.length === 0) {
        renderChapterList();
      }
    });
  }

  // Details sub-navigation (mobile column switcher)
  if (el.detailsSubNav) {
    el.detailsSubNav.addEventListener("click", (e) => {
      const btn = e.target.closest(".subnav-item");
      if (!btn) return;
      setMobileDetailsCol(btn.dataset.col);
    });
  }

  // Handle resize — re-apply or remove mobile column classes + sync topbar
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      syncTopbarHeight();
      const grid = document.querySelector(".details-grid");
      if (!grid) return;

      if (isMobile()) {
        if (state.page === "details") {
          el.detailsSubNav.classList.remove("hidden");
          setMobileDetailsCol(state.mobileDetailsCol || "chapters");
        }
      } else {
        // Remove mobile-specific classes on desktop
        grid.classList.remove("mobile-col-chapters", "mobile-col-matches", "mobile-col-result");
        el.detailsSubNav.classList.add("hidden");
      }
    }, 150);
  });
}

// Start
wireEvents();
syncTopbarHeight();
loadIndexAndSummary().catch((e) => setBanner(e.message));
