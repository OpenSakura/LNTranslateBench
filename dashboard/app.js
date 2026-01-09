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
  const red = [0xb4, 0x04, 0x26];
  const white = [0xf7, 0xf7, 0xf7];
  const blue = [0x05, 0x30, 0x61];

  let rgb;
  if (x < 0.5) {
    const t = x / 0.5;
    rgb = [lerp(red[0], white[0], t), lerp(red[1], white[1], t), lerp(red[2], white[2], t)];
  } else {
    const t = (x - 0.5) / 0.5;
    rgb = [lerp(white[0], blue[0], t), lerp(white[1], blue[1], t), lerp(white[2], blue[2], t)];
  }

  const [r, g, b] = rgb.map((n) => Math.round(n));
  return `rgb(${r}, ${g}, ${b})`;
}

function setPage(pageName) {
  state.page = pageName;
  if (pageName === "overview") {
    el.pageSwitcher.classList.remove("left");
    el.pageSwitcher.innerHTML = `<span class="arrow">»</span> <span class="label">Detailed View</span>`;
    el.pageOverview.classList.remove("hidden");
    el.pageDetails.classList.add("hidden");
  } else {
    el.pageSwitcher.classList.add("left");
    el.pageSwitcher.innerHTML = `<span class="arrow">«</span> <span class="label">Overview</span>`;
    el.pageOverview.classList.add("hidden");
    el.pageDetails.classList.remove("hidden");
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
  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");

    const tdRank = document.createElement("td");
    tdRank.className = "col-rank";
    tdRank.textContent = String(idx + 1);
    tr.appendChild(tdRank);

    const tdModel = document.createElement("td");
    tdModel.textContent = r.model || "—";
    tr.appendChild(tdModel);

    const tdElo = document.createElement("td");
    tdElo.className = "col-num";
    tdElo.textContent = fmt(r.elo, 2);
    tr.appendChild(tdElo);

    const tdGames = document.createElement("td");
    tdGames.className = "col-num";
    tdGames.textContent = String(r.games ?? 0);
    tr.appendChild(tdGames);

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
  const barW = 28;
  const gap = 3;
  const padding = { l: 10, r: 10, t: 24, b: 100 };
  const width = padding.l + padding.r + n * (barW + gap) - gap;
  const height = 320;
  const innerH = height - padding.t - padding.b;

  const maxElo = Math.max(...ratings.map((r) => r.elo));
  const minElo = Math.min(...ratings.map((r) => r.elo));
  const range = maxElo - minElo || 1;

  function barColor(rank, total) {
    const t = total > 1 ? rank / (total - 1) : 0;
    if (t < 0.5) {
      const r = Math.round(34 + (250 - 34) * (t * 2));
      const g = Math.round(197 - (197 - 204) * (t * 2));
      const b = Math.round(94 - (94 - 21) * (t * 2));
      return `rgb(${r},${g},${b})`;
    } else {
      const r = Math.round(250 + (239 - 250) * ((t - 0.5) * 2));
      const g = Math.round(204 - (204 - 68) * ((t - 0.5) * 2));
      const b = Math.round(21 + (68 - 21) * ((t - 0.5) * 2));
      return `rgb(${r},${g},${b})`;
    }
  }

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));

  for (let i = 0; i < n; i++) {
    const r = ratings[i];
    const x = padding.l + i * (barW + gap);
    const barH = ((r.elo - minElo) / range) * innerH * 0.85 + innerH * 0.15;
    const y = padding.t + innerH - barH;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(barW));
    rect.setAttribute("height", String(barH));
    rect.setAttribute("rx", "4");
    rect.setAttribute("fill", barColor(i, n));
    svg.appendChild(rect);

    const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
    score.setAttribute("x", String(x + barW / 2));
    score.setAttribute("y", String(y - 6));
    score.setAttribute("fill", "currentColor");
    score.setAttribute("font-size", "10");
    score.setAttribute("text-anchor", "middle");
    score.textContent = fmt(r.elo, 0);
    svg.appendChild(score);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(x + barW / 2));
    label.setAttribute("y", String(padding.t + innerH + 10));
    label.setAttribute("fill", "currentColor");
    label.setAttribute("font-size", "11");
    label.setAttribute("text-anchor", "end");
    label.setAttribute("transform", `rotate(-45, ${x + barW / 2}, ${padding.t + innerH + 10})`);
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
    d.appendChild(s);
    el.heatmap.appendChild(d);
  }

  for (let i = 0; i < n; i++) {
    const rl = document.createElement("div");
    rl.className = "hm-row-label";
    rl.textContent = models[i];
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

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${count}`;

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

    const title = document.createElement("div");
    title.textContent = `${m.model_1} vs ${m.model_2}`;
    title.style.fontWeight = "600";
    title.style.fontSize = "12px";

    const sub = document.createElement("div");
    sub.className = "meta";

    const winnerDisplay = m.winner_model ? m.winner_model : "Tie";
    sub.textContent = `Winner: ${winnerDisplay}`;

    content.appendChild(title);
    content.appendChild(sub);

    item.appendChild(content);
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
}

function renderMatchDetail(match) {
  el.matchDetail.innerHTML = "";

  const header = document.createElement("div");
  header.style.marginBottom = "14px";
  header.style.paddingBottom = "10px";
  header.style.borderBottom = "1px solid var(--border)";
  header.innerHTML = `
    <h2 style="font-size:16px; margin-bottom:4px">${match.model_1} vs ${match.model_2}</h2>
    <div class="muted" style="font-size:12px">
      Judge: <strong>${match.judge_model}</strong> • 
      Winner: <span class="pill ${match.winner_model ? 'win' : 'tie'}">${match.winner_model || 'Tie'}</span>
      ${match.decision?.confidence ? `• Confidence: ${fmt(match.decision.confidence, 2)}` : ''}
    </div>
  `;
  el.matchDetail.appendChild(header);

  // Decision Info
  const body = document.createElement("div");
  body.className = "chapter-body";
  body.style.border = "none";
  body.style.padding = "0";

  const decision = match.decision || {};

  if (decision.final_summary) {
    const kvs = document.createElement("div");
    kvs.className = "kv";
    kvs.innerHTML = `<div class="k">Final Summary</div><div>${decision.final_summary}</div>`;
    body.appendChild(kvs);
  }

  if (Array.isArray(decision.key_differences) && decision.key_differences.length) {
    const kvd = document.createElement("div");
    kvd.className = "kv";
    const k = document.createElement("div");
    k.className = "k";
    k.textContent = "Key differences";
    const v = document.createElement("div");
    const ul = document.createElement("ul");
    ul.style.marginTop = "0";
    for (const item of decision.key_differences) {
      const li = document.createElement("li");
      li.textContent = String(item);
      ul.appendChild(li);
    }
    v.appendChild(ul);
    kvd.appendChild(k);
    kvd.appendChild(v);
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
      row.appendChild(cat);

      const details = scores[k] || {};
      const a = document.createElement("td");
      a.className = "col-num";
      a.textContent = details.A !== undefined ? String(details.A) : "—";
      row.appendChild(a);

      const b = document.createElement("td");
      b.className = "col-num";
      b.textContent = details.B !== undefined ? String(details.B) : "—";
      row.appendChild(b);

      const notes = document.createElement("td");
      notes.textContent = details.notes !== undefined ? String(details.notes) : "";
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
}

// Start
wireEvents();
loadIndexAndSummary().catch((e) => setBanner(e.message));
