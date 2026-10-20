(function () {
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  var figs = JSON.parse($("figdata").textContent);
  var presets = JSON.parse($("presetdata").textContent);
  var base = JSON.parse($("urlbase").textContent);
  var byId = {};
  figs.forEach(function (f) { byId[f.id] = f; });

  // ---- model table: sort + highlight the best value per column
  var table = $("model-table");
  if (table) {
    var tbody = table.tBodies[0], heads = table.tHead.rows[0].cells;
    var LOWER = { mae: 1, rmse: 1, params: 0, secs: 1 };   // columns where smaller is better
    function markBest() {
      for (var c = 3; c < heads.length; c++) {
        var k = heads[c].getAttribute("data-k"); if (k === "params") continue;
        var vals = [].map.call(tbody.rows, function (r) { return parseFloat(r.cells[c].getAttribute("data-v")); });
        var cand = vals.filter(function (v) { return v >= 0 || k !== "secs"; });
        var best = LOWER[k] ? Math.min.apply(null, cand) : Math.max.apply(null, cand);
        var bi = vals.indexOf(best), shownBest = tbody.rows[bi].cells[c].textContent;
        [].forEach.call(tbody.rows, function (r) { r.cells[c].classList.toggle("bestcell", r.cells[c].textContent === shownBest); });
      }
    }
    markBest();
    var dir = {};
    [].forEach.call(heads, function (th, c) {
      th.addEventListener("click", function () {
        var k = th.getAttribute("data-k"), text = th.getAttribute("data-t") === "text";
        dir[k] = !dir[k];
        var rows = [].slice.call(tbody.rows);
        rows.sort(function (a, b) {
          var x, y;
          if (text) { x = a.cells[c].textContent; y = b.cells[c].textContent; return dir[k] ? x.localeCompare(y) : y.localeCompare(x); }
          x = parseFloat(a.cells[c].getAttribute("data-v") || a.cells[c].textContent);
          y = parseFloat(b.cells[c].getAttribute("data-v") || b.cells[c].textContent);
          return dir[k] ? x - y : y - x;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
        [].forEach.call(heads, function (h) { h.classList.remove("sorted", "desc"); });
        th.classList.add("sorted"); if (!dir[k]) th.classList.add("desc");
      });
    });
  }

  // ---- model chart: one bar per model, switch the metric
  var chart = $("chart");
  if (chart) {
    var models = JSON.parse($("modeldata").textContent);
    var METRICS = [
      { k: "mae", name: "MAE", lower: true, fmt: function (v) { return v.toFixed(2); }, note: "Mean absolute error in vehicles per 15 minutes. Lower is better." },
      { k: "rmse", name: "RMSE", lower: true, fmt: function (v) { return v.toFixed(2); }, note: "Root mean squared error in vehicles per 15 minutes. Penalises big misses. Lower is better." },
      { k: "r2", name: "R\u00b2", lower: false, fmt: function (v) { return v.toFixed(4); }, note: "Share of variance explained. Higher is better." },
      { k: "acc", name: "Accuracy", lower: false, fmt: function (v) { return (v * 100).toFixed(1) + "%"; }, note: "Share of test windows where the congestion level was right. Higher is better." },
      { k: "f1", name: "Macro-F1", lower: false, fmt: function (v) { return v.toFixed(3); }, note: "Average F1 across the four levels, so rare High and Severe count as much as Low. Higher is better." },
      { k: "secs", name: "Train time", lower: true, fmt: function (v) { return v < 1 ? "<1 s" : v.toFixed(0) + " s"; }, note: "Seconds to train on this machine. Lower is faster." }
    ];
    var COLORS = { "Baseline": "#9aa7b1", "Classical ML": "#38bdf8", "Deep learning": "#3fb26b", "Transformer": "#f0715a" };
    var mchips = $("metric-chips"), cur = METRICS[0];
    function drawChart() {
      chart.textContent = "";
      var rows = models.filter(function (m) { return m[cur.k] !== null && m[cur.k] !== undefined; });
      rows.sort(function (a, b) { return cur.lower ? a[cur.k] - b[cur.k] : b[cur.k] - a[cur.k]; });
      var vals = rows.map(function (m) { return m[cur.k]; });
      var max = Math.max.apply(null, vals), min = Math.min.apply(null, vals);
      var zero = cur.lower || cur.k === "secs";
      var lo = zero ? 0 : Math.max(0, min - (max - min) * 0.6);
      rows.forEach(function (m, i) {
        var row = el("div", "row" + (i === 0 ? " top" : ""));
        row.appendChild(el("span", "n", m.title)); row.lastChild.title = m.family;
        var bar = el("span", "bar"), fill = el("span");
        fill.style.background = COLORS[m.family] || "#777";
        var w = max === lo ? 100 : Math.max(2, (m[cur.k] - lo) / (max - lo) * 100);
        bar.appendChild(fill); row.appendChild(bar);
        row.appendChild(el("span", "v", cur.fmt(m[cur.k])));
        chart.appendChild(row);
        requestAnimationFrame(function () { fill.style.width = w + "%"; });
      });
      $("chart-note").textContent = cur.note + (zero ? "" : " The bars start at " + cur.fmt(lo) + " so small differences are visible, not at zero.");
      chart.setAttribute("aria-label", "Bar chart of " + cur.name + " for " + rows.length + " models, best first");
    }
    METRICS.forEach(function (mt) {
      var b = el("button", "chip" + (mt === cur ? " on" : ""), mt.name); b.type = "button";
      b.addEventListener("click", function () {
        cur = mt;
        [].forEach.call(mchips.children, function (c) { c.classList.toggle("on", c === b); });
        drawChart();
      });
      mchips.appendChild(b);
    });
    drawChart();
  }

  // animate the metric bars
  document.querySelectorAll(".metric .fill").forEach(function (b) {
    requestAnimationFrame(function () { b.style.width = b.getAttribute("data-width") + "%"; });
  });

  // ---- lightbox
  var lb = $("lightbox"), lbImg = $("lb-img"), lastFocus = null;
  function openLb(src, title) {
    lastFocus = document.activeElement;
    lbImg.src = src; lbImg.alt = title; lb.classList.add("open"); $("lb-close").focus();
  }
  function closeLb() { lb.classList.remove("open"); lbImg.removeAttribute("src"); if (lastFocus) lastFocus.focus(); }
  document.addEventListener("click", function (e) {
    var p = e.target.closest(".plate");
    if (p) { openLb(p.getAttribute("data-src"), p.getAttribute("data-title")); return; }
    if (e.target === lb || e.target.id === "lb-close") closeLb();
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && lb.classList.contains("open")) closeLb(); });

  // ---- gallery filter
  $("filters").addEventListener("click", function (e) {
    var b = e.target.closest(".chip"); if (!b) return;
    document.querySelectorAll("#filters .chip").forEach(function (c) { c.classList.toggle("on", c === b); });
    var f = b.getAttribute("data-f");
    document.querySelectorAll(".gallery .fig").forEach(function (fig) {
      fig.hidden = !(f === "all" || fig.getAttribute("data-group") === f);
    });
  });

  // ---- side-by-side comparison
  var selA = $("pick-a"), selB = $("pick-b"), side = $("side"), explain = $("explain"), chips = $("preset-chips");
  figs.forEach(function (f) {
    [selA, selB].forEach(function (s) {
      var o = el("option", "", f.title); o.value = f.id; s.appendChild(o);
    });
  });

  function card(f) {
    var fig = el("figure", "fig");
    var btn = el("button", "plate"); btn.type = "button";
    var src = base + f.path;
    btn.setAttribute("data-src", src); btn.setAttribute("data-title", f.title);
    btn.setAttribute("aria-label", "Enlarge " + f.title);
    var img = el("img"); img.src = src; img.alt = f.title + ": " + f.text;
    btn.appendChild(img); fig.appendChild(btn);
    var cap = el("figcaption"); cap.appendChild(el("b", "", f.title)); cap.appendChild(el("span", "", f.text));
    fig.appendChild(cap);
    return fig;
  }

  var activePreset = null;
  function render() {
    var a = byId[selA.value], b = byId[selB.value];
    side.textContent = ""; explain.textContent = "";
    if (!a || !b) return;
    side.appendChild(card(a)); side.appendChild(card(b));
    var p = presets.filter(function (x) { return (x.a === a.id && x.b === b.id) || (x.a === b.id && x.b === a.id); })[0];
    document.querySelectorAll("#preset-chips .chip").forEach(function (c) {
      c.classList.toggle("on", !!p && c.getAttribute("data-i") === String(presets.indexOf(p)));
    });
    if (a.id === b.id) { explain.appendChild(el("p", "hint", "Both sides show the same chart. Pick a different chart on one side to compare.")); return; }
    if (p) {
      explain.appendChild(el("h3", "", "How they differ: " + p.title));
      explain.appendChild(el("p", "sub", p.intro));
      var ul = el("ul", "diffs");
      p.diffs.forEach(function (d) { ul.appendChild(el("li", "", d)); });
      explain.appendChild(ul);
    } else {
      explain.appendChild(el("h3", "", "How they differ"));
      var ul2 = el("ul", "diffs");
      ul2.appendChild(el("li", "", a.title + ": " + a.text));
      ul2.appendChild(el("li", "", b.title + ": " + b.text));
      ul2.appendChild(el("li", "", "Choose one of the ready-made pairs above for a written comparison that includes the numbers."));
      explain.appendChild(ul2);
    }
  }

  presets.forEach(function (p, i) {
    var c = el("button", "chip", p.title); c.type = "button"; c.setAttribute("data-i", i);
    c.addEventListener("click", function () { selA.value = p.a; selB.value = p.b; render(); });
    chips.appendChild(c);
  });
  if (presets.length) { selA.value = presets[0].a; selB.value = presets[0].b; }
  else if (figs.length > 1) { selA.value = figs[0].id; selB.value = figs[1].id; }
  selA.addEventListener("change", render); selB.addEventListener("change", render);
  render();
})();
