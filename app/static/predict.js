(function () {
  var LEVELS = ["Low", "Moderate", "High", "Severe"];
  var KEY = "trafficPredictorLast";
  var form = document.getElementById("predict-form");

  // Put back the values the user typed before pressing Predict
  var saved = {};
  try { saved = JSON.parse(form.getAttribute("data-saved") || "{}"); } catch (e) {}
  Object.keys(saved).forEach(function (name) {
    var field = form.elements[name];
    if (field) { field.value = saved[name]; }
  });

  // Fill the probability bars
  document.querySelectorAll(".fill").forEach(function (bar) {
    requestAnimationFrame(function () { bar.style.width = bar.getAttribute("data-width") + "%"; });
  });

  var result = document.getElementById("result");
  var box = document.getElementById("delta");
  if (!result || !box) return;
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });

  var cur = { form: saved, label: result.getAttribute("data-label"), probs: {} };
  try { cur.probs = JSON.parse(result.getAttribute("data-probs") || "{}"); } catch (e) {}

  var store = {};
  try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }

  var prev = null;
  if (store.cur && JSON.stringify(store.cur.form) === JSON.stringify(cur.form)) {
    prev = store.prev || null;                 // page was reloaded, keep the earlier comparison
  } else {
    prev = store.cur || null;
    try { localStorage.setItem(KEY, JSON.stringify({ cur: cur, prev: prev })); } catch (e) {}
  }
  if (!prev) return;

  function labelFor(name) {
    var f = form.elements[name];
    var l = f && f.closest("div") && f.closest("div").querySelector("label");
    return l ? l.textContent.trim() : name;
  }
  function shown(name, v) {
    var f = form.elements[name];
    if (f && f.tagName === "SELECT") {
      for (var i = 0; i < f.options.length; i++) if (f.options[i].value === String(v)) return f.options[i].text;
    }
    return v;
  }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function mini(title, d) {
    var m = el("div", "mini " + d.label);
    m.appendChild(el("small", "", title));
    m.appendChild(el("b", "", d.label));
    return m;
  }

  box.appendChild(el("h2", "", "Compared with your previous prediction"));
  var cmp = el("div", "cmp");
  cmp.appendChild(mini("Previous", prev));
  cmp.appendChild(mini("Now", cur));
  box.appendChild(cmp);

  // what changed in the inputs
  var changed = [];
  Object.keys(cur.form).forEach(function (k) {
    if (String(cur.form[k]) !== String(prev.form[k])) {
      changed.push(labelFor(k) + ": " + shown(k, prev.form[k]) + " to " + shown(k, cur.form[k]));
    }
  });

  var sentence;
  if (prev.label === cur.label) sentence = "The predicted level stays at " + cur.label + ".";
  else {
    var up = LEVELS.indexOf(cur.label) > LEVELS.indexOf(prev.label);
    sentence = "The predicted level " + (up ? "rises" : "falls") + " from " + prev.label + " to " + cur.label + ".";
  }
  box.appendChild(el("p", "", sentence));

  var big = null;
  LEVELS.forEach(function (n) {
    if (prev.probs[n] === undefined || cur.probs[n] === undefined) return;
    var d = (cur.probs[n] - prev.probs[n]) * 100;
    if (!big || Math.abs(d) > Math.abs(big.d)) big = { n: n, d: d };
  });
  if (big && Math.abs(big.d) >= 0.5) {
    box.appendChild(el("p", "hint", "Biggest probability shift: " + big.n + " " + (big.d > 0 ? "+" : "") + big.d.toFixed(1) + " points."));
  }

  if (changed.length) {
    box.appendChild(el("h2", "", "What you changed"));
    var ul = el("ul", "diffs");
    changed.forEach(function (c) { ul.appendChild(el("li", "", c)); });
    box.appendChild(ul);
  }
  box.hidden = false;
})();
