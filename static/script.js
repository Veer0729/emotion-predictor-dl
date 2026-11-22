(() => {
  const META = {
    sadness:  { emoji: "😢", color: "#5b8def", line: "A heavy, tender feeling." },
    joy:      { emoji: "😄", color: "#ffc94a", line: "Bright and full of energy." },
    love:     { emoji: "❤️", color: "#ff6f9f", line: "Warm, close and caring." },
    anger:    { emoji: "😠", color: "#ff5a4d", line: "Sharp, hot and frustrated." },
    fear:     { emoji: "😨", color: "#9b6bff", line: "Uneasy and on edge." },
    surprise: { emoji: "😲", color: "#3fd9c2", line: "Unexpected and a little breathless." },
  };
  const RING = 326.7; // circumference for r = 52

  const $ = (id) => document.getElementById(id);
  const input = $("textInput"), btn = $("analyzeBtn"), count = $("charCount");
  const result = $("result"), errorBox = $("errorMsg"), bars = $("bars");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  input.addEventListener("input", () => {
    count.textContent = input.value.length;
    btn.disabled = !input.value.trim();
  });

  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !btn.disabled) analyze();
  });


  btn.addEventListener("click", analyze);

  async function analyze() {
    const text = input.value.trim();
    if (!text) return;
    setLoading(true);
    errorBox.hidden = true;
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(
          res.status === 503
            ? "The model is still loading. Try again in a few seconds."
            : typeof data.detail === "string" ? data.detail : "Something went wrong while analyzing your text."
        );
      }
      render(await res.json());
    } catch (err) {
      errorBox.textContent = err instanceof TypeError
        ? "Can't reach the server. Check your connection and try again."
        : err.message;
      errorBox.hidden = false;
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on) {
    btn.classList.toggle("loading", on);
    btn.disabled = on || !input.value.trim();
    btn.querySelector(".btn-label").textContent = on ? "Reading…" : "Analyze emotion";
  }

  function render(data) {
    const emotion = data.predicted_emotion;
    const meta = META[emotion] || META.joy;
    const probs = data.all_probabilites || data.all_probabilities || {};
    const conf = Math.round(data.confidence * 1000) / 10;

    document.body.dataset.emotion = emotion;
    document.body.classList.add("pulse");
    setTimeout(() => document.body.classList.remove("pulse"), 900);

    $("emotionName").textContent = emotion;
    $("quote").textContent = meta.line;
    const emoji = $("emoji");
    emoji.textContent = meta.emoji;

    // restart entrance animations
    result.hidden = false;
    [result, emoji].forEach((el) => el.classList.remove("show", "pop"));
    void result.offsetWidth;
    result.classList.add("show");
    emoji.classList.add("pop");

    // confidence ring + counter
    const ring = $("ringFill");
    ring.style.strokeDashoffset = RING;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ring.style.strokeDashoffset = RING * (1 - data.confidence);
    }));
    countUp($("confNum"), conf);

    // probability bars, highest first
    bars.innerHTML = "";
    Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([name, p], i) => {
      const m = META[name] || {};
      const li = document.createElement("li");
      if (name === emotion) li.className = "top";
      li.innerHTML = `<span>${m.emoji || ""}</span><span class="name">${name}</span>
        <span class="track-bar"><i style="--c:${m.color || "#888"}"></i></span>
        <span class="pct">${(p * 100).toFixed(1)}%</span>`;
      bars.appendChild(li);
      setTimeout(() => (li.querySelector("i").style.width = `${p * 100}%`), 120 + i * 90);
    });

    result.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
  }

  function countUp(el, target) {
    if (reduced) { el.textContent = target.toFixed(1); return; }
    const start = performance.now(), dur = 1200;
    (function tick(now) {
      const t = Math.min((now - start) / dur, 1);
      el.textContent = (target * (1 - Math.pow(1 - t, 3))).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
    })(start);
  }
})();
