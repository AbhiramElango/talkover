const RESOLVE_MS = 1200;   // after you stop talking, a still-talking bot counts as "kept talking"
const ATTACH_MS = 2500;    // a bubble with no words by then was only noise

let threshold = 0.5;
let active = null;

class Side {
  constructor(el) {
    this.el = el;
    this.name = el.dataset.side;
    this.orb = el.querySelector(".orb");
    this.state = el.querySelector(".state");
    this.meter = el.querySelector(".meter");
    this.fill = el.querySelector(".fill");
    this.chat = el.querySelector(".chat");
    this.tally = el.querySelector(".tally");
    this.button = el.querySelector(".call");
    this.button.onclick = () => (this.pc ? this.stop() : this.start());
    this.reset();
  }

  async start() {
    if (active) await active.stop();
    active = this;
    this.button.disabled = true;
    this.button.textContent = "Connecting…";
    this.setState("Connecting…");
    try {
      await this.connect();
      this.el.classList.add("live");
      this.button.textContent = "End call";
      this.setState("Listening");
    } catch (err) {
      console.error(err);
      this.button.textContent = "Couldn't connect, retry";
      this.teardown();
    }
    this.button.disabled = false;
  }

  async connect() {
    this.reset();
    this.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.audio = new AudioContext();
    this.micLevel = levelMeter(this.audio, this.mic);

    const pc = (this.pc = new RTCPeerConnection());
    pc.addTransceiver(this.mic.getAudioTracks()[0], { direction: "sendrecv" });
    pc.ontrack = (e) => {
      const stream = new MediaStream([e.track]);
      this.player = Object.assign(new Audio(), { srcObject: stream, autoplay: true });
      this.botLevel = levelMeter(this.audio, stream);
    };
    const channel = pc.createDataChannel("events", { ordered: true });
    channel.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.event) this.onEvent(m);
    };
    channel.onopen = () => (this.ping = setInterval(() => channel.send(`ping: ${Date.now()}`), 1000));
    pc.onconnectionstatechange = () => {
      if (["failed", "closed", "disconnected"].includes(pc.connectionState)) this.stop();
    };

    await pc.setLocalDescription(await pc.createOffer());
    await iceGathered(pc);
    const res = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: "offer", requestData: { side: this.name } }),
    });
    if (!res.ok) throw new Error(await res.text());
    await pc.setRemoteDescription(await res.json());
    this.animate();
  }

  async stop() {
    this.teardown();
    if (active === this) active = null;
    this.button.textContent = "Start call";
  }

  teardown() {
    clearInterval(this.ping);
    clearInterval(this.revealTimer);
    cancelAnimationFrame(this.frame);
    this.pc?.close();
    this.mic?.getTracks().forEach((t) => t.stop());
    this.audio?.close();
    if (this.player) this.player.srcObject = null;
    this.pc = this.mic = this.audio = this.player = this.micLevel = this.botLevel = null;
    this.el.classList.remove("live", "halt");
    this.orb.style.setProperty("--level", 0);
    this.orb.style.setProperty("--user", 0);
    this.setMeter(0);
    this.setState("Not connected");
  }

  reset() {
    this.botSpeaking = false;
    this.bot = null;        // bot bubble receiving words
    this.lastBot = null;    // last bot bubble, for the cut-off mark
    this.you = null;        // your bubble while you are speaking
    this.lastYou = null;
    this.sentence = null;
    this.chat.replaceChildren();
    this.counts = { stopped: 0, kept: 0 };
    this.renderTally();
  }

  onEvent(m) {
    switch (m.event) {
      case "bot":
        this.botSpeaking = m.speaking;
        if (m.speaking) {
          if (this.lastYou) this.lastYou.sealed = true;
          this.bot ??= this.addBot();
          this.setState("Bot speaking");
        } else {
          this.revealWords(true);
          if (this.bot?.empty()) this.bot.el.remove();
          this.bot = null;
          this.setMeter(0);
          this.setState("Listening");
        }
        break;
      case "caption":
        this.revealWords(true);
        this.startSentence(m.text, m.seconds, m.elapsed);
        break;
      case "user":
        if (m.speaking) this.youStarted();
        else this.youStopped();
        break;
      case "interrupted": {
        this.sentence = null;
        this.lastBot?.cut();
        this.bot = null;
        this.flash();
        this.setState("Stopped by you");
        const you = this.you || this.openYou() || this.openTemporarily(this.addYou("pending"));
        this.resolve(you, "stopped", true);
        break;
      }
      case "transcript":
        this.attach(m.text, m.final);
        break;
      case "prob":
        if (this.botSpeaking) this.setMeter(m.p);
        break;
    }
  }

  // Your bubble takes transcripts until the bot's next words appear (Deepgram finals can lag).
  openYou() {
    return this.lastYou && !this.lastYou.sealed ? this.lastYou : null;
  }

  openTemporarily(you) {
    this.you = you;
    setTimeout(() => this.you === you && this.youStopped(), RESOLVE_MS);
    return you;
  }

  youStarted() {
    if (this.you) return;
    const recent = this.openYou();
    if (recent?.status === "stopped" && performance.now() - recent.createdAt < RESOLVE_MS) {
      this.you = recent;   // the sound that just stopped the bot is still going
      return;
    }
    const overlap = this.botSpeaking;
    this.you = this.addYou(overlap ? "pending" : "turn");
    if (overlap && !this.bot?.empty()) this.bot = null;   // the bot's next words go in a new bubble below yours
  }

  youStopped() {
    const you = this.you;
    if (!you) return;
    this.you = null;
    if (you.status === "pending") setTimeout(() => this.resolve(you, "kept"), RESOLVE_MS);
    else if (you.status === "turn") setTimeout(() => !you.words && you.el.remove(), ATTACH_MS);   // a sound with no words is not a turn
  }

  // Bot words appear as they are spoken: each caption carries the sentence's audio length.
  startSentence(text, seconds, elapsed) {
    const words = text.split(/\s+/).filter(Boolean);
    const total = words.reduce((n, w) => n + w.length + 1, 0);
    let seen = 0;
    const ends = words.map((w) => (seen += w.length + 1) / total);
    this.sentence = { words, ends, next: 0, start: performance.now() - elapsed * 1000, ms: Math.max(seconds * 1000, 1) };
    clearInterval(this.revealTimer);
    this.revealTimer = setInterval(() => this.revealWords(), 50);
    this.revealWords();
  }

  revealWords(all = false) {
    const s = this.sentence;
    if (!s) return;
    const progress = all ? 1 : (performance.now() - s.start) / s.ms;
    const due = [];
    while (s.next < s.words.length && s.ends[s.next] - 0.5 / s.words.length <= progress) due.push(s.words[s.next++]);
    if (due.length) {
      this.bot ??= this.addBot();
      this.bot.add(due.join(" "));
    }
    if (s.next >= s.words.length) {
      this.sentence = null;
      clearInterval(this.revealTimer);
    }
  }

  addBot() {
    const continued = this.lastYou?.overlap && this.lastYou.el === this.chat.lastElementChild;
    const el = this.bubble("bot");
    if (continued) el.classList.add("cont");
    el.innerHTML = `<span class="text"><i class="dots"><b></b><b></b><b></b></i></span>`;
    const text = el.querySelector(".text");
    let words = "";
    const startedAt = performance.now();
    const bubble = {
      el,
      empty: () => !words,
      add: (more) => {
        words = `${words} ${more}`.trim();
        text.textContent = words;
        this.scroll();
      },
      cut: () => {
        if (el.querySelector(".cut")) return;
        if (!words && performance.now() - startedAt < 600) return el.remove();   // barely started: nothing to show
        if (!words) text.textContent = "…";
        el.append(Object.assign(document.createElement("span"), { className: "cut", textContent: "cut off" }));
        this.scroll();
      },
    };
    this.lastBot = bubble;
    return bubble;
  }

  addYou(status) {
    const el = this.bubble("you");
    el.innerHTML = `<span class="text"><i class="dots"><b></b><b></b><b></b></i></span>`;
    const you = { el, text: el.querySelector(".text"), words: "", status, overlap: status === "pending", sealed: false, createdAt: performance.now() };
    if (this.lastYou) this.lastYou.sealed = true;
    this.lastYou = you;
    return you;
  }

  bubble(kind) {
    const el = document.createElement("div");
    el.className = `msg ${kind}`;
    this.chat.append(el);
    this.scroll();
    return el;
  }

  attach(text, final) {
    if (!text.trim()) return;
    const unfinished = final && this.lastYou?.partial && !this.lastYou.words ? this.lastYou : null;   // its final can land after the bot restarts
    const you = this.you || this.openYou() || unfinished || this.openTemporarily(this.addYou("turn"));
    you.partial = !final;
    if (final) you.words = `${you.words} ${text}`.trim();
    you.text.textContent = final ? you.words : `${you.words} ${text}`.trim();
    you.text.classList.toggle("pending", !final);
    this.scroll();
  }

  resolve(you, status, force = false) {
    if (you.status === status || (you.status !== "pending" && !force)) return;
    if (you.status === "kept" || you.status === "stopped") this.counts[you.status]--;
    you.status = status;
    this.counts[status]++;
    you.el.classList.remove("kept", "stopped");
    you.el.classList.add(status);
    if (!you.words) {
      you.text.textContent = "(sound)";
      you.text.classList.add("pending");
    }
    you.el.querySelector(".tag")?.remove();
    you.el.append(Object.assign(document.createElement("span"), {
      className: "tag",
      textContent: status === "stopped" ? "stopped the bot" : "bot kept talking",
    }));
    this.renderTally();
    this.scroll();
  }

  flash() {
    this.el.classList.remove("halt");
    void this.el.offsetWidth;
    this.el.classList.add("halt");
    clearTimeout(this.haltTimer);
    this.haltTimer = setTimeout(() => this.el.classList.remove("halt"), 1400);
  }

  scroll() {
    this.chat.scrollTo({ top: this.chat.scrollHeight, behavior: "smooth" });
  }

  setState(text) {
    this.state.textContent = text;
  }

  renderTally() {
    const before = [...this.tally.querySelectorAll("b")].map((b) => b.textContent);
    this.tally.innerHTML = `<span>Stopped <b>${this.counts.stopped}</b></span><span>Kept talking <b>${this.counts.kept}</b></span>`;
    this.tally.querySelectorAll("b").forEach((b, i) => before.length && b.textContent !== before[i] && b.classList.add("bump"));
  }

  setMeter(p) {
    if (!this.fill) return;
    this.fill.style.width = `${Math.round(p * 100)}%`;
    this.fill.classList.toggle("over", p >= threshold);
    this.meter.classList.toggle("idle", !this.botSpeaking);
  }

  animate() {
    const tick = () => {
      this.orb.style.setProperty("--level", this.botLevel?.() ?? 0);
      this.orb.style.setProperty("--user", this.micLevel?.() ?? 0);
      this.frame = requestAnimationFrame(tick);
    };
    tick();
  }
}

function levelMeter(ctx, stream) {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const data = new Float32Array(analyser.fftSize);
  return () => {
    analyser.getFloatTimeDomainData(data);
    const rms = Math.sqrt(data.reduce((s, x) => s + x * x, 0) / data.length);
    return Math.min(1, Math.max(0, (20 * Math.log10(rms + 1e-9) + 55) / 40));
  };
}

function iceGathered(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    pc.addEventListener("icegatheringstatechange", () => pc.iceGatheringState === "complete" && resolve());
    setTimeout(resolve, 2000);
  });
}

fetch("/api/config")
  .then((r) => r.json())
  .then((c) => {
    threshold = c.threshold;
    document.querySelectorAll(".tick").forEach((t) => (t.style.left = `${threshold * 100}%`));
  });
document.querySelectorAll(".side").forEach((el) => new Side(el));
