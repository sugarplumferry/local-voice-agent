/**
 * Local Voice Agent — main app logic (AudioWorklet + WebSocket edition).
 *
 * Per-utterance flow:
 *   getUserMedia → AudioWorkletNode (VADProcessor)
 *     ↓ {type:"utterance", pcm: Float32Buffer}
 *   float32ToWav() → WebSocket.send(binary WAV)
 *     ↓ {type:"transcript_update"} … {type:"token"} … {type:"audio"} …
 *   update UI + play TTS
 *
 * Interrupt: if the worklet reports 6+ consecutive ticks above threshold
 * while TTS is playing, audio is stopped immediately (≈174 ms of speech).
 */

const _wsProto      = location.protocol === "https:" ? "wss:" : "ws:";
const BACKEND_WS    = `${_wsProto}//${location.host}/ws`;   // same origin, nginx proxies /ws → backend
const SESSION_ID    = crypto.randomUUID();
const VU_BARS       = 24;
const INTERRUPT_TICKS = 6;   // × ~29 ms per tick ≈ 174 ms sustained speech

// ── State ─────────────────────────────────────────────────────────────────────

let audioCtx        = null;
let workletNode     = null;
let stream          = null;
let ws              = null;
let isListening     = false;
let exceedCount     = 0;
let currentThresholdDb = -25;

/** Accumulated Float32 PCM for the current speaking turn.
 *  Reset to null when the turn completes (done event) or recording stops. */
let turnPcm = null;

/** Currently playing { source: AudioBufferSourceNode, ctx: AudioContext } */
let currentSource   = null;
/** Ordered queue of base64 WAV chunks waiting to play. */
const audioQueue    = [];

// ── DOM refs ──────────────────────────────────────────────────────────────────

const btnStart        = document.getElementById("btn-start");
const statusEl        = document.getElementById("status");
const conversationEl  = document.getElementById("conversation");
const vuMeterEl       = document.getElementById("vu-meter");
const slSilence       = document.getElementById("sl-silence");
const lblSilence      = document.getElementById("lbl-silence");
const lblThreshold    = document.getElementById("lbl-threshold");
const threshMarkerEl  = document.getElementById("threshold-marker");

// ── VU meter init ─────────────────────────────────────────────────────────────

for (let i = 0; i < VU_BARS; i++) {
    const b = document.createElement("div");
    b.className = "vu-bar";
    vuMeterEl.insertBefore(b, threshMarkerEl);  // bars go before the marker
}
const vuBars = Array.from(vuMeterEl.querySelectorAll(".vu-bar"));

function _dbFraction(db) {
    return Math.max(0, Math.min(1, (db + 60) / 60));
}

function updateVu(db) {
    const active     = Math.round(_dbFraction(db) * VU_BARS);
    const threshBar  = Math.round(_dbFraction(currentThresholdDb) * VU_BARS);
    vuBars.forEach((b, i) => {
        if (i >= active)        b.style.background = "var(--meter-off)";
        else if (i >= threshBar) b.style.background = "var(--meter-voice)"; // above threshold → green
        else                    b.style.background = "var(--meter-on)";     // below threshold → blue
    });
}

function _positionMarker() {
    const pct = _dbFraction(currentThresholdDb) * 100;
    threshMarkerEl.style.left = `${pct}%`;
    lblThreshold.textContent  = `${Math.round(currentThresholdDb)}dB`;
}

// ── Threshold drag on VU meter ────────────────────────────────────────────────

function _applyThresholdFromX(clientX) {
    const rect   = vuMeterEl.getBoundingClientRect();
    const frac   = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    currentThresholdDb = Math.round(frac * 60 - 60);          // map 0–1 → -60–0
    currentThresholdDb = Math.max(-60, Math.min(-10, currentThresholdDb));
    workletNode?.port.postMessage({ type: "config", thresholdDb: currentThresholdDb });
    _positionMarker();
}

let _dragging = false;

vuMeterEl.addEventListener("mousedown", e => {
    _dragging = true;
    vuMeterEl.classList.add("dragging");
    _applyThresholdFromX(e.clientX);
});
document.addEventListener("mousemove", e => {
    if (_dragging) _applyThresholdFromX(e.clientX);
});
document.addEventListener("mouseup", () => {
    _dragging = false;
    vuMeterEl.classList.remove("dragging");
});

vuMeterEl.addEventListener("touchstart", e => {
    _dragging = true;
    vuMeterEl.classList.add("dragging");
    _applyThresholdFromX(e.touches[0].clientX);
}, { passive: true });
document.addEventListener("touchmove", e => {
    if (_dragging) { e.preventDefault(); _applyThresholdFromX(e.touches[0].clientX); }
}, { passive: false });
document.addEventListener("touchend", () => {
    _dragging = false;
    vuMeterEl.classList.remove("dragging");
});

_positionMarker();  // set initial marker position

// ── Silence slider ────────────────────────────────────────────────────────────

slSilence.addEventListener("input", () => {
    lblSilence.textContent = `${slSilence.value}ms`;
    workletNode?.port.postMessage({ type: "config", silenceDurationMs: Number(slSilence.value) });
});

// ── Button ────────────────────────────────────────────────────────────────────

btnStart.addEventListener("click", () => {
    if (isListening) stopRecording();
    else startRecording();
});

// ── Recording ─────────────────────────────────────────────────────────────────

async function startRecording() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (err) {
        setStatus(`Mic denied: ${err.message}`);
        return;
    }

    audioCtx = new AudioContext();
    // iOS suspends AudioContext until resumed inside a user-gesture handler
    if (audioCtx.state === "suspended") await audioCtx.resume();

    try {
        await audioCtx.audioWorklet.addModule("vad-processor.js");
    } catch (err) {
        setStatus(`Worklet load failed: ${err.message}`);
        stream.getTracks().forEach(t => t.stop());
        audioCtx.close();
        stream = audioCtx = null;
        return;
    }

    workletNode = new AudioWorkletNode(audioCtx, "vad-processor", {
        processorOptions: {
            volumeThresholdDb: currentThresholdDb,
            silenceDurationMs: Number(slSilence.value),
        },
    });
    workletNode.port.onmessage = onWorkletMessage;

    audioCtx.createMediaStreamSource(stream).connect(workletNode);
    // workletNode intentionally not connected to destination (no loopback)

    connectWS();

    isListening = true;
    btnStart.classList.add("listening");
    btnStart.textContent = "⏹";
    btnStart.setAttribute("aria-label", "Stop listening");
    setStatus("Connecting…");
}

function stopRecording() {
    workletNode?.disconnect();
    workletNode = null;

    stream?.getTracks().forEach(t => t.stop());
    stream = null;

    audioCtx?.close();
    audioCtx = null;

    ws?.close();
    ws = null;

    isListening    = false;
    exceedCount    = 0;
    activeUserEl   = null;
    activeAgentEl  = null;
    turnPcm        = null;

    btnStart.classList.remove("listening");
    btnStart.textContent = "🎙";
    btnStart.setAttribute("aria-label", "Start listening");
    setStatus("Ready");
    updateVu(-Infinity);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
    ws = new WebSocket(BACKEND_WS);

    ws.onopen = () => {
        ws.send(JSON.stringify({ type: "init_session", session_id: SESSION_ID }));
        setStatus("Listening…");
    };

    ws.onmessage = ({ data }) => onWSMessage(JSON.parse(data));

    ws.onclose = () => {
        if (isListening) setStatus("Disconnected — reload to reconnect");
    };

    ws.onerror = () => setStatus("WebSocket error");
}

// ── Worklet messages ──────────────────────────────────────────────────────────

let activeUserEl  = null;   // currently updating user bubble
let activeAgentEl = null;   // currently updating agent bubble

function onWorkletMessage({ data }) {
    if (data.type === "db") {
        updateVu(data.value);

        // Interrupt TTS if user sustains speech for INTERRUPT_TICKS ticks
        if (data.value > currentThresholdDb) {
            exceedCount++;
            if (exceedCount >= INTERRUPT_TICKS && currentSource) {
                stopCurrentAudio(); // also clears audioQueue
            }
        } else {
            exceedCount = 0;
        }

    } else if (data.type === "utterance") {
        exceedCount = 0;
        const float32 = new Float32Array(data.pcm);
        if (float32.length === 0) return;

        if (activeUserEl) {
            // Same turn — user paused briefly and continued speaking.
            // Concatenate with everything said so far in this turn.
            const prev = turnPcm ?? new Float32Array(0);
            const combined = new Float32Array(prev.length + float32.length);
            combined.set(prev);
            combined.set(float32, prev.length);
            turnPcm = combined;

            // Reset agent bubble so stale tokens don't mix with the new response
            if (activeAgentEl) {
                const body = activeAgentEl.querySelector(".body");
                body.textContent = "";
                body.classList.remove("cursor");
            }
        } else {
            // Fresh turn
            turnPcm       = float32;
            activeUserEl  = appendMsg("user", "…");
            activeAgentEl = null;
        }

        // Always send the full accumulated audio for this turn so the backend
        // gets the complete utterance and can cancel + redo transcription.
        const wavBuf = float32ToWav(turnPcm, audioCtx.sampleRate);
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(wavBuf);
        }

        setStatus("Transcribing…");
    }
}

// ── WebSocket messages ────────────────────────────────────────────────────────

function onWSMessage(msg) {
    switch (msg.type) {

        case "transcript_update":
            // Progressive segment update — refine placeholder while whisper works
            if (activeUserEl) {
                activeUserEl.querySelector(".body").textContent = msg.text;
            }
            break;

        case "transcript":
            // Final confirmed transcript — ensure bubble matches
            if (activeUserEl) {
                activeUserEl.querySelector(".body").textContent = msg.text;
            } else {
                activeUserEl = appendMsg("user", msg.text);
            }
            if (!activeAgentEl) {
                activeAgentEl = appendMsg("agent", "", true);
                setStatus("Thinking…");
            }
            break;

        case "token": {
            const bodyEl = activeAgentEl?.querySelector(".body");
            if (bodyEl) {
                bodyEl.textContent += msg.content;
                bodyEl.classList.add("cursor");
                scrollBottom();
            }
            break;
        }

        case "feedback":
            appendMsg("feedback", msg.content);
            break;

        case "audio":
            enqueueAudio(msg.content);
            break;

        case "done":
            activeAgentEl?.querySelector(".body").classList.remove("cursor");
            activeUserEl  = null;
            activeAgentEl = null;
            turnPcm       = null;   // turn complete — next speech starts fresh
            setStatus(isListening ? "Listening…" : "Ready");
            break;

        case "error":
            setStatus(`Error: ${msg.message}`);
            break;
    }
}

// ── Audio playback ────────────────────────────────────────────────────────────

function enqueueAudio(b64wav) {
    audioQueue.push(b64wav);
    if (!currentSource) _playNextInQueue();
}

function _playNextInQueue() {
    if (audioQueue.length === 0) return;
    playBase64Audio(audioQueue.shift());
}

async function playBase64Audio(b64wav) {
    const bytes = Uint8Array.from(atob(b64wav), c => c.charCodeAt(0));
    const ctx   = new AudioContext();
    let decoded;
    try {
        decoded = await ctx.decodeAudioData(bytes.buffer);
    } catch {
        ctx.close();
        _playNextInQueue();
        return;
    }

    const src   = ctx.createBufferSource();
    src.buffer  = decoded;
    src.connect(ctx.destination);
    currentSource = { source: src, ctx };

    src.onended = () => {
        ctx.close();
        currentSource = null;
        _playNextInQueue();
    };
    src.start();
}

function stopCurrentAudio() {
    if (!currentSource) return;
    try { currentSource.source.stop(); } catch {}
    try { currentSource.ctx.close();   } catch {}
    currentSource = null;
    audioQueue.length = 0;  // discard queued chunks on interrupt
}

// ── Conversation helpers ──────────────────────────────────────────────────────

function appendMsg(role, text, withCursor = false) {
    const div   = document.createElement("div");
    div.className = `msg ${role}`;

    const label = document.createElement("div");
    label.className   = "label";
    label.textContent = role === "user" ? "You" : role === "feedback" ? "Feedback" : "Agent";
    div.appendChild(label);

    const body = document.createElement("div");
    body.className   = `body${withCursor ? " cursor" : ""}`;
    body.textContent = text;
    div.appendChild(body);

    conversationEl.appendChild(div);
    scrollBottom();
    return div;
}

function scrollBottom() {
    const last = conversationEl.lastElementChild;
    if (last) last.scrollIntoView({ block: "end", behavior: "instant" });
}

function setStatus(msg) { statusEl.textContent = msg; }
