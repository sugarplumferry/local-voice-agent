/**
 * Local Voice Agent — main app logic (AudioWorklet + WebSocket edition).
 *
 * Transcription display uses a live-caption pattern:
 *   transcript_update → stream partial text into a dimmed preview span (extend/correct)
 *   transcript        → un-dim if preview matches, or retype corrected final text
 */

const _wsProto      = location.protocol === "https:" ? "wss:" : "ws:";
const BACKEND_WS    = `${_wsProto}//${location.host}/ws`;
const SESSION_ID    = typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
        const r = crypto.getRandomValues(new Uint8Array(1))[0] & 15;
        return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
    });
const VU_BARS       = 24;
const INTERRUPT_TICKS = 6;   // × ~29 ms per tick ≈ 174 ms sustained speech

// ── State ─────────────────────────────────────────────────────────────────────

let audioCtx           = null;
let workletNode        = null;
let stream             = null;
let ws                 = null;
let isListening        = false;
let exceedCount        = 0;
let currentThresholdDb = -25;
let turnPcm            = null;
let currentSource      = null;
const audioQueue       = [];
let _audioInterruptedAt = null;  // set by stopCurrentAudio() to block echo false-positives

// ── DOM refs ──────────────────────────────────────────────────────────────────

const btnStart       = document.getElementById("btn-start");
const statusEl       = document.getElementById("status");
const conversationEl = document.getElementById("conversation");
const vuMeterEl      = document.getElementById("vu-meter");
const slSilence      = document.getElementById("sl-silence");
const lblSilence     = document.getElementById("lbl-silence");
const lblThreshold   = document.getElementById("lbl-threshold");
const threshMarkerEl = document.getElementById("threshold-marker");

// ── TypeAnimator ──────────────────────────────────────────────────────────────
// Streams text into the DOM at ~50 cps / 25 fps.
// add(text, writeFn, instant) — writeFn(chunk) receives each chunk to write.
// Callers close over their target element so the animator stays DOM-agnostic.

function makeTypeAnimator(cps = 50, fps = 25) {
    const CHUNK = Math.max(1, Math.round(cps / fps));
    const FRAME = 1000 / fps;
    let queue = [];   // [{text, writeFn, pos}]
    let raf   = null;
    let last  = 0;

    function tick(now) {
        raf = null;
        if (!queue.length) return;
        if (now - last < FRAME) { raf = requestAnimationFrame(tick); return; }
        last = now;
        const item = queue[0];
        const end  = Math.min(item.pos + CHUNK, item.text.length);
        item.writeFn(item.text.slice(item.pos, end));
        item.pos = end;
        if (item.pos >= item.text.length) queue.shift();
        if (queue.length) raf = requestAnimationFrame(tick);
    }

    return {
        add(text, writeFn, instant = false) {
            if (!text) return;
            if (instant) { writeFn(text); return; }
            queue.push({ text, writeFn, pos: 0 });
            if (!raf) raf = requestAnimationFrame(tick);
        },
        clearQueue() {
            queue = [];
            if (raf) { cancelAnimationFrame(raf); raf = null; }
        },
        flush() {
            const q = queue;
            queue = [];
            if (raf) { cancelAnimationFrame(raf); raf = null; }
            q.forEach(({ text, writeFn, pos }) => writeFn(text.slice(pos)));
        },
    };
}

const typer = makeTypeAnimator();

// ── Live-region state ─────────────────────────────────────────────────────────
// Mirrors the Tkinter "live_start mark + _live_active/_live_text" pattern.

let _liveActive = false;   // preview span is open
let _liveText   = "";      // text currently shown in the preview span
let _liveSpan   = null;    // <span class="preview"> anchored inside .body

function _resetLive() {
    _liveActive = false;
    _liveText   = "";
    _liveSpan   = null;
}

// ── VU meter init ─────────────────────────────────────────────────────────────

for (let i = 0; i < VU_BARS; i++) {
    const b = document.createElement("div");
    b.className = "vu-bar";
    vuMeterEl.insertBefore(b, threshMarkerEl);
}
const vuBars = Array.from(vuMeterEl.querySelectorAll(".vu-bar"));

function _dbFraction(db) {
    return Math.max(0, Math.min(1, (db + 60) / 60));
}

function updateVu(db) {
    const active    = Math.round(_dbFraction(db) * VU_BARS);
    const threshBar = Math.round(_dbFraction(currentThresholdDb) * VU_BARS);
    vuBars.forEach((b, i) => {
        if (i >= active)         b.style.background = "var(--meter-off)";
        else if (i >= threshBar) b.style.background = "var(--meter-voice)";
        else                     b.style.background = "var(--meter-on)";
    });
}

function _positionMarker() {
    threshMarkerEl.style.left = `${_dbFraction(currentThresholdDb) * 100}%`;
    lblThreshold.textContent  = `${Math.round(currentThresholdDb)}dB`;
}

// ── Threshold drag on VU meter ────────────────────────────────────────────────

function _applyThresholdFromX(clientX) {
    const rect = vuMeterEl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    currentThresholdDb = Math.max(-60, Math.min(-10, Math.round(frac * 60 - 60)));
    workletNode?.port.postMessage({ type: "config", thresholdDb: currentThresholdDb });
    _positionMarker();
}

let _dragging = false;

vuMeterEl.addEventListener("mousedown", e => {
    _dragging = true; vuMeterEl.classList.add("dragging"); _applyThresholdFromX(e.clientX);
});
document.addEventListener("mousemove",  e => { if (_dragging) _applyThresholdFromX(e.clientX); });
document.addEventListener("mouseup",    ()  => { _dragging = false; vuMeterEl.classList.remove("dragging"); });
vuMeterEl.addEventListener("touchstart", e => {
    _dragging = true; vuMeterEl.classList.add("dragging"); _applyThresholdFromX(e.touches[0].clientX);
}, { passive: true });
document.addEventListener("touchmove", e => {
    if (_dragging) { e.preventDefault(); _applyThresholdFromX(e.touches[0].clientX); }
}, { passive: false });
document.addEventListener("touchend", () => { _dragging = false; vuMeterEl.classList.remove("dragging"); });

_positionMarker();

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

    connectWS();

    isListening = true;
    btnStart.classList.add("listening");
    btnStart.textContent = "⏹";
    btnStart.setAttribute("aria-label", "Stop listening");
    setStatus("Connecting…");
}

function stopRecording() {
    typer.clearQueue();
    _resetLive();

    workletNode?.disconnect();
    workletNode = null;
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
    audioCtx?.close();
    audioCtx = null;
    ws?.close();
    ws = null;

    isListening   = false;
    exceedCount   = 0;
    activeUserEl  = null;
    activeAgentEl = null;
    turnPcm       = null;

    btnStart.classList.remove("listening");
    btnStart.textContent = "🎙";
    btnStart.setAttribute("aria-label", "Start listening");
    setStatus("Ready");
    updateVu(-Infinity);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
    ws = new WebSocket(BACKEND_WS);
    ws.onopen  = () => {
        console.log(`[WS] connected  session=${SESSION_ID}`);
        ws.send(JSON.stringify({ type: "init_session", session_id: SESSION_ID }));
        setStatus("Listening…");
    };
    ws.onmessage = ({ data }) => onWSMessage(JSON.parse(data));
    ws.onclose   = () => {
        console.log("[WS] closed");
        if (isListening) setStatus("Disconnected — reload to reconnect");
    };
    ws.onerror   = (e) => { console.error("[WS] error", e); setStatus("WebSocket error"); };
}

// ── Worklet messages ──────────────────────────────────────────────────────────

let activeUserEl  = null;
let activeAgentEl = null;

function onWorkletMessage({ data }) {
    if (data.type === "db") {
        updateVu(data.value);
        if (data.value > currentThresholdDb) {
            exceedCount++;
            if (exceedCount >= INTERRUPT_TICKS && currentSource) stopCurrentAudio();
        } else {
            exceedCount = 0;
        }

    } else if (data.type === "utterance") {
        exceedCount = 0;
        const float32 = new Float32Array(data.pcm);
        if (float32.length === 0) return;

        // Discard utterances that fire shortly after an audio interrupt —
        // the VAD almost certainly triggered on the phone speaker output,
        // not genuine user speech.  600 ms covers the echo tail + VAD
        // silence window (174 ms interrupt lag + 300 ms silence timer + buffer).
        if (_audioInterruptedAt !== null) {
            const elapsed = Date.now() - _audioInterruptedAt;
            _audioInterruptedAt = null;
            if (elapsed < 600) {
                console.log(`[VAD] utterance discarded — echo guard (${elapsed} ms after interrupt)`);
                return;
            }
        }

        if (activeUserEl) {
            // Continuing turn — flush any in-progress animation so the body
            // is up-to-date before the backend re-processes the combined audio.
            console.log(`[VAD] continuing turn  +${float32.length} samples  total=${(turnPcm?.length ?? 0) + float32.length}`);
            typer.flush();
            _resetLive();

            const prev = turnPcm ?? new Float32Array(0);
            const combined = new Float32Array(prev.length + float32.length);
            combined.set(prev);
            combined.set(float32, prev.length);
            turnPcm = combined;

            if (activeAgentEl) {
                // Dim the old response but keep the pointer set.
                // The transcript case re-activates this same bubble for the new
                // response, preventing orphaned empty agent bubbles that would
                // appear if we null'd here and the old task's stale "transcript"
                // event arrived before the backend could cancel it.
                activeAgentEl.querySelector(".body").classList.remove("cursor");
                activeAgentEl.classList.add("interrupted");
            }
        } else {
            // Fresh turn — empty body; live-region animation will fill it
            console.log(`[VAD] fresh turn  ${float32.length} samples`);
            turnPcm      = float32;
            activeUserEl = appendMsg("user", "");
            activeAgentEl = null;
        }

        const wavBuf = float32ToWav(turnPcm, audioCtx.sampleRate);
        console.log(`[WS] sending WAV  ${wavBuf.byteLength} bytes`);
        if (ws?.readyState === WebSocket.OPEN) ws.send(wavBuf);
        setStatus("Transcribing…");
    }
}

// ── WebSocket messages ────────────────────────────────────────────────────────

function onWSMessage(msg) {
    if (msg.type !== "token") console.log(`[WS ←] ${msg.type}`, msg.type === "transcript_update" ? msg.text?.slice(0, 60) : msg.type === "transcript" ? msg.text : msg.type === "error" ? msg.message : "");
    switch (msg.type) {

        // ── Phase 1 / 2 / 3: live preview ──────────────────────────────────
        case "transcript_update": {
            const pending = msg.text;
            if (!activeUserEl || !pending) break;

            const bodyEl = activeUserEl.querySelector(".body");

            if (!_liveActive) {
                // Open the live region: clear placeholder, anchor a preview span
                bodyEl.textContent = "";
                _liveSpan = document.createElement("span");
                _liveSpan.className = "preview";
                bodyEl.appendChild(_liveSpan);
                _liveActive = true;
                _liveText   = "";
            }

            if (pending === _liveText) break;  // idempotent

            if (pending.startsWith(_liveText)) {
                // Extension — animate only the newly added tail
                const tail = pending.slice(_liveText.length);
                const span = _liveSpan;
                typer.add(tail, chunk => { span.textContent += chunk; scrollBottom(); });
            } else {
                // Correction — cancel in-flight animation, retype from scratch
                typer.clearQueue();
                _liveSpan.textContent = "";
                const span = _liveSpan;
                typer.add(pending, chunk => { span.textContent += chunk; scrollBottom(); });
            }
            _liveText = pending;
            break;
        }

        // ── Phase 4: finalize ───────────────────────────────────────────────
        case "transcript": {
            const finalText = msg.text;
            typer.flush();   // complete in-flight animation so the span has full text before we inspect it

            if (activeUserEl) {
                const bodyEl = activeUserEl.querySelector(".body");

                if (_liveActive) {
                    if (finalText === _liveText) {
                        // Preview already matches — just un-dim it in place
                        _liveSpan.className = "";
                    } else {
                        // Whisper corrected words — clear preview, retype final
                        bodyEl.textContent = "";
                        typer.add(finalText, chunk => {
                            bodyEl.appendChild(document.createTextNode(chunk));
                            scrollBottom();
                        });
                    }
                } else {
                    // No preview shown (very fast transcription) — write directly
                    bodyEl.textContent = "";
                    typer.add(finalText, chunk => {
                        bodyEl.appendChild(document.createTextNode(chunk));
                        scrollBottom();
                    });
                }
            } else {
                activeUserEl = appendMsg("user", finalText);
            }

            _resetLive();

            if (!activeAgentEl) {
                activeAgentEl = appendMsg("agent", "", true);
                setStatus("Thinking…");
            } else if (activeAgentEl.classList.contains("interrupted")) {
                // Re-activate the dimmed bubble for the incoming response.
                // Clears any partial content left from the interrupted task.
                const body = activeAgentEl.querySelector(".body");
                body.textContent = "";
                body.classList.add("cursor");
                activeAgentEl.classList.remove("interrupted");
                setStatus("Thinking…");
            }
            break;
        }

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
            typer.flush();
            _resetLive();
            activeAgentEl?.querySelector(".body").classList.remove("cursor");
            // Remove empty user bubble — happens when VAD triggered on noise but
            // Whisper returned nothing (backend skips transcript and sends done directly).
            if (activeUserEl && !activeUserEl.querySelector(".body").textContent.trim()) {
                activeUserEl.remove();
                console.log("[UI] removed empty user bubble (empty transcript)");
            }
            activeUserEl  = null;
            activeAgentEl = null;
            turnPcm       = null;
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
    const src = ctx.createBufferSource();
    src.buffer = decoded;
    src.connect(ctx.destination);
    currentSource = { source: src, ctx };
    src.onended = () => { ctx.close(); currentSource = null; _playNextInQueue(); };
    src.start();
}

function stopCurrentAudio() {
    if (!currentSource) return;
    try { currentSource.source.stop(); } catch {}
    try { currentSource.ctx.close();   } catch {}
    currentSource = null;
    audioQueue.length = 0;
    // Mark the interrupt time so the next VAD utterance (likely speaker echo)
    // can be discarded before it reaches the backend.
    _audioInterruptedAt = Date.now();
}

// ── Conversation helpers ──────────────────────────────────────────────────────

function appendMsg(role, text, withCursor = false) {
    const div = document.createElement("div");
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
