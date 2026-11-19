/**
 * AudioWorklet processor — Voice Activity Detection.
 *
 * Runs in the dedicated audio rendering thread (no main-thread blocking).
 * Posts two message types to the main thread:
 *   { type: "db",        value: number }          — once per analysis window (~20 ms)
 *   { type: "utterance", pcm: ArrayBuffer }        — transferable Float32 PCM when utterance ends
 *
 * The VU meter and the VAD decision use the SAME number: RMS dB over a ~20 ms
 * window. (Previously the VAD fired on any single 128-sample frame (~2.7 ms)
 * above threshold while the meter only showed every 10th frame, so clicks and
 * breath spikes triggered recording without ever appearing on the meter.)
 *
 * Speech onset additionally requires ONSET_WINDOWS consecutive loud windows,
 * so a single transient can't start an utterance.
 *
 * Config can be updated at runtime:
 *   port.postMessage({ type: "config", thresholdDb: -35, silenceDurationMs: 700 })
 */

const WINDOW_SEC    = 0.02;  // analysis window length
const ONSET_WINDOWS = 2;     // consecutive loud windows (~40 ms) needed to start speech

class VADProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        const o = options.processorOptions ?? {};
        this._thresholdDb     = o.volumeThresholdDb  ?? -25;
        this._silenceSec      = (o.silenceDurationMs ?? 300) / 1000;
        this._maxSamples      = sampleRate * (o.maxBufferSeconds ?? 20);
        this._windowSamples   = Math.max(128, Math.round(sampleRate * WINDOW_SEC));

        this._chunks       = [];
        this._totalSamples = 0;
        this._silenceT     = null;  // currentTime when silence began
        this._hasVoice     = false;
        this._onset        = 0;     // consecutive loud windows while not yet in speech

        // Current analysis window
        this._win        = [];
        this._winSamples = 0;
        this._winSumSq   = 0;

        this.port.onmessage = ({ data }) => {
            if (data.type !== "config") return;
            if (data.thresholdDb      != null) this._thresholdDb = data.thresholdDb;
            if (data.silenceDurationMs != null) this._silenceSec  = data.silenceDurationMs / 1000;
        };
    }

    process(inputs) {
        const ch = inputs[0]?.[0];
        if (!ch) return true;

        const copy = new Float32Array(ch);
        this._win.push(copy);
        this._winSamples += copy.length;
        for (let i = 0; i < copy.length; i++) this._winSumSq += copy[i] * copy[i];

        if (this._winSamples >= this._windowSamples) this._endWindow();
        return true;
    }

    _endWindow() {
        const rms = Math.sqrt(this._winSumSq / this._winSamples);
        const db  = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
        const frames = this._win;
        this._win = []; this._winSamples = 0; this._winSumSq = 0;

        this.port.postMessage({ type: "db", value: db });

        const loud = db >= this._thresholdDb;

        if (!this._hasVoice) {
            if (loud) {
                // Candidate onset — keep the audio so the first syllable isn't clipped
                frames.forEach(f => this._push(f));
                if (++this._onset >= ONSET_WINDOWS) {
                    this._hasVoice = true;
                    this._silenceT = null;
                }
            } else if (this._onset > 0) {
                // Transient that didn't last — discard it
                this._reset();
            }
            return;
        }

        // In speech: keep everything, including trailing silence for a natural cut
        frames.forEach(f => this._push(f));
        if (loud) {
            this._silenceT = null;
        } else if (this._silenceT === null) {
            this._silenceT = currentTime;  // AudioWorkletGlobalScope global
        } else if (currentTime - this._silenceT >= this._silenceSec) {
            const pcm = this._flush();  // transferable Float32 buffer
            this.port.postMessage({ type: "utterance", pcm }, [pcm]);
            this._reset();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    _push(float32) {
        this._chunks.push(float32);
        this._totalSamples += float32.length;
        // Safety cap — drop oldest chunk on overflow
        while (this._totalSamples > this._maxSamples && this._chunks.length > 1) {
            this._totalSamples -= this._chunks.shift().length;
        }
    }

    _flush() {
        const out = new Float32Array(this._totalSamples);
        let off = 0;
        for (const c of this._chunks) { out.set(c, off); off += c.length; }
        return out.buffer;  // caller transfers this
    }

    _reset() {
        this._chunks       = [];
        this._totalSamples = 0;
        this._silenceT     = null;
        this._hasVoice     = false;
        this._onset        = 0;
    }
}

registerProcessor("vad-processor", VADProcessor);
