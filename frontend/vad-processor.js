/**
 * AudioWorklet processor — Voice Activity Detection.
 *
 * Runs in the dedicated audio rendering thread (no main-thread blocking).
 * Posts two message types to the main thread:
 *   { type: "db",        value: number }          — throttled, every 10 frames (~29ms)
 *   { type: "utterance", pcm: ArrayBuffer }        — transferable Float32 PCM when utterance ends
 *
 * Config can be updated at runtime:
 *   port.postMessage({ type: "config", thresholdDb: -35, silenceDurationMs: 700 })
 */
class VADProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        const o = options.processorOptions ?? {};
        this._thresholdDb     = o.volumeThresholdDb  ?? -25;
        this._silenceSec      = (o.silenceDurationMs ?? 300) / 1000;
        this._maxSamples      = sampleRate * (o.maxBufferSeconds ?? 20);

        this._chunks      = [];
        this._totalSamples = 0;
        this._silenceT     = null;  // currentTime when silence began
        this._hasVoice     = false;
        this._frameN       = 0;

        this.port.onmessage = ({ data }) => {
            if (data.type !== "config") return;
            if (data.thresholdDb      != null) this._thresholdDb = data.thresholdDb;
            if (data.silenceDurationMs != null) this._silenceSec  = data.silenceDurationMs / 1000;
        };
    }

    process(inputs) {
        const ch = inputs[0]?.[0];
        if (!ch) return true;

        const db        = this._rmsDb(ch);
        const isSilence = db < this._thresholdDb;

        // Throttle VU meter messages — every 10 frames ≈ 29 ms at 44100 Hz
        if (++this._frameN % 10 === 0) {
            this.port.postMessage({ type: "db", value: db });
        }

        if (!isSilence) {
            this._push(ch);
            this._silenceT = null;
            this._hasVoice = true;
        } else if (this._hasVoice) {
            // Keep trailing silence so the audio cut-point sounds natural
            this._push(ch);

            if (this._silenceT === null) {
                this._silenceT = currentTime;  // AudioWorkletGlobalScope global
            } else if (currentTime - this._silenceT >= this._silenceSec) {
                const pcm = this._flush();  // transferable Float32 buffer
                this.port.postMessage({ type: "utterance", pcm }, [pcm]);
                this._reset();
            }
        }

        return true;
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    _rmsDb(float32) {
        let sum = 0;
        for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
        const rms = Math.sqrt(sum / float32.length);
        return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
    }

    _push(float32) {
        this._chunks.push(new Float32Array(float32));
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
    }
}

registerProcessor("vad-processor", VADProcessor);
