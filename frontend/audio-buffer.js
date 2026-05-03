/**
 * Voice Activity Detector — accumulates PCM chunks, detects silence,
 * and returns complete utterances for transcription.
 *
 * Based on the road-rescue AudioBuffer spec:
 *   - Silence判定: volume_threshold_db + silence_duration_ms
 *   - 靜音段保留在尾部，確保語音完整
 *   - getBufferedAudio() 同時清空緩衝
 */
class VoiceActivityDetector {
    /**
     * @param {object} opts
     * @param {number} opts.silenceDurationMs  700  — ms of silence before cutting
     * @param {number} opts.volumeThresholdDb  -35  — dBFS below which = silence
     * @param {number} opts.sampleRate              — from AudioContext.sampleRate
     * @param {number} opts.maxBufferSeconds   20   — safety cap
     */
    constructor({
        silenceDurationMs = 700,
        volumeThresholdDb = -35,
        sampleRate = 44100,
        maxBufferSeconds = 20,
    } = {}) {
        this.silenceDurationMs = silenceDurationMs;
        this.volumeThresholdDb = volumeThresholdDb;
        this.sampleRate = sampleRate;
        this.maxBufferSamples = sampleRate * maxBufferSeconds;

        this._chunks = [];        // Float32Array[]
        this._totalSamples = 0;
        this._silenceStart = null;
        this._lastDb = -Infinity;
    }

    /**
     * Feed one audio frame. Call on every ScriptProcessor tick.
     * @param {Float32Array} float32Data — from e.inputBuffer.getChannelData(0)
     * @returns {number} current volume in dBFS (for VU meter)
     */
    addChunk(float32Data) {
        const db = this._toDb(float32Data);
        this._lastDb = db;
        const isSilence = db < this.volumeThresholdDb;

        if (!isSilence) {
            this._pushChunk(float32Data);
            this._silenceStart = null;
        } else {
            // Include trailing silence so the cut-point sounds natural
            if (this._chunks.length > 0) {
                this._pushChunk(float32Data);
                if (this._silenceStart === null) {
                    this._silenceStart = Date.now();
                }
            }
        }

        return db;
    }

    /**
     * Returns a merged Float32Array when silence_duration_ms has elapsed,
     * then resets the buffer. Returns null otherwise.
     * @returns {Float32Array|null}
     */
    getBufferedAudio() {
        if (this._chunks.length === 0 || this._silenceStart === null) return null;
        if (Date.now() - this._silenceStart < this.silenceDurationMs) return null;

        const merged = this._merge();
        this._reset();
        return merged;
    }

    /** True when speech frames are accumulating (useful for "interrupt" check). */
    isNewAudio() {
        return this._chunks.length > 0 && this._silenceStart !== null;
    }

    /** Force-flush whatever is buffered — used when stopping. */
    flushAndReset() {
        const merged = this._chunks.length > 0 ? this._merge() : null;
        this._reset();
        return merged;
    }

    get lastDb() { return this._lastDb; }

    // ──────────────────────────────────────────────────────
    _pushChunk(float32Data) {
        const copy = new Float32Array(float32Data);
        this._chunks.push(copy);
        this._totalSamples += copy.length;

        // Safety cap — drop oldest chunk if buffer overflows
        while (this._totalSamples > this.maxBufferSamples && this._chunks.length > 1) {
            this._totalSamples -= this._chunks.shift().length;
        }
    }

    _merge() {
        const out = new Float32Array(this._totalSamples);
        let offset = 0;
        for (const c of this._chunks) { out.set(c, offset); offset += c.length; }
        return out;
    }

    _reset() {
        this._chunks = [];
        this._totalSamples = 0;
        this._silenceStart = null;
    }

    _toDb(float32Data) {
        let sum = 0;
        for (let i = 0; i < float32Data.length; i++) sum += float32Data[i] ** 2;
        const rms = Math.sqrt(sum / float32Data.length);
        return rms === 0 ? -Infinity : 20 * Math.log10(rms);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WAV encoding
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Encode Float32 PCM → WAV ArrayBuffer (16-bit, mono).
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @returns {ArrayBuffer}
 */
function float32ToWav(samples, sampleRate) {
    const dataSize = samples.length * 2; // 16-bit = 2 bytes/sample
    const buf = new ArrayBuffer(44 + dataSize);
    const v = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };

    str(0, 'RIFF');  v.setUint32(4, 36 + dataSize, true);
    str(8, 'WAVE');  str(12, 'fmt ');
    v.setUint32(16, 16, true);           // PCM chunk size
    v.setUint16(20, 1, true);            // PCM format
    v.setUint16(22, 1, true);            // mono
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true); // byte rate
    v.setUint16(32, 2, true);            // block align
    v.setUint16(34, 16, true);           // bits per sample
    str(36, 'data'); v.setUint32(40, dataSize, true);

    let off = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        off += 2;
    }
    return buf;
}
