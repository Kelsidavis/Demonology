from __future__ import annotations

import io
import os
import re
import wave
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Tool

logger = logging.getLogger(__name__)

# -------- Workspace confinement --------
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()

def _safe_path(path: str, want_dir: bool = False) -> Path:
    if not path:
        raise ValueError("Empty path")
    p = Path(path)
    p = (WORKSPACE_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except Exception:
        raise PermissionError(f"Path escapes workspace root: {p}")
    # disallow symlink traversal (best effort)
    for parent in [p] + list(p.parents):
        try:
            if parent.is_symlink():
                raise PermissionError(f"Symlinked path not allowed: {parent}")
        except FileNotFoundError:
            pass
    if want_dir:
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

# -------- Global audio limits / helpers --------
MAX_DURATION_SEC = float(os.environ.get("DEMONOLOGY_AUDIO_MAX_DURATION", "30.0"))
MAX_SAMPLE_RATE = int(os.environ.get("DEMONOLOGY_AUDIO_MAX_SR", "48000"))
MAX_SPEC_FRAMES = 1000        # cap frames returned in JSON
MAX_SPEC_BINS = 1024          # cap freq bins returned in JSON
MAX_SPECTRUM_BINS = 2048      # cap averaged spectrum bins
EPS = 1e-12

def _validate_sr_duration(sr: int, duration: float) -> Optional[str]:
    if sr <= 0 or sr > MAX_SAMPLE_RATE:
        return f"Sample rate must be 1..{MAX_SAMPLE_RATE} Hz"
    if duration <= 0 or duration > MAX_DURATION_SEC:
        return f"Duration must be >0 and ≤ {MAX_DURATION_SEC} seconds"
    # rough sample cap
    if sr * duration > MAX_SAMPLE_RATE * MAX_DURATION_SEC:
        return "Requested audio exceeds global sample cap"
    return None

def _to_int16(x: np.ndarray) -> np.ndarray:
    # clip then scale [-1,1] → int16
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def _mix_to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1)

def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

def _frame_signal(x: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if len(x) < frame_size:
        pad = np.zeros(frame_size - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad])
    num = 1 + (len(x) - frame_size) // hop if len(x) >= frame_size else 1
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num, frame_size),
        strides=(x.strides[0]*hop, x.strides[0]),
        writeable=False
    )
    return frames

def _welch_spectrum(x: np.ndarray, sr: int, nfft: int = 2048, hop: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum across frames (Welch-like)."""
    hop = hop or (nfft // 2)
    win = _hann(nfft)
    frames = _frame_signal(x, nfft, hop)
    windowed = frames * win
    fft = np.fft.rfft(windowed, n=nfft, axis=1)
    mag = np.abs(fft)
    avg = mag.mean(axis=0)
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)
    return freqs, avg

def _spectrogram(x: np.ndarray, sr: int, nfft: int = 1024, hop: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Log-magnitude spectrogram (dB), truncated for JSON."""
    win = _hann(nfft)
    frames = _frame_signal(x, nfft, hop)
    windowed = frames * win
    fft = np.fft.rfft(windowed, n=nfft, axis=1)
    mag = np.abs(fft) + EPS
    S_db = 20.0 * np.log10(mag)
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)
    return freqs, S_db

def _truncate_1d(arr: np.ndarray, max_len: int) -> Tuple[List[float], bool]:
    if arr.shape[0] > max_len:
        return arr[:max_len].astype(float).tolist(), True
    return arr.astype(float).tolist(), False

def _truncate_2d(mat: np.ndarray, max_rows: int, max_cols: int) -> Tuple[List[List[float]], bool]:
    rows = min(mat.shape[0], max_rows)
    cols = min(mat.shape[1], max_cols)
    trimmed = mat[:rows, :cols]
    return trimmed.astype(float).tolist(), (rows < mat.shape[0] or cols < mat.shape[1])

# -------- WAV I/O (PCM16) --------
def _read_wav_pcm16(path: Path) -> Tuple[np.ndarray, int, int]:
    """Return float32 in [-1,1], sample_rate, channels. Only PCM16 WAV."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV is supported")
        raw = wf.readframes(n_frames)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data, sr, n_channels

def _write_wav_pcm16(path: Path, data: np.ndarray, sr: int) -> None:
    if data.ndim == 1:
        channels = 1
        frames = _to_int16(data)
        interleaved = frames.tobytes()
    elif data.ndim == 2:
        channels = data.shape[1]
        frames = _to_int16(data)
        interleaved = frames.reshape(-1, channels).astype(np.int16).tobytes()
    else:
        raise ValueError("Audio array must be mono [N] or stereo [N,2]")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved)

# ============================================================
# Waveform generator
# ============================================================

class WaveformGeneratorTool(Tool):
    """Generate simple waveforms (sine/square/saw/white_noise) as WAV (PCM16)."""

    def __init__(self):
        super().__init__("waveform_generate", "Generate short waveforms in WAV (PCM16) with safe limits.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "waveform": {
                        "type": "string",
                        "enum": ["sine", "square", "saw", "white_noise"],
                        "description": "Waveform type"
                    },
                    "frequency": {"type": "number", "description": "Hz (ignored for white_noise)", "default": 440.0},
                    "amplitude": {"type": "number", "description": "0..1", "default": 0.5},
                    "duration": {"type": "number", "description": f"Seconds (≤ {MAX_DURATION_SEC})", "default": 2.0},
                    "sample_rate": {"type": "integer", "description": f"Hz (≤ {MAX_SAMPLE_RATE})", "default": 44100},
                    "output_file": {"type": "string", "description": "Relative path under workspace"},
                    "stereo": {"type": "boolean", "description": "Duplicate mono to L/R", "default": False},
                },
                "required": ["waveform", "output_file"]
            }
        }

    async def execute(
        self,
        waveform: str,
        frequency: float = 440.0,
        amplitude: float = 0.5,
        duration: float = 2.0,
        sample_rate: int = 44100,
        output_file: str = "out.wav",
        stereo: bool = False,
        **_,
    ) -> Dict[str, Any]:
        # Validate
        err = _validate_sr_duration(sample_rate, duration)
        if err:
            return {"success": False, "error": err}
        if not (0.0 <= amplitude <= 1.0):
            return {"success": False, "error": "Amplitude must be 0..1"}
        if waveform not in {"sine", "square", "saw", "white_noise"}:
            return {"success": False, "error": f"Unsupported waveform: {waveform}"}

        n = int(round(sample_rate * duration))
        t = np.arange(n, dtype=np.float32) / float(sample_rate)

        if waveform == "sine":
            x = np.sin(2 * np.pi * frequency * t)
        elif waveform == "square":
            x = np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == "saw":
            # naive saw: 2*(f*t - floor(0.5+f*t))
            x = 2.0 * (frequency * t - np.floor(0.5 + frequency * t))
        else:  # white_noise
            x = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)

        x *= float(amplitude)
        if stereo:
            x = np.stack([x, x], axis=1)

        try:
            out_path = _safe_path(output_file, want_dir=False)
            _write_wav_pcm16(out_path, x, sample_rate)
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "operation": "generate",
            "waveform": waveform,
            "frequency": frequency,
            "amplitude": amplitude,
            "duration": duration,
            "sample_rate": sample_rate,
            "output_file": str(out_path.relative_to(WORKSPACE_ROOT)),
            "channels": 2 if stereo else 1,
        }

# ============================================================
# Audio analysis
# ============================================================

class AudioAnalysisTool(Tool):
    """Analyze a PCM16 WAV: basic stats, averaged spectrum, and truncated spectrogram."""

    def __init__(self):
        super().__init__("audio_analyze", "Analyze PCM16 WAV files with compact JSON outputs.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path under workspace"},
                    "want_spectrogram": {"type": "boolean", "description": "Include truncated spectrogram", "default": True},
                    "nfft": {"type": "integer", "description": "Frame FFT size for spectrum/spectrogram", "default": 1024},
                    "hop": {"type": "integer", "description": "Hop size (samples)", "default": 256},
                },
                "required": ["input_file"]
            }
        }

    async def execute(
        self,
        input_file: str,
        want_spectrogram: bool = True,
        nfft: int = 1024,
        hop: int = 256,
        **_,
    ) -> Dict[str, Any]:
        try:
            in_path = _safe_path(input_file, want_dir=False)
            if not in_path.exists():
                return {"success": False, "error": f"File not found: {in_path}"}
            data, sr, ch = _read_wav_pcm16(in_path)
        except Exception as e:
            return {"success": False, "error": f"Read error: {e}"}

        # Basic stats
        mono = _mix_to_mono(data)
        duration = float(mono.shape[0]) / float(sr)
        rms = float(np.sqrt(np.mean(np.square(mono)) + EPS))
        peak = float(np.max(np.abs(mono)))

        # Averaged spectrum
        nfft = int(max(256, min(nfft, 8192)))
        hop = int(max(64, min(hop, nfft)))
        freqs, avg_mag = _welch_spectrum(mono, sr, nfft=nfft, hop=hop)
        freqs_out, freqs_trunc = _truncate_1d(freqs, MAX_SPECTRUM_BINS)
        mag_out, mag_trunc = _truncate_1d(avg_mag, MAX_SPECTRUM_BINS)

        result: Dict[str, Any] = {
            "success": True,
            "operation": "analyze",
            "file": str(in_path.relative_to(WORKSPACE_ROOT)),
            "sample_rate": sr,
            "channels": ch,
            "duration_sec": duration,
            "level": {"rms": rms, "peak": peak},
            "spectrum": {
                "freqs_hz": freqs_out,
                "magnitude": mag_out,
                "truncated": bool(freqs_trunc or mag_trunc),
                "nfft": nfft,
            },
        }

        if want_spectrogram:
            try:
                f_spec, S_db = _spectrogram(mono, sr, nfft=nfft, hop=hop)
                # truncate 2D
                spec_out, spec_trunc = _truncate_2d(S_db, MAX_SPEC_FRAMES, MAX_SPEC_BINS)
                f_out, f_trunc = _truncate_1d(f_spec, MAX_SPEC_BINS)
                result["spectrogram"] = {
                    "freqs_hz": f_out,
                    "db": spec_out,
                    "shape": [int(S_db.shape[0]), int(S_db.shape[1])],
                    "truncated": bool(spec_trunc or f_trunc),
                    "nfft": nfft,
                    "hop": hop,
                }
            except Exception as e:
                result["spectrogram_error"] = str(e)

        return result

# ============================================================
# Minimal “synth” utility (envelope + simple lowpass)
# ============================================================

class SynthesizerTool(Tool):
    """
    Simple synthesizer: oscillator + linear ADSR + 1-pole lowpass.
    Output: WAV (PCM16). Limited for safety.
    """

    def __init__(self):
        super().__init__("audio_synthesize", "Simple synth (osc + ADSR + lowpass) rendered to WAV (PCM16).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "osc": {"type": "string", "enum": ["sine", "square", "saw"], "description": "Oscillator"},
                    "frequency": {"type": "number", "description": "Hz", "default": 440.0},
                    "duration": {"type": "number", "description": f"Seconds (≤ {MAX_DURATION_SEC})", "default": 2.0},
                    "sample_rate": {"type": "integer", "description": f"Hz (≤ {MAX_SAMPLE_RATE})", "default": 44100},
                    "adsr": {
                        "type": "object",
                        "properties": {
                            "attack": {"type": "number", "default": 0.01},
                            "decay": {"type": "number", "default": 0.1},
                            "sustain": {"type": "number", "default": 0.8},
                            "release": {"type": "number", "default": 0.1},
                        },
                        "required": ["attack", "decay", "sustain", "release"]
                    },
                    "lowpass_hz": {"type": "number", "description": "Cutoff Hz (0 disables)", "default": 0},
                    "amplitude": {"type": "number", "description": "0..1", "default": 0.8},
                    "stereo": {"type": "boolean", "description": "Duplicate mono to L/R", "default": False},
                    "output_file": {"type": "string", "description": "Relative path under workspace"},
                },
                "required": ["osc", "output_file"]
            }
        }

    async def execute(
        self,
        osc: str,
        frequency: float = 440.0,
        duration: float = 2.0,
        sample_rate: int = 44100,
        adsr: Optional[Dict[str, float]] = None,
        lowpass_hz: float = 0.0,
        amplitude: float = 0.8,
        stereo: bool = False,
        output_file: str = "synth.wav",
        **_,
    ) -> Dict[str, Any]:
        err = _validate_sr_duration(sample_rate, duration)
        if err:
            return {"success": False, "error": err}
        if osc not in {"sine", "square", "saw"}:
            return {"success": False, "error": f"Unsupported osc: {osc}"}
        if not (0.0 <= amplitude <= 1.0):
            return {"success": False, "error": "Amplitude must be 0..1"}
        if frequency <= 0 or frequency > sample_rate/2:
            return {"success": False, "error": "Frequency must be within (0, Nyquist]"}

        n = int(round(sample_rate * duration))
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        if osc == "sine":
            x = np.sin(2 * np.pi * frequency * t)
        elif osc == "square":
            x = np.sign(np.sin(2 * np.pi * frequency * t))
        else:  # saw
            x = 2.0 * (frequency * t - np.floor(0.5 + frequency * t))

        # ADSR
        adsr = adsr or {"attack": 0.01, "decay": 0.1, "sustain": 0.8, "release": 0.1}
        a = max(0.0, float(adsr.get("attack", 0.01)))
        d = max(0.0, float(adsr.get("decay", 0.1)))
        s = float(np.clip(adsr.get("sustain", 0.8), 0.0, 1.0))
        r = max(0.0, float(adsr.get("release", 0.1)))
        a_n = int(min(n, round(a * sample_rate)))
        d_n = int(min(n - a_n, round(d * sample_rate)))
        r_n = int(min(n, round(r * sample_rate)))
        s_n = max(0, n - (a_n + d_n + r_n))
        env = np.concatenate([
            np.linspace(0.0, 1.0, max(1, a_n), endpoint=False),
            np.linspace(1.0, s, max(1, d_n), endpoint=False),
            np.full(max(1, s_n), s, dtype=np.float32),
            np.linspace(s, 0.0, max(1, r_n), endpoint=True),
        ])[:n]
        x *= env

        # simple 1-pole lowpass if requested
        if lowpass_hz and lowpass_hz > 0:
            alpha = math.exp(-2.0 * math.pi * float(lowpass_hz) / float(sample_rate))
            y = np.empty_like(x)
            prev = 0.0
            for i in range(len(x)):
                prev = alpha * prev + (1.0 - alpha) * x[i]
                y[i] = prev
            x = y

        x *= float(amplitude)
        if stereo:
            x = np.stack([x, x], axis=1)

        try:
            out_path = _safe_path(output_file, want_dir=False)
            _write_wav_pcm16(out_path, x, sample_rate)
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "operation": "synthesize",
            "osc": osc,
            "frequency": frequency,
            "duration": duration,
            "sample_rate": sample_rate,
            "adsr": {"attack": a, "decay": d, "sustain": s, "release": r},
            "lowpass_hz": float(lowpass_hz),
            "amplitude": amplitude,
            "channels": 2 if stereo else 1,
            "output_file": str(out_path.relative_to(WORKSPACE_ROOT)),
        }

# ============================================================
# Music helpers (pitch/key/scale) for DescribedSFXTool
# ============================================================
_A4 = 440.0
_A4_MIDI = 69
_NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11
}
_SEMI_TO_NOTE_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _midi_to_hz(m: int) -> float:
    return _A4 * (2.0 ** ((m - _A4_MIDI) / 12.0))

def _hz_to_midi(f: float) -> float:
    return _A4_MIDI + 12.0 * math.log2(max(f, 1e-6) / _A4)

def _note_name_from_midi(m: float) -> str:
    m_rounded = int(round(m))
    pc = _SEMI_TO_NOTE_SHARP[m_rounded % 12]
    octv = (m_rounded // 12) - 1
    return f"{pc}{octv}"

def _parse_key(s: Optional[str]) -> Optional[tuple[str, str]]:
    """Return (tonic, mode) like ('A','minor') or None."""
    if not s: return None
    t = s.strip().lower()
    m = re.search(r"\b([a-g](?:#|b)?)\s*(major|minor|maj|min|ionian|aeolian)\b", t)
    if not m: return None
    tonic = m.group(1).capitalize().replace("b","b").replace("#","#")
    mode = m.group(2)
    if mode in ("maj","ionian"): mode = "major"
    if mode in ("min","aeolian"): mode = "minor"
    if tonic.upper() in _NOTE_TO_SEMI and mode in ("major","minor"):
        return (tonic.upper(), mode)
    return None

def _key_from_text_or_param(description: str, key_param: Optional[str]) -> Optional[tuple[str,str]]:
    # explicit param wins; else detect in description
    return _parse_key(key_param) or _parse_key(description)

def _scale_intervals(mode: str) -> list[int]:
    return [0,2,4,5,7,9,11] if mode == "major" else [0,2,3,5,7,8,10]

def _quantize_hz_to_key(freq: float, tonic: str, mode: str) -> tuple[float,str]:
    """Snap a frequency to nearest pitch in the given key (any octave)."""
    target_midi = _hz_to_midi(freq)
    tonic_semi = _NOTE_TO_SEMI[tonic]
    scale = set((tonic_semi + i) % 12 for i in _scale_intervals(mode))
    # search nearby semitones for closest in scale
    best_m = None
    best_err = 1e9
    m_floor = int(math.floor(target_midi)) - 24
    m_ceil  = int(math.ceil (target_midi)) + 24
    for m in range(m_floor, m_ceil + 1):
        if (m % 12) in scale:
            err = abs(m - target_midi)
            if err < best_err:
                best_err, best_m = err, m
    if best_m is None:
        return freq, "?"
    snapped_hz = _midi_to_hz(best_m)
    return snapped_hz, _note_name_from_midi(best_m)

# New helpers for melodic rendering
def _parse_bpm(text: str, default: float = 100.0) -> float:
    m = re.search(r'(\d{2,3})\s*bpm', text.lower())
    if m:
        bpm = float(m.group(1))
    else:
        m2 = re.search(r'\b(\d{2,3})\b', text)
        bpm = float(m2.group(1)) if m2 else default
    return float(max(30.0, min(220.0, bpm)))

def _parse_rhythm(text: str) -> float:
    t = text.lower()
    if '16th' in t or 'sixteenth' in t: return 0.25
    if '8th' in t or 'eighth' in t: return 0.5
    if 'half' in t: return 2.0
    if 'whole' in t: return 4.0
    return 1.0  # quarter by default

def _phrase_pattern(name: str = "arched") -> List[int]:
    if name == "steps":   return [0,1,2,3,4,3,2,1]
    if name == "leaps":   return [0,2,4,7,4,2,0,-2]
    return [0,2,4,5,7,5,4,2]  # arched

# ============================================================
# Text-described sound effects & musical noises (key-aware)
# ============================================================

class DescribedSFXTool(Tool):
    """
    Turn short natural-language descriptions into rendered SFX/music-y noises.
    Now with *melodic* rendering when description suggests melody/tempo/rhythm.
    """

    def __init__(self):
        super().__init__("audio_describe", "Render described noises and musical effects into WAV files (melodic if requested).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "e.g., 'flute melody in G major, 8th notes at 62 BPM' or 'airy whoosh riser'"},
                    "duration": {"type": "number", "description": f"Seconds (≤ {MAX_DURATION_SEC})", "default": 2.5},
                    "sample_rate": {"type": "integer", "description": f"Hz (≤ {MAX_SAMPLE_RATE})", "default": 44100},
                    "output_file": {"type": "string", "description": "Relative path under workspace", "default": "sfx.wav"},
                    "stereo": {"type": "boolean", "description": "Duplicate mono to L/R", "default": True},
                    "key": {"type": "string", "description": "e.g., 'A minor', 'F# major' (quantizes melody or sweep)"}
                },
                "required": ["description", "output_file"]
            }
        }

    async def execute(
        self,
        description: str,
        duration: float = 2.5,
        sample_rate: int = 44100,
        output_file: str = "sfx.wav",
        stereo: bool = True,
        key: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        err = _validate_sr_duration(sample_rate, duration)
        if err:
            return {"success": False, "error": err}

        try:
            spec = self._interpret(description, key)
            x, music_meta = self._render_chain(spec, duration, sample_rate, description)
            if stereo:
                x = np.stack([x, x], axis=1)

            out_path = _safe_path(output_file, want_dir=False)
            _write_wav_pcm16(out_path, x, sample_rate)

            return {
                "success": True,
                "operation": "describe",
                "description": description,
                "duration": duration,
                "sample_rate": sample_rate,
                "stereo": bool(stereo),
                "output_file": str(out_path.relative_to(WORKSPACE_ROOT)),
                "spec": spec,
                "music": music_meta,
            }
        except Exception as e:
            logger.exception("audio_describe failed")
            return {"success": False, "error": str(e)}

    # ---------- prompt interpretation ----------

    def _interpret(self, text: str, key_param: Optional[str] = None) -> Dict[str, Any]:
        """
        Lightweight tagger -> parameter spec.
        Detects when the user wants a *melody* (vs a single sweep/noise).
        """
        t = (text or "").lower()

        # base “archetype” presets
        preset = "generic"
        if any(k in t for k in ["whoosh", "swish", "swoosh"]): preset = "whoosh"
        if any(k in t for k in ["riser", "build", "sweep up"]): preset = "riser"
        if any(k in t for k in ["impact", "hit", "thud", "boom"]): preset = "impact"
        if any(k in t for k in ["drone", "pad", "sustain", "droning"]): preset = "drone"
        if any(k in t for k in ["bleep", "blip"]): preset = "bleep"
        if any(k in t for k in ["laser", "pew"]): preset = "laser"
        if any(k in t for k in ["ui", "click", "tap"]): preset = "ui_click"
        if any(k in t for k in ["alert", "notification"]): preset = "alert"
        if any(k in t for k in ["glitch", "stutter"]): preset = "glitch"

        # ---- melody detection ----
        wants_melody = any(k in t for k in [
            "melody", "melodic", "notes", "arpeggio", "arpeggios", "lead",
            "bpm", "tempo", "8th", "eighth", "16th", "sixteenth", "quarter", "half", "whole"
        ])
        rhythm_beats = _parse_rhythm(t)
        bpm = _parse_bpm(t, default=100.0)

        # timbre hints
        noise = None
        if any(k in t for k in ["air", "airy", "breath", "wind", "shh"]): noise = "pink"
        if any(k in t for k in ["harsh", "static", "noisy"]): noise = "white"

        osc = None
        if any(k in t for k in ["flute", "hollow", "organ"]): osc = "sine"
        if any(k in t for k in ["buzzy", "saw"]): osc = "saw"
        if any(k in t for k in ["nasal", "square"]): osc = "square"

        metallic = any(k in t for k in ["metallic", "clang", "shimmer"])
        bright = any(k in t for k in ["bright", "shiny", "shimmer"])
        dark = any(k in t for k in ["dark", "dull", "low"])
        warm = any(k in t for k in ["warm", "soft"])
        cold = any(k in t for k in ["cold", "icy"])

        # motion/FX hints
        vibrato = any(k in t for k in ["vibrato", "wobble"])
        tremolo = any(k in t for k in ["tremolo", "am wobble"])
        delay = any(k in t for k in ["echo", "delay"])
        reverb = any(k in t for k in ["reverb", "room", "hall", "space"])
        satur = any(k in t for k in ["saturat", "distort", "overdrive"])
        bitcrush = any(k in t for k in ["bitcrush", "lofi", "8-bit", "8bit"])

        # envelope shape hints
        fast_attack = any(k in t for k in ["snappy", "fast", "sharp"])
        long_release = any(k in t for k in ["tail", "long", "pad", "lush"])
        plucky = any(k in t for k in ["plucky", "pluck", "short"])

        # pitch direction
        upward = any(k in t for k in ["rise", "up", "ascend", "riser"])
        downward = any(k in t for k in ["fall", "down", "descend", "drop"])

        # assemble spec with sane defaults by preset
        spec = self._preset_defaults(preset)

        # override / augment from hints
        if noise: spec["noise"] = noise
        if osc: spec["osc"] = osc
        if metallic: spec["metallic"] = True
        spec["tone"] = "bright" if bright else "dark" if dark else "neutral"
        spec["warmth"] = "warm" if warm else "cold" if cold else "neutral"
        spec["fx"]["delay"] = spec["fx"]["delay"] or delay
        spec["fx"]["reverb"] = spec["fx"]["reverb"] or reverb
        spec["fx"]["satur"] = spec["fx"]["satur"] or satur
        spec["fx"]["bitcrush"] = spec["fx"]["bitcrush"] or bitcrush
        spec["mod"]["vibrato"] = spec["mod"]["vibrato"] or vibrato
        spec["mod"]["tremolo"] = spec["mod"]["tremolo"] or tremolo

        if fast_attack: spec["env"]["attack"] = min(spec["env"]["attack"], 0.01)
        if long_release: spec["env"]["release"] = max(spec["env"]["release"], 0.6)
        if plucky:
            spec["env"]["decay"] = 0.08
            spec["env"]["sustain"] = 0.0
            spec["env"]["release"] = min(spec["env"]["release"], 0.12)

        if upward and not downward:
            spec["pitch"]["sweep"] = ("up", 12)   # semitones
        elif downward and not upward:
            spec["pitch"]["sweep"] = ("down", 12)

        # Resolve key from param or text
        resolved = _key_from_text_or_param(text, key_param)
        spec["music"] = {"key": f"{resolved[0]} {resolved[1]}"} if resolved else None

        # Melody block
        spec["melodic"] = bool(wants_melody)
        if wants_melody:
            # ensure an oscillator exists for tonal notes
            if spec.get("osc") is None:
                spec["osc"] = "sine"
            # install melody parameters
            pattern = "arched"
            if any(k in t for k in ["arpeggio", "arpeggios"]): pattern = "leaps"
            if any(k in t for k in ["stepwise", "steps"]): pattern = "steps"
            spec["melody"] = {
                "bpm": float(bpm),
                "beats_per_note": float(rhythm_beats),
                "pattern": pattern,
                "octaves": 2
            }
            # neutralize sweep to avoid fighting melody
            spec["pitch"]["sweep"] = None

        return spec

    def _preset_defaults(self, name: str) -> Dict[str, Any]:
        # Defaults are intentionally gentle; FX mix stays subtle to avoid clipping.
        table = {
            "whoosh":  dict(osc=None, noise="pink", env={"attack":0.02,"decay":0.3,"sustain":0.0,"release":0.6},
                            filter={"type":"lp", "start":400, "end":6000}, pitch={"start":220,"sweep":None},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":True,"satur":False,"bitcrush":False},
                            metallic=False, tone="neutral", warmth="warm"),
            "riser":   dict(osc="saw", noise="pink", env={"attack":0.05,"decay":0.4,"sustain":0.2,"release":0.4},
                            filter={"type":"bp", "start":300, "end":5000}, pitch={"start":110,"sweep":("up",12)},
                            mod={"vibrato":True,"tremolo":False}, fx={"delay":True,"reverb":True,"satur":True,"bitcrush":False},
                            metallic=False, tone="bright", warmth="neutral"),
            "impact":  dict(osc="sine", noise="white", env={"attack":0.001,"decay":0.3,"sustain":0.0,"release":0.4},
                            filter={"type":"lp", "start":120, "end":600}, pitch={"start":70,"sweep":("down",7)},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":True,"satur":True,"bitcrush":False},
                            metallic=True, tone="dark", warmth="neutral"),
            "drone":   dict(osc="sine", noise=None, env={"attack":0.4,"decay":0.8,"sustain":0.7,"release":0.8},
                            filter={"type":"lp", "start":800, "end":1200}, pitch={"start":110,"sweep":None},
                            mod={"vibrato":True,"tremolo":True}, fx={"delay":False,"reverb":True,"satur":False,"bitcrush":False},
                            metallic=False, tone="warm", warmth="warm"),
            "bleep":   dict(osc="square", noise=None, env={"attack":0.001,"decay":0.1,"sustain":0.0,"release":0.06},
                            filter={"type":"bp", "start":1200, "end":3000}, pitch={"start":880,"sweep":None},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":False,"satur":False,"bitcrush":False},
                            metallic=False, tone="bright", warmth="neutral"),
            "laser":   dict(osc="saw", noise=None, env={"attack":0.001,"decay":0.2,"sustain":0.0,"release":0.1},
                            filter={"type":"bp", "start":800, "end":4000}, pitch={"start":2000,"sweep":("down",19)},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":False,"satur":True,"bitcrush":False},
                            metallic=False, tone="bright", warmth="cold"),
            "ui_click":dict(osc=None, noise="white", env={"attack":0.001,"decay":0.06,"sustain":0.0,"release":0.03},
                            filter={"type":"lp", "start":1500, "end":2000}, pitch={"start":1000,"sweep":None},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":False,"satur":False,"bitcrush":False},
                            metallic=False, tone="neutral", warmth="neutral"),
            "alert":   dict(osc="square", noise=None, env={"attack":0.005,"decay":0.15,"sustain":0.2,"release":0.15},
                            filter={"type":"bp", "start":1000, "end":3000}, pitch={"start":880,"sweep":None},
                            mod={"vibrato":True,"tremolo":False}, fx={"delay":False,"reverb":False,"satur":False,"bitcrush":False},
                            metallic=False, tone="bright", warmth="neutral"),
            "glitch":  dict(osc=None, noise="white", env={"attack":0.001,"decay":0.08,"sustain":0.0,"release":0.05},
                            filter={"type":"hp", "start":2000, "end":6000}, pitch={"start":440,"sweep":None},
                            mod={"vibrato":False,"tremolo":True}, fx={"delay":False,"reverb":False,"satur":True,"bitcrush":True},
                            metallic=False, tone="bright", warmth="cold"),
            "generic": dict(osc="sine", noise=None, env={"attack":0.02,"decay":0.2,"sustain":0.1,"release":0.3},
                            filter={"type":"lp", "start":500, "end":3000}, pitch={"start":440,"sweep":None},
                            mod={"vibrato":False,"tremolo":False}, fx={"delay":False,"reverb":True,"satur":False,"bitcrush":False},
                            metallic=False, tone="neutral", warmth="neutral"),
        }
        base = table.get(name, table["generic"])
        # make a shallow copy so later tweaks don’t mutate table
        return {
            "osc": base["osc"], "noise": base["noise"],
            "env": dict(base["env"]), "filter": dict(base["filter"]), "pitch": dict(base["pitch"]),
            "mod": dict(base["mod"]), "fx": dict(base["fx"]),
            "metallic": base["metallic"], "tone": base["tone"], "warmth": base["warmth"]
        }

    # ---------- rendering ----------

    def _render_chain(self, spec: Dict[str, Any], duration: float, sr: int, description: str) -> tuple[np.ndarray, Dict[str, Any]]:
        n = int(round(duration * sr))
        t = np.arange(n, dtype=np.float32) / float(sr)

        # detect if melodic rendering is requested
        melodic = bool(spec.get("melodic"))
        music_meta = {"key": None, "start_note": None, "end_note": None, "quantized": False, "melodic": melodic}

        # determine oscillator + pitch
        start_hz = float(spec["pitch"]["start"])

        # Key handling
        tonic = mode = None
        if spec.get("music"):
            try:
                tonic, mode = spec["music"]["key"].split()
                music_meta["key"] = f"{tonic} {mode}"
            except Exception:
                tonic = mode = None

        # Build frequency trajectory
        if melodic:
            # Build a stepwise *melody* from key, tempo, rhythm, and a phrase pattern
            bpm = float(spec["melody"]["bpm"]) if spec.get("melody") else _parse_bpm(description, 100.0)
            bpn = float(spec["melody"]["beats_per_note"]) if spec.get("melody") else _parse_rhythm(description)
            pattern_name = spec["melody"]["pattern"] if spec.get("melody") else "arched"
            pattern = _phrase_pattern(pattern_name)

            # choose a scale palette
            # default to C major if no key
            if not (tonic and mode):
                tonic, mode = "C", "major"
                music_meta["key"] = "C major (default)"

            tonic_semi = _NOTE_TO_SEMI[tonic]
            degrees = _scale_intervals(mode)  # e.g., [0,2,4,5,7,9,11]
            # center around 5th octave root
            base_midi_center = 12*5 + tonic_semi
            # build 2-octave palette around center
            palette = [base_midi_center + d for d in degrees] + [base_midi_center + 12 + d for d in degrees]

            total_beats = duration * (bpm/60.0)
            beats = 0.0
            idx = 0
            inst_freq = np.empty(n, dtype=np.float32)
            inst_freq[:] = _midi_to_hz(base_midi_center)  # fallback

            while beats < total_beats - 1e-6:
                deg_off = pattern[idx % len(pattern)]
                center_index = len(degrees)  # around second-octave root
                scale_index = max(0, min(len(palette)-1, center_index + deg_off))
                midi = palette[scale_index]
                note_hz = _midi_to_hz(midi)

                start_s = beats * 60.0 / bpm
                dur_s = max(0.05, bpn * 60.0 / bpm)
                n0 = int(round(start_s * sr))
                n1 = min(n, n0 + int(round(dur_s * sr)))
                if n0 >= n: break
                inst_freq[n0:n1] = note_hz

                if music_meta["start_note"] is None:
                    music_meta["start_note"] = _note_name_from_midi(midi)
                music_meta["end_note"] = _note_name_from_midi(midi)

                beats += bpn
                idx += 1

            # fill any trailing samples
            if n1 < n:
                inst_freq[n1:] = inst_freq[n1-1]

            # render oscillator with per-sample frequency
            phase = 2 * np.pi * np.cumsum(inst_freq) / float(sr)
        else:
            # classic SFX path: continuous tone (possibly sweeping),
            # with optional key-quantized endpoints
            end_hz = None
            if spec["pitch"].get("sweep"):
                direction, semis = spec["pitch"]["sweep"]
                ratio = 2 ** (abs(semis) / 12.0)
                end_hz = start_hz * (ratio if direction == "up" else 1.0/ratio)

            if tonic and mode:
                q_start, start_name = _quantize_hz_to_key(start_hz, tonic, mode)
                start_hz = q_start
                music_meta.update({"start_note": start_name, "quantized": True})
                if end_hz is not None:
                    q_end, end_name = _quantize_hz_to_key(end_hz, tonic, mode)
                    end_hz = q_end
                    music_meta["end_note"] = end_name

            if end_hz is not None:
                g = np.linspace(0.0, 1.0, n, dtype=np.float32)
                inst_freq = start_hz * (end_hz / start_hz) ** g
            else:
                inst_freq = np.full(n, start_hz, dtype=np.float32)

            phase = 2 * np.pi * np.cumsum(inst_freq) / float(sr)

        # source
        x = np.zeros(n, dtype=np.float32)
        if spec.get("noise"):
            mode_noise = spec["noise"]
            if mode_noise == "white":
                x += np.random.uniform(-1, 1, size=n).astype(np.float32)
            elif mode_noise == "pink":
                b = np.random.uniform(-1, 1, size=(16, n)).astype(np.float32)
                x += (b.cumsum(axis=1)[-1] / np.arange(1, n+1)).astype(np.float32)
                x /= np.max(np.abs(x) + 1e-6)

        if spec.get("osc"):
            wave = spec["osc"]
            if wave == "sine":
                x += np.sin(phase)
            elif wave == "square":
                x += np.sign(np.sin(phase))
            elif wave == "saw":
                x += 2.0 * (np.modf(phase / (2*np.pi))[0] - 0.5)

        # normalize source a bit
        x /= max(1.0, np.max(np.abs(x)) + 1e-6)

        # filter sweep (one-pole approximations)
        f = spec["filter"]
        if f["type"] in ("lp", "hp", "bp"):
            f_start, f_end = float(f["start"]), float(f["end"])
            sweep = np.linspace(f_start, f_end, n).astype(np.float32)
            if f["type"] == "lp":
                x = self._one_pole_lp_sweep(x, sweep, sr)
            elif f["type"] == "hp":
                x = self._one_pole_hp_sweep(x, sweep, sr)
            else:  # bp
                lo = self._one_pole_lp_sweep(x, sweep, sr)
                hi = self._one_pole_hp_sweep(x, np.maximum(sweep/3, 40), sr)
                x = lo - hi

        # ADSR (global)
        e = spec["env"]
        env = self._adsr(n, sr, e["attack"], e["decay"], e["sustain"], e["release"])

        # If melodic, add a per-note gating to articulate notes clearly
        if melodic:
            bpm = float(spec["melody"]["bpm"])
            bpn = float(spec["melody"]["beats_per_note"])
            note_len_s = bpn * 60.0 / bpm
            gate = np.zeros(n, dtype=np.float32)
            t_note = 0.0
            while t_note < duration - 1e-6:
                n0 = int(round(t_note * sr))
                n1 = min(n, n0 + int(round(note_len_s * sr)))
                if n0 >= n: break
                a = max(1, int(0.01 * sr))  # 10ms attack
                r = max(1, int(0.04 * sr))  # 40ms release
                seg = np.ones(max(1, n1 - n0), dtype=np.float32)
                a = min(a, seg.shape[0]//2)
                r = min(r, seg.shape[0]//2)
                if a > 0: seg[:a] = np.linspace(0, 1, a, dtype=np.float32)
                if r > 0: seg[-r:] = np.linspace(1, 0, r, dtype=np.float32)
                gate[n0:n1] = np.maximum(gate[n0:n1], seg)
                t_note += note_len_s
            env = env * gate

        x *= env

        # modulation
        if spec["mod"].get("tremolo"):
            rate = 6.0  # Hz
            trem = 0.5 * (1 + np.sin(2*np.pi*rate*t))
            x *= trem
        if spec["mod"].get("vibrato"):
            rate = 5.0
            depth = 0.002  # seconds of delay deviation (phasor trick)
            vib = np.sin(2*np.pi*rate*t) * depth
            x = np.sin(2*np.pi*2.0 * t + 50 * vib) * x * 0.5 + x * 0.5

        # coloration
        if spec["fx"].get("satur"):
            x = np.tanh(1.5 * x)
        if spec["fx"].get("bitcrush"):
            step = 1/32.0
            x = np.round(x/step)*step

        # delay (feedback echo)
        if spec["fx"].get("delay"):
            x = self._delay(x, sr, ms=180, feedback=0.25, wet=0.2)

        # tiny reverb (Schroeder-ish)
        if spec["fx"].get("reverb"):
            x = self._reverb(x, sr, wet=0.18)

        # subtle metallic sparkle
        if spec.get("metallic"):
            x = self._comb(x, sr, ms=12.0, gain=0.35)

        # final trim
        x = np.clip(x, -1.0, 1.0)
        return x.astype(np.float32), music_meta

    # ---------- small DSP helpers ----------

    def _adsr(self, n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
        a_n = int(min(n, round(max(0.0, a) * sr)))
        d_n = int(min(n - a_n, round(max(0.0, d) * sr)))
        r_n = int(min(n, round(max(0.0, r) * sr)))
        s_n = max(0, n - (a_n + d_n + r_n))
        return np.concatenate([
            np.linspace(0.0, 1.0, max(1, a_n), endpoint=False),
            np.linspace(1.0, s, max(1, d_n), endpoint=False),
            np.full(max(1, s_n), s, dtype=np.float32),
            np.linspace(s, 0.0, max(1, r_n), endpoint=True)
        ])[:n]

    def _one_pole_lp_sweep(self, x: np.ndarray, cutoff: np.ndarray, sr: int) -> np.ndarray:
        y = np.empty_like(x)
        z = 0.0
        for i in range(len(x)):
            alpha = math.exp(-2.0 * math.pi * cutoff[i] / sr)
            z = alpha * z + (1.0 - alpha) * x[i]
            y[i] = z
        return y

    def _one_pole_hp_sweep(self, x: np.ndarray, cutoff: np.ndarray, sr: int) -> np.ndarray:
        y = np.empty_like(x)
        z = 0.0
        for i in range(len(x)):
            alpha = math.exp(-2.0 * math.pi * cutoff[i] / sr)
            z = alpha * z + (1.0 - alpha) * x[i]
            y[i] = x[i] - z
        return y

    def _delay(self, x: np.ndarray, sr: int, ms: float, feedback: float, wet: float) -> np.ndarray:
        d = max(1, int(sr * (ms / 1000.0)))
        y = x.copy()
        for i in range(d, len(x)):
            y[i] += wet * (x[i - d] + feedback * y[i - d])
        m = np.max(np.abs(y)) + 1e-6
        y /= max(1.0, m)
        return y

    def _reverb(self, x: np.ndarray, sr: int, wet: float = 0.2) -> np.ndarray:
        def comb(sig, ms, g):
            d = max(1, int(sr * (ms/1000.0)))
            y = sig.copy()
            for i in range(d, len(sig)):
                y[i] += g * y[i - d]
            return y
        def allpass(sig, ms, g):
            d = max(1, int(sr * (ms/1000.0)))
            y = np.zeros_like(sig)
            for i in range(len(sig)):
                dry = sig[i]
                delayed = y[i - d] if i >= d else 0.0
                xin = dry + g * delayed
                y[i] = -g * xin + (y[i - d] if i >= d else 0.0) + dry
            return y

        a = comb(x, 29.7, 0.77)
        b = comb(x, 37.1, 0.71)
        c = comb(x, 41.1, 0.68)
        mix = (a + b + c) / 3.0
        ap = allpass(mix, 5.0, 0.5)
        out = (1 - wet) * x + wet * ap
        m = np.max(np.abs(out)) + 1e-6
        return out / max(1.0, m)

    def _comb(self, x: np.ndarray, sr: int, ms: float, gain: float) -> np.ndarray:
        d = max(1, int(sr * (ms/1000.0)))
        y = x.copy()
        for i in range(d, len(x)):
            y[i] += gain * x[i - d]
        m = np.max(np.abs(y)) + 1e-6
        return y / max(1.0, m)
