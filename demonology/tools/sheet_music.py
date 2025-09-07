from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import music21  # type: ignore
    _HAS_M21 = True
except Exception:
    _HAS_M21 = False

try:
    import numpy as _np  # type: ignore
    import soundfile as _sf  # type: ignore
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

from .base import Tool

logger = logging.getLogger(__name__)

# ----- Workspace confinement (mirrors media/audio patterns) -----
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
    # Disallow symlink traversal (best effort)
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

class SheetMusicOMRTool(Tool):
    """
    Convert sheet music (image/PDF) to MusicXML/MIDI and optional audio (WAV).
    Prefers Audiveris for OMR if available; otherwise requires MusicXML input.
    """

    def __init__(self):
        super().__init__(
            "sheet_music_omr",
            "Decipher sheet music to MusicXML/MIDI (via Audiveris if available) and optionally synthesize audio."
        )

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to sheet image/PDF or MusicXML under workspace"
                    },
                    "output_stem": {
                        "type": "string",
                        "description": "Output stem (no extension) under workspace",
                        "default": "scores/output"
                    },
                    "omr_engine": {
                        "type": "string",
                        "enum": ["auto", "audiveris", "none"],
                        "description": "OMR method: use Audiveris if available, or skip if input is MusicXML",
                        "default": "auto"
                    },
                    "make_midi": {
                        "type": "boolean",
                        "description": "Export MIDI if possible (music21 required)",
                        "default": True
                    },
                    "make_wav": {
                        "type": "boolean",
                        "description": "Quick synth to WAV (sine-based; numpy+soundfile required)",
                        "default": False
                    },
                    "tempo_bpm": {
                        "type": "number",
                        "description": "Override tempo (BPM) for WAV rendering",
                        "default": 100
                    },
                    "transpose_semitones": {
                        "type": "integer",
                        "description": "Transpose output by N semitones before synth",
                        "default": 0
                    }
                },
                "required": ["input_path"]
            }
        }

    async def execute(
        self,
        input_path: str,
        output_stem: str = "scores/output",
        omr_engine: str = "auto",
        make_midi: bool = True,
        make_wav: bool = False,
        tempo_bpm: float = 100.0,
        transpose_semitones: int = 0,
        **_: Any
    ) -> Dict[str, Any]:
        try:
            in_path = _safe_path(input_path, want_dir=False)
            out_stem = _safe_path(output_stem, want_dir=False)
        except Exception as e:
            return {"success": False, "error": f"Path error: {e}"}

        result: Dict[str, Any] = {
            "success": False,
            "operation": "sheet_music_omr",
            "input": str(in_path.relative_to(WORKSPACE_ROOT)),
            "outputs": {}
        }

        # Step 1: Ensure we have MusicXML (either input already XML or via OMR)
        xml_path: Optional[Path] = None
        if in_path.suffix.lower() in {".xml", ".mxl", ".musicxml"}:
            xml_path = in_path
            result["omr"] = {"skipped": True, "reason": "Input is already MusicXML"}
        else:
            if omr_engine in ("auto", "audiveris"):
                xml_path, omr_meta = await self._run_audiveris(in_path, out_stem)
                result["omr"] = omr_meta
            else:
                return {"success": False, "error": "OMR disabled and input is not MusicXML"}

        if not xml_path or not xml_path.exists():
            result["error"] = "Failed to obtain MusicXML from input."
            return result

        result["outputs"]["musicxml"] = str(xml_path.relative_to(WORKSPACE_ROOT))

        # Step 2: Optionally make MIDI (via music21)
        midi_path: Optional[Path] = None
        if make_midi:
            if not _HAS_M21:
                result.setdefault("warnings", []).append("music21 not installed; skipping MIDI export")
            else:
                try:
                    midi_path = await self._xml_to_midi(xml_path, out_stem.with_suffix(".mid"), transpose_semitones)
                    result["outputs"]["midi"] = str(midi_path.relative_to(WORKSPACE_ROOT))
                except Exception as e:
                    result.setdefault("warnings", []).append(f"MIDI export failed: {e}")

        # Step 3: Optionally synth to WAV (basic sine synth)
        if make_wav:
            if not _HAS_M21 or not _HAS_AUDIO:
                result.setdefault("warnings", []).append("music21/numpy/soundfile missing; WAV synth skipped")
            else:
                try:
                    wav_path = await self._xml_to_wav(
                        xml_path, out_stem.with_suffix(".wav"),
                        tempo_bpm=float(tempo_bpm),
                        transpose_semitones=int(transpose_semitones)
                    )
                    result["outputs"]["wav"] = str(wav_path.relative_to(WORKSPACE_ROOT))
                except Exception as e:
                    result.setdefault("warnings", []).append(f"WAV synth failed: {e}")

        result["success"] = True
        return result

    # ----------------------- helpers -----------------------

    async def _run_audiveris(self, input_file: Path, out_stem: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
        """
        Use Audiveris (if available) to convert image/PDF to MusicXML.
        Search order: AUDIVERIS_CLI, 'audiveris' on PATH, AUDIVERIS_JAR (java -jar).
        """
        meta: Dict[str, Any] = {"engine": "audiveris", "ok": False}
        cli = os.environ.get("AUDIVERIS_CLI") or shutil.which("audiveris")
        jar = os.environ.get("AUDIVERIS_JAR")

        out_dir = out_stem.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if cli:
            cmd = [cli, "-batch", "-export", "-output", str(out_dir), str(input_file)]
        elif jar:
            cmd = ["java", "-jar", jar, "-batch", "-export", "-output", str(out_dir), str(input_file)]
        else:
            meta["error"] = "Audiveris not found (set AUDIVERIS_CLI or AUDIVERIS_JAR)"
            return None, meta

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(os.environ.get("OMR_TIMEOUT", "180")))
            except asyncio.TimeoutError:
                proc.kill()
                meta["error"] = "Audiveris timeout"
                return None, meta
            meta["status"] = proc.returncode
            meta["stdout"] = (stdout or b"")[-4000:].decode("utf-8", errors="replace")
            meta["stderr"] = (stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if proc.returncode != 0:
                meta["error"] = "Audiveris non-zero exit"
                return None, meta

            # Find an XML/MXL in out_dir
            candidates = list(out_dir.glob("*.mxl")) + list(out_dir.glob("*.xml"))
            if not candidates:
                meta["error"] = "No MusicXML produced"
                return None, meta
            # Prefer MXL
            candidates.sort(key=lambda p: (p.suffix.lower() != ".mxl", p.name))
            meta["ok"] = True
            return candidates[0], meta
        except FileNotFoundError as e:
            meta["error"] = f"Audiveris exec failed: {e}"
            return None, meta

    async def _xml_to_midi(self, xml_path: Path, midi_out: Path, transpose_semitones: int) -> Path:
        score = music21.converter.parse(str(xml_path))
        if transpose_semitones:
            score = score.transpose(transpose_semitones)
        midi_out.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(midi_out))
        return midi_out

    async def _xml_to_wav(self, xml_path: Path, wav_out: Path, tempo_bpm: float, transpose_semitones: int) -> Path:
        """
        Very simple sine synth from the first part:
        - Reads quarterLength for durations
        - Maps MIDI -> freq (A4=440)
        - Mixes notes with naive overlap handling
        """
        score = music21.converter.parse(str(xml_path))
        if transpose_semitones:
            score = score.transpose(transpose_semitones)

        # Flatten to a single stream for basic rendering
        s = score.parts[0].flat if score.parts else score.flat

        sr = int(os.environ.get("SYNTH_SAMPLE_RATE", "44100"))
        sec_per_quarter = 60.0 / max(1e-6, float(tempo_bpm))

        # Build note list (start_time_seconds, duration_seconds, midi_pitch, velocity-like)
        notes: List[Tuple[float, float, int, float]] = []
        t_cursor = 0.0
        for el in s.notesAndRests:
            dur = float(el.quarterLength) * sec_per_quarter
            if el.isRest:
                t_cursor += dur
                continue
            if hasattr(el, "pitch"):
                midi = int(el.pitch.midi)
                vel = 0.8
                notes.append((t_cursor, dur, midi, vel))
            t_cursor += dur

        if not notes:
            raise RuntimeError("No notes found to synthesize")

        total_dur = notes[-1][0] + notes[-1][1] + 0.5
        n_samples = int(total_dur * sr)
        buf = _np.zeros(n_samples, dtype=_np.float32)

        def midi_to_hz(m: int) -> float:
            return 440.0 * (2.0 ** ((m - 69) / 12.0))

        for start, dur, midi, vel in notes:
            f = midi_to_hz(midi)
            n0 = int(start * sr)
            n1 = min(n_samples, n0 + int(dur * sr))
            t = _np.arange(0, n1 - n0, dtype=_np.float32) / sr
            # Simple sine with fast attack/decay
            wave = _np.sin(2 * _np.pi * f * t).astype(_np.float32)
            env_len = max(1, int(0.01 * sr))
            env = _np.ones_like(wave)
            env[:env_len] = _np.linspace(0, 1, env_len, dtype=_np.float32)
            env[-env_len:] *= _np.linspace(1, 0, env_len, dtype=_np.float32)
            wave *= env * vel * 0.3
            buf[n0:n1] += wave

        # Normalize
        peak = float(_np.max(_np.abs(buf))) or 1.0
        buf /= max(1.0, peak)
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        _sf.write(str(wav_out), buf, sr, subtype="PCM_16")
        return wav_out
