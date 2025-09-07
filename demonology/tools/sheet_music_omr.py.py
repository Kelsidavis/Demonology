from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    from .base import Tool
except Exception:
    # Allow running as a loose module for testing
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description
        def to_openai_function(self) -> Dict[str, Any]: ...
        async def execute(self, **kwargs) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)


class SheetMusicOMRTool(Tool):
    """Optical Music Recognition (OMR) via Audiveris + optional MusicXML fixup (music21).

    - Converts a sheet music image/PDF to MusicXML/MIDI with Audiveris.
    - Optionally post-processes with music21 to set/override time signature, key, and tempo,
      and to clean/quantize the score for better playback.
    """

    def __init__(self):
        super().__init__(
            "sheet_music_omr",
            "Convert sheet music images/PDF to MusicXML/MIDI using Audiveris, with optional TS/key/tempo fix."
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
                        "description": "Path to input image/PDF (e.g., PNG, JPG, PDF)"
                    },
                    "output_stem": {
                        "type": "string",
                        "description": "Output stem path without extension (e.g., 'scores/output')"
                    },
                    "audiveris_bin": {
                        "type": "string",
                        "description": "Path to the Audiveris executable; defaults to $AUDIVERIS_BIN or 'audiveris'"
                    },
                    "timesig": {
                        "type": "string",
                        "description": "Optional time signature to enforce (e.g., '4/4')"
                    },
                    "key_text": {
                        "type": "string",
                        "description": "Optional key to enforce (e.g., 'A minor' or 'C major')"
                    },
                    "tempo": {
                        "type": "number",
                        "description": "Optional tempo in BPM to enforce"
                    },
                    "autofix_missing": {
                        "type": "boolean",
                        "description": "If true, run fix only when TS/key/tempo are missing in the OMR result",
                        "default": True
                    },
                    "force_fix": {
                        "type": "boolean",
                        "description": "If true, always run fix and override with provided TS/key/tempo when set",
                        "default": False
                    }
                },
                "required": ["input_path"]
            }
        }

    def _find_audiveris(self, audiveris_bin: Optional[str]) -> Optional[str]:
        candidate = audiveris_bin or os.environ.get("AUDIVERIS_BIN") or "audiveris"
        if shutil.which(candidate):
            return candidate
        return None

    async def _run_audiveris(self, bin_path: str, input_path: Path, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Basic CLI: audiveris -batch -export -output <dir> <input>
        cmd = [bin_path, "-batch", "-export", "-output", str(out_dir), str(input_path)]
        logger.info("Running Audiveris: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace")
        }

    def _pick_musicxml(self, out_dir: Path, stem_hint: Optional[str]) -> Optional[Path]:
        # Try exact stem first (musicxml or mxl), then any exported file
        if stem_hint:
            p1 = Path(stem_hint).with_suffix(".musicxml")
            p2 = Path(stem_hint).with_suffix(".mxl")
            if p1.exists():
                return p1
            if p2.exists():
                return p2
        # fallback: any *.musicxml or *.mxl in out_dir
        cands = list(out_dir.glob("*.musicxml")) + list(out_dir.glob("*.mxl"))
        return cands[0] if cands else None

    def _needs_fix(self, xml_path: Path) -> Dict[str, bool]:
        """Inspect MusicXML for time signature / key / tempo; True if any missing."""
        needs = {"ts": False, "key": False, "tempo": False}
        try:
            from music21 import converter, meter, key as keymod, tempo as tempomod
            s = converter.parse(str(xml_path))
            # TS
            ts = list(s.recurse().getElementsByClass(meter.TimeSignature))
            needs["ts"] = (len(ts) == 0)
            # Key
            k = list(s.recurse().getElementsByClass(keymod.Key))
            needs["key"] = (len(k) == 0)
            # Tempo
            m = list(s.recurse().getElementsByClass(tempomod.MetronomeMark))
            needs["tempo"] = (len(m) == 0)
        except Exception as e:
            logger.warning("Could not inspect MusicXML: %s", e)
        return needs

    def _apply_fix(self, xml_in: Path, out_stem: Path, timesig: Optional[str], key_text: Optional[str], tempo_bpm: Optional[float]) -> Dict[str, Any]:
        try:
            from music21 import converter, meter, key as keymod, tempo, stream
        except Exception as e:
            return {"ok": False, "error": f"music21 missing: {e}"}

        try:
            s = converter.parse(str(xml_in))
            target = s.parts.stream() if hasattr(s, "parts") and len(s.parts) else s

            if timesig:
                try:
                    target.insert(0, meter.TimeSignature(timesig))
                except Exception as e:
                    logger.warning("TS set failed (%s): %s", timesig, e)

            if key_text:
                try:
                    tok = key_text.strip().split()
                    if len(tok) == 2:
                        tonic, mode = tok
                        k = keymod.Key(tonic, mode.lower())
                    else:
                        k = keymod.Key(key_text)
                    target.insert(0, k)
                except Exception as e:
                    logger.warning("Key set failed (%s): %s", key_text, e)

            if tempo_bpm:
                try:
                    mm = tempo.MetronomeMark(number=float(tempo_bpm))
                    target.insert(0, mm)
                except Exception as e:
                    logger.warning("Tempo set failed (%s): %s", tempo_bpm, e)

            try:
                target.makeMeasures(inPlace=True)
                target.makeNotation(inPlace=True, hideRests=False, bestClef=True)
            except Exception as e:
                logger.debug("makeNotation warn: %s", e)

            out_xml = out_stem.with_suffix(".musicxml")
            out_mid = out_stem.with_suffix(".mid")
            s.write("musicxml", fp=str(out_xml))
            s.write("midi", fp=str(out_mid))
            return {"ok": True, "musicxml": str(out_xml), "midi": str(out_mid)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def execute(
        self,
        input_path: str,
        output_stem: Optional[str] = None,
        audiveris_bin: Optional[str] = None,
        timesig: Optional[str] = None,
        key_text: Optional[str] = None,
        tempo: Optional[float] = None,
        autofix_missing: bool = True,
        force_fix: bool = False,
        **kwargs
    ) -> Dict[str, Any]:

        inp = Path(input_path).expanduser().resolve()
        if not inp.exists():
            return {"success": False, "error": f"Input not found: {inp}"}

        # Resolve output dir & stem
        out_dir = Path(output_stem).parent if output_stem else inp.parent
        out_dir = out_dir if out_dir != Path("") else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        aud = self._find_audiveris(audiveris_bin)
        if not aud:
            return {"success": False, "error": "Audiveris not found. Set $AUDIVERIS_BIN or install 'audiveris' in PATH."}

        omr = await self._run_audiveris(aud, inp, out_dir)
        if omr["returncode"] != 0:
            return {"success": False, "stage": "audiveris", "stdout": omr["stdout"], "stderr": omr["stderr"]}

        # Try to locate the exported MusicXML
        xml = self._pick_musicxml(out_dir, output_stem)
        if not xml:
            return {
                "success": False,
                "stage": "export",
                "error": "No MusicXML/MXL produced by Audiveris",
                "stdout": omr["stdout"],
                "stderr": omr["stderr"],
            }

        results: Dict[str, Any] = {
            "success": True,
            "stage": "audiveris_done",
            "musicxml": str(xml),
            "stdout": omr["stdout"][-2000:],
            "stderr": omr["stderr"][-1000:],
            "fix_applied": False,
            "midi": None,
        }

        # Decide whether to apply fix
        will_fix = force_fix
        if not will_fix and autofix_missing:
            miss = self._needs_fix(xml)
            # Only fix if we can add something (provided or just cleaning)
            will_fix = (miss.get("ts") and timesig) or (miss.get("key") and key_text) or (miss.get("tempo") and tempo)

        if will_fix or timesig or key_text or tempo:
            stem = Path(output_stem) if output_stem else xml.with_suffix("")
            fix = self._apply_fix(xml, stem, timesig, key_text, tempo)
            results["fix_applied"] = True
            if fix.get("ok"):
                results["stage"] = "fixed"
                results["musicxml"] = fix.get("musicxml", results["musicxml"])
                results["midi"] = fix.get("midi")
            else:
                results["stage"] = "fix_failed"
                results["fix_error"] = fix.get("error")

        return results
