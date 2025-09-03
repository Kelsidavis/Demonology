# demonology/tools/audio.py
from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, _safe_path

logger = logging.getLogger(__name__)


class WaveformGeneratorTool(Tool):
    """Generate basic waveforms (sine, square, sawtooth, triangle)."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("waveform_generator", "Generate basic audio waveforms and save as WAV files.")
        from .base import SAFE_ROOT
        self.safe_root = safe_root or SAFE_ROOT
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "waveform_type": {
                        "type": "string",
                        "enum": ["sine", "square", "sawtooth", "triangle", "noise"],
                        "description": "Type of waveform to generate",
                        "default": "sine"
                    },
                    "frequency": {
                        "type": "number",
                        "description": "Frequency in Hz",
                        "default": 440.0
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds", 
                        "default": 1.0
                    },
                    "sample_rate": {
                        "type": "integer",
                        "description": "Sample rate in Hz",
                        "default": 44100
                    },
                    "amplitude": {
                        "type": "number",
                        "description": "Amplitude (0.0 to 1.0)",
                        "default": 0.5
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output WAV file path",
                        "default": "generated_waveform.wav"
                    }
                },
                "required": []
            }
        }
    
    async def execute(self, waveform_type: str = "sine", frequency: float = 440.0,
                     duration: float = 1.0, sample_rate: int = 44100,
                     amplitude: float = 0.5, output_file: str = "generated_waveform.wav",
                     **_) -> Dict[str, Any]:
        try:
            import numpy as np
            import wave
            import struct
        except ImportError:
            return {"success": False, "error": "NumPy required for audio generation. Install with: pip install numpy"}
        
        try:
            # Validate parameters
            if frequency <= 0:
                return {"success": False, "error": "Frequency must be positive"}
            if duration <= 0:
                return {"success": False, "error": "Duration must be positive"}
            if not 0 <= amplitude <= 1:
                return {"success": False, "error": "Amplitude must be between 0 and 1"}
            
            # Generate time array
            num_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            
            # Generate waveform
            if waveform_type == "sine":
                samples = amplitude * np.sin(2 * np.pi * frequency * t)
            elif waveform_type == "square":
                samples = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
            elif waveform_type == "sawtooth":
                samples = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
            elif waveform_type == "triangle":
                samples = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - amplitude
            elif waveform_type == "noise":
                samples = amplitude * (2 * np.random.random(num_samples) - 1)
            else:
                return {"success": False, "error": f"Unknown waveform type: {waveform_type}"}
            
            # Convert to 16-bit integers
            samples_int = (samples * 32767).astype(np.int16)
            
            # Save to WAV file
            output_path = _safe_path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(output_path), 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples_int.tobytes())
            
            return {
                "success": True,
                "waveform_type": waveform_type,
                "frequency": frequency,
                "duration": duration,
                "sample_rate": sample_rate,
                "amplitude": amplitude,
                "output_file": str(output_path),
                "file_size": output_path.stat().st_size,
                "num_samples": num_samples
            }
            
        except Exception as e:
            logger.exception("WaveformGeneratorTool error")
            return {"success": False, "error": str(e)}


class SynthesizerTool(Tool):
    """Advanced synthesis including FM, AM, and subtractive synthesis."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("synthesizer", "Advanced audio synthesis with FM, AM, and filtering.")
        from .base import SAFE_ROOT
        self.safe_root = safe_root or SAFE_ROOT
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "synthesis_type": {
                        "type": "string",
                        "enum": ["fm", "am", "subtractive", "additive"],
                        "description": "Type of synthesis",
                        "default": "fm"
                    },
                    "carrier_freq": {
                        "type": "number",
                        "description": "Carrier frequency in Hz",
                        "default": 440.0
                    },
                    "modulator_freq": {
                        "type": "number", 
                        "description": "Modulator frequency in Hz (for FM/AM)",
                        "default": 100.0
                    },
                    "modulation_depth": {
                        "type": "number",
                        "description": "Modulation depth (0.0 to 1.0)",
                        "default": 0.5
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds",
                        "default": 2.0
                    },
                    "sample_rate": {
                        "type": "integer",
                        "description": "Sample rate in Hz",
                        "default": 44100
                    },
                    "amplitude": {
                        "type": "number",
                        "description": "Amplitude (0.0 to 1.0)",
                        "default": 0.5
                    },
                    "filter_cutoff": {
                        "type": "number",
                        "description": "Low-pass filter cutoff frequency (for subtractive)",
                        "default": 1000.0
                    },
                    "envelope": {
                        "type": "object",
                        "description": "ADSR envelope parameters",
                        "properties": {
                            "attack": {"type": "number", "default": 0.1},
                            "decay": {"type": "number", "default": 0.3},
                            "sustain": {"type": "number", "default": 0.7},
                            "release": {"type": "number", "default": 0.5}
                        }
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output WAV file path",
                        "default": "synthesized_audio.wav"
                    }
                },
                "required": []
            }
        }
    
    async def execute(self, synthesis_type: str = "fm", carrier_freq: float = 440.0,
                     modulator_freq: float = 100.0, modulation_depth: float = 0.5,
                     duration: float = 2.0, sample_rate: int = 44100,
                     amplitude: float = 0.5, filter_cutoff: float = 1000.0,
                     envelope: Optional[Dict] = None, output_file: str = "synthesized_audio.wav",
                     **_) -> Dict[str, Any]:
        try:
            import numpy as np
            import wave
            from scipy import signal
        except ImportError:
            return {"success": False, "error": "NumPy and SciPy required. Install with: pip install numpy scipy"}
        
        try:
            # Default envelope if not provided
            if envelope is None:
                envelope = {"attack": 0.1, "decay": 0.3, "sustain": 0.7, "release": 0.5}
            
            # Generate time array
            num_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            
            # Generate synthesis
            if synthesis_type == "fm":
                # FM synthesis
                modulator = np.sin(2 * np.pi * modulator_freq * t)
                carrier = np.sin(2 * np.pi * carrier_freq * t + modulation_depth * modulator)
                samples = amplitude * carrier
                
            elif synthesis_type == "am":
                # AM synthesis
                modulator = (1 + modulation_depth * np.sin(2 * np.pi * modulator_freq * t))
                carrier = np.sin(2 * np.pi * carrier_freq * t)
                samples = amplitude * modulator * carrier
                
            elif synthesis_type == "subtractive":
                # Subtractive synthesis with noise and filtering
                noise = 2 * np.random.random(num_samples) - 1
                # Apply low-pass filter
                nyquist = sample_rate / 2
                normalized_cutoff = filter_cutoff / nyquist
                b, a = signal.butter(4, normalized_cutoff, btype='low')
                filtered_noise = signal.filtfilt(b, a, noise)
                samples = amplitude * filtered_noise
                
            elif synthesis_type == "additive":
                # Additive synthesis with harmonics
                samples = np.zeros(num_samples)
                for harmonic in range(1, 6):  # First 5 harmonics
                    harmonic_amp = amplitude / harmonic
                    harmonic_freq = carrier_freq * harmonic
                    samples += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
                    
            else:
                return {"success": False, "error": f"Unknown synthesis type: {synthesis_type}"}
            
            # Apply ADSR envelope
            envelope_samples = self._generate_adsr_envelope(
                num_samples, sample_rate, 
                envelope["attack"], envelope["decay"], 
                envelope["sustain"], envelope["release"]
            )
            samples = samples * envelope_samples
            
            # Convert to 16-bit integers
            samples_int = (samples * 32767).astype(np.int16)
            
            # Save to WAV file
            output_path = _safe_path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(output_path), 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples_int.tobytes())
            
            return {
                "success": True,
                "synthesis_type": synthesis_type,
                "carrier_freq": carrier_freq,
                "modulator_freq": modulator_freq,
                "modulation_depth": modulation_depth,
                "duration": duration,
                "sample_rate": sample_rate,
                "amplitude": amplitude,
                "envelope": envelope,
                "output_file": str(output_path),
                "file_size": output_path.stat().st_size
            }
            
        except Exception as e:
            logger.exception("SynthesizerTool error")
            return {"success": False, "error": str(e)}
    
    def _generate_adsr_envelope(self, num_samples: int, sample_rate: int,
                               attack: float, decay: float, sustain: float, release: float) -> 'np.ndarray':
        """Generate ADSR envelope."""
        import numpy as np
        
        envelope = np.zeros(num_samples)
        
        # Convert times to samples
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        release_samples = int(release * sample_rate)
        sustain_samples = max(0, num_samples - attack_samples - decay_samples - release_samples)
        
        idx = 0
        
        # Attack phase
        if attack_samples > 0:
            envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples
        
        # Decay phase
        if decay_samples > 0:
            envelope[idx:idx+decay_samples] = np.linspace(1, sustain, decay_samples)
            idx += decay_samples
        
        # Sustain phase
        if sustain_samples > 0:
            envelope[idx:idx+sustain_samples] = sustain
            idx += sustain_samples
        
        # Release phase
        if release_samples > 0 and idx < num_samples:
            remaining = min(release_samples, num_samples - idx)
            envelope[idx:idx+remaining] = np.linspace(sustain, 0, remaining)
        
        return envelope


class AudioAnalysisTool(Tool):
    """Analyze audio files with FFT, spectrograms, and feature extraction."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("audio_analysis", "Analyze audio files and extract features.")
        from .base import SAFE_ROOT
        self.safe_root = safe_root or SAFE_ROOT
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_file": {
                        "type": "string",
                        "description": "Path to the audio file to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["spectrum", "spectrogram", "features", "all"],
                        "description": "Type of analysis to perform",
                        "default": "spectrum"
                    },
                    "fft_size": {
                        "type": "integer",
                        "description": "FFT size for frequency analysis",
                        "default": 2048
                    },
                    "hop_length": {
                        "type": "integer",
                        "description": "Hop length for spectrogram analysis",
                        "default": 512
                    },
                    "save_plots": {
                        "type": "boolean",
                        "description": "Save analysis plots as images",
                        "default": True
                    }
                },
                "required": ["audio_file"]
            }
        }
    
    def is_available(self) -> bool:
        """Check if required libraries are available."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            return False
    
    async def execute(self, audio_file: str, analysis_type: str = "spectrum",
                     fft_size: int = 2048, hop_length: int = 512,
                     save_plots: bool = True, **_) -> Dict[str, Any]:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import wave
        except ImportError:
            return {"success": False, "error": "NumPy and Matplotlib required. Install with: pip install numpy matplotlib"}
        
        try:
            # Load audio file
            audio_path = _safe_path(audio_file)
            if not audio_path.exists():
                return {"success": False, "error": f"Audio file not found: {audio_file}"}
            
            # Read WAV file
            with wave.open(str(audio_path), 'r') as wav_file:
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                num_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(num_frames)
            
            # Convert to numpy array
            if wav_file.getsampwidth() == 2:  # 16-bit
                samples = np.frombuffer(audio_data, dtype=np.int16)
            else:
                return {"success": False, "error": "Only 16-bit WAV files supported"}
            
            # Convert to mono if stereo
            if num_channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            
            # Normalize
            samples = samples.astype(np.float32) / 32768.0
            
            results = {
                "success": True,
                "audio_file": str(audio_path),
                "sample_rate": sample_rate,
                "duration": len(samples) / sample_rate,
                "num_samples": len(samples),
                "num_channels": num_channels
            }
            
            if analysis_type in ["spectrum", "all"]:
                spectrum_result = await self._analyze_spectrum(samples, sample_rate, fft_size)
                results["spectrum"] = spectrum_result
                
                if save_plots:
                    plot_path = await self._plot_spectrum(spectrum_result, audio_path.stem)
                    results["spectrum"]["plot_file"] = plot_path
            
            if analysis_type in ["spectrogram", "all"]:
                spectrogram_result = await self._analyze_spectrogram(samples, sample_rate, fft_size, hop_length)
                results["spectrogram"] = spectrogram_result
                
                if save_plots:
                    plot_path = await self._plot_spectrogram(spectrogram_result, audio_path.stem, sample_rate)
                    results["spectrogram"]["plot_file"] = plot_path
            
            if analysis_type in ["features", "all"]:
                features_result = await self._extract_features(samples, sample_rate)
                results["features"] = features_result
            
            return results
            
        except Exception as e:
            logger.exception("AudioAnalysisTool error")
            return {"success": False, "error": str(e)}
    
    async def _analyze_spectrum(self, samples: 'np.ndarray', sample_rate: int, fft_size: int) -> Dict[str, Any]:
        """Analyze frequency spectrum using FFT."""
        import numpy as np
        
        # Apply windowing
        windowed_samples = samples[:fft_size] * np.hanning(min(len(samples), fft_size))
        if len(windowed_samples) < fft_size:
            windowed_samples = np.pad(windowed_samples, (0, fft_size - len(windowed_samples)))
        
        # Compute FFT
        fft = np.fft.rfft(windowed_samples)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Frequency bins
        freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_freq_idx]
        
        return {
            "fft_size": fft_size,
            "frequencies": freqs.tolist(),
            "magnitude": magnitude.tolist(),
            "phase": phase.tolist(),
            "dominant_frequency": float(dominant_freq),
            "dominant_magnitude": float(magnitude[dominant_freq_idx])
        }
    
    async def _analyze_spectrogram(self, samples: 'np.ndarray', sample_rate: int, 
                                  fft_size: int, hop_length: int) -> Dict[str, Any]:
        """Generate spectrogram using STFT."""
        import numpy as np
        
        # Compute STFT
        window = np.hanning(fft_size)
        spectrogram = []
        
        for i in range(0, len(samples) - fft_size + 1, hop_length):
            windowed_frame = samples[i:i+fft_size] * window
            fft = np.fft.rfft(windowed_frame)
            magnitude = np.abs(fft)
            spectrogram.append(magnitude)
        
        spectrogram = np.array(spectrogram).T
        
        # Time and frequency axes
        times = np.arange(spectrogram.shape[1]) * hop_length / sample_rate
        freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        return {
            "spectrogram": spectrogram.tolist(),
            "times": times.tolist(),
            "frequencies": freqs.tolist(),
            "fft_size": fft_size,
            "hop_length": hop_length
        }
    
    async def _extract_features(self, samples: 'np.ndarray', sample_rate: int) -> Dict[str, Any]:
        """Extract basic audio features."""
        import numpy as np
        
        # Time domain features
        rms = np.sqrt(np.mean(samples**2))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(samples))))
        
        # Spectral features
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(samples), 1/sample_rate)
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Spectral rolloff (85% of energy)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
        spectral_rolloff = freqs[rolloff_idx]
        
        return {
            "rms_energy": float(rms),
            "zero_crossing_rate": float(zero_crossing_rate),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "dynamic_range": float(np.max(samples) - np.min(samples))
        }
    
    async def _plot_spectrum(self, spectrum_data: Dict, filename_base: str) -> str:
        """Plot frequency spectrum."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 6))
        freqs = np.array(spectrum_data["frequencies"])
        magnitude = np.array(spectrum_data["magnitude"])
        
        plt.subplot(1, 2, 1)
        plt.plot(freqs, 20 * np.log10(magnitude + 1e-10))  # Convert to dB
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Frequency Spectrum')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        phase = np.array(spectrum_data["phase"])
        plt.plot(freqs, phase)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.title('Phase Spectrum')
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_path = _safe_path(f"{filename_base}_spectrum.png")
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _plot_spectrogram(self, spectrogram_data: Dict, filename_base: str, sample_rate: int) -> str:
        """Plot spectrogram."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        spectrogram = np.array(spectrogram_data["spectrogram"])
        times = np.array(spectrogram_data["times"])
        freqs = np.array(spectrogram_data["frequencies"])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(20 * np.log10(spectrogram + 1e-10), 
                  aspect='auto', origin='lower', 
                  extent=[times[0], times[-1], freqs[0], freqs[-1]])
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        
        plot_path = _safe_path(f"{filename_base}_spectrogram.png")
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


class MIDITool(Tool):
    """MIDI file generation and parsing."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("midi_tool", "Generate and parse MIDI files.")
        from .base import SAFE_ROOT
        self.safe_root = safe_root or SAFE_ROOT
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["generate", "parse"],
                        "description": "Operation to perform",
                        "default": "generate"
                    },
                    "midi_file": {
                        "type": "string",
                        "description": "MIDI file path (for parsing)"
                    },
                    "notes": {
                        "type": "array",
                        "description": "Array of notes for generation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "note": {"type": "integer", "description": "MIDI note number (0-127)"},
                                "velocity": {"type": "integer", "description": "Note velocity (0-127)"},
                                "start_time": {"type": "number", "description": "Start time in beats"},
                                "duration": {"type": "number", "description": "Duration in beats"}
                            }
                        }
                    },
                    "tempo": {
                        "type": "integer",
                        "description": "Tempo in BPM",
                        "default": 120
                    },
                    "time_signature": {
                        "type": "array",
                        "description": "Time signature [numerator, denominator]",
                        "default": [4, 4]
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output MIDI file path",
                        "default": "generated.mid"
                    }
                },
                "required": []
            }
        }
    
    async def execute(self, operation: str = "generate", midi_file: Optional[str] = None,
                     notes: Optional[List[Dict]] = None, tempo: int = 120,
                     time_signature: List[int] = None, output_file: str = "generated.mid",
                     **_) -> Dict[str, Any]:
        if time_signature is None:
            time_signature = [4, 4]
        
        try:
            if operation == "generate":
                return await self._generate_midi(notes or [], tempo, time_signature, output_file)
            elif operation == "parse":
                if not midi_file:
                    return {"success": False, "error": "midi_file required for parse operation"}
                return await self._parse_midi(midi_file)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.exception("MIDITool error")
            return {"success": False, "error": str(e)}
    
    async def _generate_midi(self, notes: List[Dict], tempo: int, 
                           time_signature: List[int], output_file: str) -> Dict[str, Any]:
        """Generate MIDI file from note data."""
        # Simple MIDI generation without external dependencies
        output_path = _safe_path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Basic MIDI file structure
        midi_data = bytearray()
        
        # MIDI header
        midi_data.extend(b'MThd')  # Header chunk type
        midi_data.extend((6).to_bytes(4, 'big'))  # Header length
        midi_data.extend((0).to_bytes(2, 'big'))  # Format type 0
        midi_data.extend((1).to_bytes(2, 'big'))  # Number of tracks
        midi_data.extend((480).to_bytes(2, 'big'))  # Time division (ticks per quarter note)
        
        # Track header
        track_data = bytearray()
        
        # Set tempo
        tempo_microseconds = int(60000000 / tempo)
        track_data.extend(b'\x00\xFF\x51\x03')  # Meta event: Set Tempo
        track_data.extend(tempo_microseconds.to_bytes(3, 'big'))
        
        # Set time signature
        track_data.extend(b'\x00\xFF\x58\x04')  # Meta event: Time Signature
        track_data.extend(bytes([time_signature[0], time_signature[1], 24, 8]))
        
        # Add notes
        current_time = 0
        for note_data in notes:
            note = note_data.get('note', 60)
            velocity = note_data.get('velocity', 64)
            start_time = int(note_data.get('start_time', 0) * 480)  # Convert beats to ticks
            duration = int(note_data.get('duration', 1) * 480)
            
            # Delta time to note on
            delta = start_time - current_time
            track_data.extend(self._encode_variable_length(delta))
            track_data.extend(bytes([0x90, note, velocity]))  # Note on
            current_time = start_time
            
            # Delta time to note off
            track_data.extend(self._encode_variable_length(duration))
            track_data.extend(bytes([0x80, note, 0]))  # Note off
            current_time += duration
        
        # End of track
        track_data.extend(b'\x00\xFF\x2F\x00')
        
        # Track chunk
        midi_data.extend(b'MTrk')
        midi_data.extend(len(track_data).to_bytes(4, 'big'))
        midi_data.extend(track_data)
        
        # Write file
        with open(output_path, 'wb') as f:
            f.write(midi_data)
        
        return {
            "success": True,
            "operation": "generate_midi",
            "output_file": str(output_path),
            "file_size": len(midi_data),
            "num_notes": len(notes),
            "tempo": tempo,
            "time_signature": time_signature
        }
    
    def _encode_variable_length(self, value: int) -> bytes:
        """Encode integer as MIDI variable-length quantity."""
        result = bytearray()
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))
    
    async def _parse_midi(self, midi_file: str) -> Dict[str, Any]:
        """Parse MIDI file and extract basic information."""
        midi_path = _safe_path(midi_file)
        if not midi_path.exists():
            return {"success": False, "error": f"MIDI file not found: {midi_file}"}
        
        try:
            with open(midi_path, 'rb') as f:
                data = f.read()
            
            # Parse header
            if data[:4] != b'MThd':
                return {"success": False, "error": "Invalid MIDI file format"}
            
            header_length = int.from_bytes(data[4:8], 'big')
            format_type = int.from_bytes(data[8:10], 'big')
            num_tracks = int.from_bytes(data[10:12], 'big')
            time_division = int.from_bytes(data[12:14], 'big')
            
            return {
                "success": True,
                "operation": "parse_midi",
                "midi_file": str(midi_path),
                "file_size": len(data),
                "format_type": format_type,
                "num_tracks": num_tracks,
                "time_division": time_division,
                "header_length": header_length
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error parsing MIDI file: {str(e)}"}