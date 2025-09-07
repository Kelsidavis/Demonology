# Sheet Music OMR Tool

## Overview

The Sheet Music OMR (Optical Music Recognition) tool converts sheet music images and PDFs into digital music formats using Audiveris for OMR processing and music21 for post-processing and cleanup.

## Features

### **Input Formats**
- **Images** - PNG, JPG, GIF, BMP, TIFF
- **PDF** - Multi-page sheet music documents
- **Quality** - Works best with high-contrast, clean sheet music

### **Output Formats**
- **MusicXML** - Industry standard digital music notation
- **MIDI** - Musical Instrument Digital Interface for playback
- **WAV** - Audio rendering (via music21)

### **OMR Processing**
- **Audiveris** - Professional-grade optical music recognition
- **Staff detection** - Automatic staff line identification
- **Symbol recognition** - Notes, rests, clefs, key signatures, time signatures
- **Batch processing** - Automated workflow without GUI interaction

### **Post-Processing**
- **Automatic fixes** - Missing time signatures, key signatures, tempo
- **Manual overrides** - Force specific time signature, key, or tempo
- **Score cleanup** - Quantization, measure creation, notation cleanup
- **Smart detection** - Only applies fixes when elements are missing

## Installation Requirements

### **Audiveris Setup**
```bash
# Download from GitHub releases
wget https://github.com/Audiveris/audiveris/releases/download/5.3.1/audiveris-5.3.1.tar.gz

# Extract and install
tar -xzf audiveris-5.3.1.tar.gz
sudo mv audiveris-5.3.1 /opt/audiveris

# Set environment variable
export AUDIVERIS_BIN=/opt/audiveris/bin/Audiveris
```

### **Python Dependencies**  
```bash
pip install music21>=9.1.0 soundfile>=0.12.1
```

### **System Dependencies**
```bash
# Ubuntu/Debian
sudo apt install libsndfile1 tesseract-ocr

# macOS  
brew install libsndfile tesseract

# Fedora/RHEL
sudo dnf install libsndfile-devel tesseract
```

## Usage Examples

### **Basic OMR Processing**
```python
# Simple image to MusicXML/MIDI
result = await tool.execute(
    input_path="sheet_music.png"
)

# PDF processing
result = await tool.execute(
    input_path="score.pdf",
    output_stem="output/my_score"
)
```

### **With Post-Processing**
```python
# Fix missing time signature and tempo
result = await tool.execute(
    input_path="sheet_music.png",
    timesig="4/4",
    tempo=120,
    autofix_missing=True  # Only fix if missing
)

# Force overrides (replace existing)
result = await tool.execute(
    input_path="sheet_music.png", 
    timesig="3/4",
    key_text="G major",
    tempo=90,
    force_fix=True  # Always override
)
```

### **Advanced Configuration**
```python
# Custom Audiveris binary
result = await tool.execute(
    input_path="sheet_music.png",
    audiveris_bin="/custom/path/to/audiveris",
    output_stem="processed/output"
)

# Key signature examples
result = await tool.execute(
    input_path="sheet_music.png",
    key_text="A minor",      # Tonic + mode
    # OR
    key_text="F# major",     # With accidentals  
    # OR
    key_text="Bb",          # Just tonic (assumes major)
)
```

## Parameter Reference

### **Required Parameters**
- `input_path` - Path to sheet music image or PDF file

### **Optional Parameters**
- `output_stem` - Output file path without extension (default: same as input)
- `audiveris_bin` - Path to Audiveris executable (default: `$AUDIVERIS_BIN` or `"audiveris"`)
- `timesig` - Time signature override (e.g., `"4/4"`, `"3/4"`, `"6/8"`)
- `key_text` - Key signature override (e.g., `"C major"`, `"A minor"`, `"F# major"`)
- `tempo` - Tempo in BPM (e.g., `120`, `90`, `140`)
- `autofix_missing` - Only apply fixes when elements are missing (default: `True`)
- `force_fix` - Always apply fixes, overriding existing elements (default: `False`)

### **Time Signatures**
```python
"4/4"    # Common time
"3/4"    # Waltz time  
"2/4"    # March time
"6/8"    # Compound duple
"9/8"    # Compound triple
"12/8"   # Compound quadruple
"2/2"    # Cut time
"5/4"    # Irregular meter
```

### **Key Signatures**
```python
# Major keys
"C major"    # No accidentals
"G major"    # 1 sharp
"D major"    # 2 sharps  
"F major"    # 1 flat
"Bb major"   # 2 flats

# Minor keys  
"A minor"    # No accidentals
"E minor"    # 1 sharp
"B minor"    # 2 sharps
"D minor"    # 1 flat
"G minor"    # 2 flats

# Short forms (assumes major)
"C"    # C major
"F#"   # F# major
"Bb"   # Bb major
```

## Response Format

### **Successful Processing**
```python
{
    "success": True,
    "stage": "fixed",                    # "audiveris_done" or "fixed"
    "musicxml": "/path/to/output.musicxml",
    "midi": "/path/to/output.mid",       # Only if post-processing applied
    "fix_applied": True,                 # Whether post-processing ran
    "stdout": "Audiveris output...",     # Last 2000 chars
    "stderr": "Any warnings..."          # Last 1000 chars
}
```

### **OMR Failure**
```python
{
    "success": False,
    "stage": "audiveris",
    "stdout": "Full Audiveris output",
    "stderr": "Error messages"
}
```

### **Export Failure**
```python
{
    "success": False, 
    "stage": "export",
    "error": "No MusicXML/MXL produced by Audiveris",
    "stdout": "Audiveris output",
    "stderr": "Audiveris errors"
}
```

### **Post-Processing Failure**
```python
{
    "success": True,
    "stage": "fix_failed",
    "musicxml": "/path/to/raw_output.musicxml", # Raw OMR result
    "midi": None,
    "fix_applied": True,
    "fix_error": "music21 error message"
}
```

## Workflow Details

### **Processing Pipeline**
1. **Validation** - Check input file exists and is readable
2. **OMR Processing** - Run Audiveris with batch export options
3. **Result Location** - Find exported MusicXML/MXL files
4. **Analysis** - Inspect for missing time sig/key/tempo (if `autofix_missing`)  
5. **Post-Processing** - Apply music21 fixes and generate MIDI
6. **Cleanup** - Return file paths and statistics

### **Smart Fixing Logic**
```python
# autofix_missing=True (default)
if missing_timesig and timesig_provided:
    apply_fix()
if missing_key and key_provided:
    apply_fix()
if missing_tempo and tempo_provided:
    apply_fix()

# force_fix=True  
if timesig_provided or key_provided or tempo_provided:
    apply_fix()  # Always override
```

## Troubleshooting

### **Common Issues**

**Audiveris not found**
```bash
export AUDIVERIS_BIN=/opt/audiveris/bin/Audiveris
# OR
ln -s /opt/audiveris/bin/Audiveris /usr/local/bin/audiveris
```

**Poor OMR quality**
- Use high-resolution scans (300+ DPI)
- Ensure good contrast between notes and background
- Remove noise, dust, or artifacts from images
- Consider manual cleanup of input images

**music21 errors**  
```bash
# Install system dependencies
sudo apt install libsndfile1  # Linux
brew install libsndfile       # macOS

# Verify installation
python -c "import music21; print('OK')"
```

**Permission errors**
```bash
# Make sure Audiveris binary is executable
chmod +x /opt/audiveris/bin/Audiveris

# Check directory permissions
ls -la /opt/audiveris/bin/
```

### **Performance Tips**

1. **Batch processing** - Process multiple pages as single PDF when possible
2. **Image preprocessing** - Clean up scans before OMR for better accuracy
3. **Output organization** - Use descriptive output stems for file management
4. **Validation** - Check OMR results before applying fixes
5. **Format selection** - Use MusicXML for editing, MIDI for playback

## Integration Examples

### **With File Processing Pipeline**
```python
# Process entire sheet music library
import os
from pathlib import Path

music_dir = Path("sheet_music")
output_dir = Path("processed")

for file in music_dir.glob("*.png"):
    result = await tool.execute(
        input_path=str(file),
        output_stem=str(output_dir / file.stem),
        timesig="4/4",     # Default assumption
        tempo=120,         # Standard tempo
        autofix_missing=True
    )
    
    if result["success"]:
        print(f"✅ {file.name} → {result['musicxml']}")
    else:
        print(f"❌ {file.name} failed: {result['error']}")
```

### **With Music Analysis**
```python
# OMR + analysis workflow
result = await tool.execute(input_path="score.pdf")

if result["success"] and result.get("midi"):
    # Load MIDI for analysis
    import music21
    score = music21.converter.parse(result["midi"])
    
    # Extract musical features
    key = score.analyze('key')
    tempo = score.metronomeMarkBoundaries()[0][2].number
    time_sig = score.getTimeSignatures()[0]
    
    print(f"Key: {key}")
    print(f"Tempo: {tempo} BPM") 
    print(f"Time: {time_sig}")
```

This tool bridges the gap between physical sheet music and digital music workflows, enabling automated digitization of musical scores with intelligent post-processing capabilities.