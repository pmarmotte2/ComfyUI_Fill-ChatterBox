# ComfyUI_Fill-ChatterBox

A custom node extension for ComfyUI that adds text-to-speech (TTS) and voice conversion (VC) capabilities using the Chatterbox library.

![ChatterBox Example](web/image.png)

## Features

- **Text-to-Speech (TTS)**: Convert text to natural-sounding speech with adjustable emotion, pace, and randomness
- **Voice Conversion (VC)**: Transform voice characteristics between audio samples
- **ComfyUI Integration**: Custom styled nodes with purple background and teal text

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI_Fill-ChatterBox.git
   ```

2. Install the base dependencies:
   ```bash
   pip install -r ComfyUI_Fill-ChatterBox/requirements.txt
   ```

3. Install chatterbox-tts WITHOUT its dependencies:
   ```bash
   pip install chatterbox-tts --no-deps
   ```

   ⚠️ The `--no-deps` flag is crucial to prevent conflicts with ComfyUI's PyTorch installation!

## Usage

### Text-to-Speech Node (FL Chatterbox TTS)
- Add the "FL Chatterbox TTS" node to your workflow
- Configure text input and parameters (exaggeration, cfg_weight, temperature)
- Optionally provide an audio prompt for voice cloning

### Voice Conversion Node (FL Chatterbox VC)
- Add the "FL Chatterbox VC" node to your workflow
- Connect input audio and target voice
- Both nodes support CPU fallback if CUDA errors occur

## Requirements

- ComfyUI installation
- Python 3.8+
- CUDA-compatible GPU recommended (but not required)