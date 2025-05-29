# ComfyUI_Fill-ChatterBox

A custom node extension for ComfyUI that adds advanced text-to-speech (TTS) and voice conversion (VC) capabilities using the Chatterbox library.

## Features

- **Text-to-Speech (TTS)**: Convert text to natural-sounding speech
  - Adjustable emotion intensity, pace, and randomness
  - Voice cloning from audio prompts
  - GPU acceleration with CPU fallback

- **Voice Conversion (VC)**: Transform voice characteristics between audio samples
  - Preserve content while applying target voice style
  - High-quality voice transformation

- **ComfyUI Integration**:
  - Custom styled nodes for easy identification (purple background with teal text)
  - Compatible with ComfyUI's workflow system
  - Connect with other audio and visual nodes

## Installation

### Important Note on Dependencies

This extension uses ComfyUI's existing PyTorch installation to avoid conflicts. The installation process requires special handling for the chatterbox-tts package.

### Installation Steps

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI_Fill-ChatterBox.git
   ```

2. Install the base dependencies:
   ```bash
   pip install -r ComfyUI_Fill-ChatterBox/requirements.txt
   ```

3. **IMPORTANT**: Install chatterbox-tts WITHOUT its dependencies:
   ```bash
   pip install chatterbox-tts --no-deps
   ```

   ⚠️ The `--no-deps` flag is crucial to prevent conflicts with ComfyUI's PyTorch installation!

## Usage

### Text-to-Speech Node (FL Chatterbox TTS)

The TTS node converts text input to speech with various customization options:

1. Add the "FL Chatterbox TTS" node to your workflow
2. Configure the parameters:
   - **text**: The text to convert to speech (supports multiline)
   - **exaggeration**: Controls emotion intensity (0.25-2.0)
   - **cfg_weight**: Controls pace/classifier-free guidance (0.2-1.0)
   - **temperature**: Controls randomness in generation (0.05-5.0)
   - **audio_prompt** (optional): Reference voice for TTS voice cloning
   - **use_cpu** (optional): Force CPU usage even if CUDA is available

3. Connect the output to other audio nodes or save as output

### Voice Conversion Node (FL Chatterbox VC)

The VC node transforms the voice characteristics of input audio to match a target voice:

1. Add the "FL Chatterbox VC" node to your workflow
2. Configure the inputs:
   - **input_audio**: The audio to convert
   - **target_voice**: The voice to match
   - **use_cpu** (optional): Force CPU usage even if CUDA is available

3. Connect the output to other audio nodes or save as output

## Technical Details

### GPU/CPU Handling

Both nodes intelligently handle GPU/CPU selection:
- Automatically detect CUDA availability
- Allow forcing CPU usage via the use_cpu parameter
- Graceful fallback to CPU if CUDA errors occur

### Temporary File Management

The nodes use temporary files for audio processing:
- Creates temporary WAV files for audio processing
- Ensures proper cleanup even if errors occur
- Provides detailed status messages about file operations

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter CUDA out of memory errors:
1. Try enabling the "use_cpu" option in the node settings
2. The nodes will automatically fall back to CPU if CUDA errors occur

### Installation Issues

If you encounter issues with conflicting PyTorch versions:
1. Ensure you installed chatterbox-tts with the `--no-deps` flag
2. Try uninstalling and reinstalling with the correct flags:
   ```bash
   pip uninstall -y chatterbox-tts
   pip install chatterbox-tts --no-deps
   ```

## Requirements

- ComfyUI installation
- Python 3.8+
- CUDA-compatible GPU recommended (but not required)