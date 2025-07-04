# ComfyUI_Fill-ChatterBox

If you enjoy this project, consider supporting me on Patreon!
<p align="left">
  <a href="https://www.patreon.com/c/Machinedelusions">
    <img src="assets/Patreon.png" width="150px" alt="Patreon">
  </a>
</p>

A custom node extension for ComfyUI that adds text-to-speech (TTS) and voice conversion (VC) capabilities using the Chatterbox library.
Supports a MAXIMUM of 40 seconds. Iv tried removing this limitation, but the model falls apart really badly with anything longer than that, so it remains.

![ChatterBox Example](web/image.png)

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/filliptm/ComfyUI_Fill-ChatterBox.git
   ```

2. Install the base dependencies:
   ```bash
   pip install -r ComfyUI_Fill-ChatterBox/requirements.txt
   ```

3. (Optional) Install watermarking support:
   ```bash
   pip install resemble-perth
   ```
   **Note**: The `resemble-perth` package may have compatibility issues with Python 3.12+. If you encounter import errors, the nodes will still function without watermarking.


## Usage

### Text-to-Speech Node (FL Chatterbox TTS)
- Add the "FL Chatterbox TTS" node to your workflow
- Configure text input and parameters (exaggeration, cfg_weight, temperature)
- Optionally provide an audio prompt for voice cloning

### Voice Conversion Node (FL Chatterbox VC)
- Add the "FL Chatterbox VC" node to your workflow
- Connect input audio and target voice
- Both nodes support CPU fallback if CUDA errors occur

### Dialog TTS Node (FL Chatterbox Dialog TTS)
- Add the "FL Chatterbox Dialog TTS" node to your workflow.
- This node is designed to synthesize speech for dialogs with two distinct speakers (SPEAKER A and SPEAKER B).
- **Inputs:**
    - `dialog_text`: A multiline string where each line is prefixed by `SPEAKER A:` or `SPEAKER B:`. For example:
      ```
      SPEAKER A: Hello, how are you?
      SPEAKER B: I am fine, thank you!
      SPEAKER A: That's good to hear.
      ```
    - `speaker_a_prompt`: An audio prompt (AUDIO type) for SPEAKER A's voice.
    - `speaker_b_prompt`: An audio prompt (AUDIO type) for SPEAKER B's voice.
    - `exaggeration`: Controls emotion intensity (0.25-2.0).
    - `cfg_weight`: Controls pace/classifier-free guidance (0.2-1.0).
    - `temperature`: Controls randomness in generation (0.05-5.0).
    - `use_cpu` (optional): Boolean, defaults to False. Forces CPU usage.
    - `keep_model_loaded` (optional): Boolean, defaults to False. Keeps the model loaded in memory.
- The node will generate a single audio file with the dialog spoken by the respective voices.

## Change Log

### 6/24/2025
- Added seed parameter to both TTS and VC nodes for reproducible generation
- Seed range: 0 to 4,294,967,295 (32-bit integer)
- Enables consistent audio output for debugging and workflow control
- Made Perth watermarking optional to fix Python 3.12+ compatibility issues
- Nodes now function without watermarking if resemble-perth import fails

### 5/31/2025
- Added Persistent model loading, and loading bar functionality
- Added Mac support (needs to be tested so HMU)
- removed the chatterbox-tts library and implemented native inference code.

