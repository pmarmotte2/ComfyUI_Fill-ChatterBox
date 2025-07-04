import os
import torch
import torchaudio
import tempfile

from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from comfy.utils import ProgressBar

class FL_ChatterboxDialogTTSNode:
    """
    TTS Node that accepts dialog with speaker labels and generates audio using separate voice prompts.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialog_text": ("STRING", {"multiline": True}),
                "speaker_a_prompt": ("AUDIO",),
                "speaker_b_prompt": ("AUDIO",),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_dialog"
    CATEGORY = "ChatterBox"

    _model = None
    _device = None

    def generate_dialog(self, dialog_text, speaker_a_prompt, speaker_b_prompt,
                        exaggeration, cfg_weight, temperature,
                        use_cpu=False, keep_model_loaded=False):
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        pbar = ProgressBar(100)
        message = f"Running on {device}"

        def save_temp_audio(audio_data):
            path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            torchaudio.save(path, audio_data['waveform'].squeeze(0), audio_data['sample_rate'])
            return path

        prompt_a_path = save_temp_audio(speaker_a_prompt)
        prompt_b_path = save_temp_audio(speaker_b_prompt)
        temp_files = [prompt_a_path, prompt_b_path]

        if self._model is None or self._device != device:
            self._model = ChatterboxTTS.from_pretrained(device=device)
            self._device = device
        tts = self._model

        lines = dialog_text.strip().splitlines()
        waveform_list = []

        for i, line in enumerate(lines):
            if line.startswith("SPEAKER A:"):
                content = line[len("SPEAKER A:"):].strip()
                prompt_path = prompt_a_path
            elif line.startswith("SPEAKER B:"):
                content = line[len("SPEAKER B:"):].strip()
                prompt_path = prompt_b_path
            else:
                continue  # skip malformed line

            pbar.update_absolute(int((i / len(lines)) * 80))

            wav = tts.generate(
                text=content,
                audio_prompt_path=prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            waveform_list.append(wav)

        if not waveform_list:
            return ({"waveform": torch.zeros((1, 1, 1)), "sample_rate": 16000}, "No valid dialog lines found.")

        combined_waveform = torch.cat(waveform_list, dim=-1)
        audio = {"waveform": combined_waveform.unsqueeze(0), "sample_rate": tts.sr}

        for f in temp_files:
            os.unlink(f)

        pbar.update_absolute(100)
        return (audio, "Dialog synthesized successfully.")

NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxDialogTTS": FL_ChatterboxDialogTTSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxDialogTTS": "FL Chatterbox Dialog TTS",
}
