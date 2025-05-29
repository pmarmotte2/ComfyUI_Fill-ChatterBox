import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional

# Import directly from the chatterbox package
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

# Monkey patch torch.load to always use CPU if needed
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class FL_ChatterboxTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, audio_prompt=None, use_cpu=False):
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if use_cpu:
            message = "Using CPU for inference (CUDA disabled)"
        else:
            message = f"Using {device} for inference"
        
        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning: {audio_prompt_path}"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    message += f"\nWarning: Audio prompt file was not created properly"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        try:
            # Load the TTS model
            message += f"\nLoading TTS model on {device}..."
            tts_model = ChatterboxTTS.from_pretrained(device=device)
            
            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            
        except RuntimeError as e:
            if "CUDA" in str(e) and device != "cpu":
                message += "\nCUDA error detected during TTS. Falling back to CPU..."
                # Try again with CPU
                device = "cpu"
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
            else:
                # Re-raise if it's not a CUDA error or we're already on CPU
                message += f"\nError during TTS: {str(e)}"
                # Return empty audio data
                empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
                # Clean up any temporary files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                return (empty_audio, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        # Create audio data structure for the output
        audio_data = {
            "waveform": wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": tts_model.sr
        }
        
        message += f"\nSpeech generated successfully"
        
        return (audio_data, message)

# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(self, input_audio, target_voice, use_cpu=False):
        """
        Convert the voice in an audio file to match a target voice.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if use_cpu:
            message = "Using CPU for inference (CUDA disabled)"
        else:
            message = f"Using {device} for inference"
        
        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            input_audio_path = temp_input.name
            temp_files.append(input_audio_path)
        
        # Save the input audio to the temporary file
        input_waveform = input_audio['waveform'].squeeze(0)
        torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
        
        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)
        
        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
        
        try:
            # Load the VC model
            message += f"\nLoading VC model on {device}..."
            vc_model = ChatterboxVC.from_pretrained(device=device)
            
            # Convert voice
            message += f"\nConverting voice to match target voice"
            
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            
        except RuntimeError as e:
            if "CUDA" in str(e) and device != "cpu":
                message += "\nCUDA error detected during VC. Falling back to CPU..."
                # Try again with CPU
                device = "cpu"
                vc_model = ChatterboxVC.from_pretrained(device=device)
                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
            else:
                # Re-raise if it's not a CUDA error or we're already on CPU
                message += f"\nError during VC: {str(e)}"
                # Return the original audio
                message += f"\nError: {str(e)}"
                return (input_audio, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        # Create audio data structure for the output
        audio_data = {
            "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": vc_model.sr
        }
        
        message += f"\nVoice converted successfully"
        
        return (audio_data, message)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxTTS": FL_ChatterboxTTSNode,
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxTTS": "FL Chatterbox TTS",
    "FL_ChatterboxVC": "FL Chatterbox VC",
}