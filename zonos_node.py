import os
import torch
import torchaudio
import uuid
from datetime import datetime
from .zonos.model import Zonos
from .zonos.conditioning import make_cond_dict
from .zonos.utils import DEFAULT_DEVICE
import json
import safetensors



class ZonosTextToSpeech:
    def __init__(self):
        self.device = DEFAULT_DEVICE
        self.model = None
        self.current_model_name = None
        self.models_dir = os.path.join(os.getcwd(), 'models', 'TTS', 'Zonos')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def get_local_model_path(self, model_name):
        # Convert HF model name to local path
        model_folder = model_name.split('/')[-1]
        return os.path.join(self.models_dir, model_folder)
        
    def load_model(self, model_name):
        try:
            local_path = self.get_local_model_path(model_name)
            config_path = os.path.join(local_path, 'config.json')
            weights_path = os.path.join(local_path, 'model.safetensors')
            
            # If either config or weights don't exist, download the model
            if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                print(f"Downloading model {model_name} to {local_path}")
                model = Zonos.from_pretrained(model_name, device=self.device)
                
                # Save the model locally
                os.makedirs(local_path, exist_ok=True)
                
                # Save config
                config_dict = {
                    "backbone": model.config.backbone.__dict__,
                    "prefix_conditioner": model.config.prefix_conditioner.__dict__,
                    "eos_token_id": model.config.eos_token_id,
                    "masked_token_id": model.config.masked_token_id
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Save weights
                state_dict = model.state_dict()
                safetensors.torch.save_file(state_dict, weights_path)
                
                return model
            else:
                print(f"Loading model from {local_path}")
                # Load from local path
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                from zonos.config import BackboneConfig, PrefixConditionerConfig, ZonosConfig
                
                # Reconstruct config objects
                backbone_config = BackboneConfig(**config_dict["backbone"])
                prefix_conditioner_config = PrefixConditionerConfig(**config_dict["prefix_conditioner"])
                
                config = ZonosConfig(
                    backbone=backbone_config,
                    prefix_conditioner=prefix_conditioner_config,
                    eos_token_id=config_dict["eos_token_id"],
                    masked_token_id=config_dict["masked_token_id"]
                )
                
                model = Zonos(config)
                
                # Load model weights
                state_dict = safetensors.torch.load_file(weights_path)
                model.load_state_dict(state_dict)
                return model.to(self.device)
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to direct download")
            # If anything fails, fall back to direct download without saving
            return Zonos.from_pretrained(model_name, device=self.device)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello, world!"}),
                "language": (["en-us", "ja-jp"], {"default": "en-us"}),
                "model_name": (["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"], {"default": "Zyphra/Zonos-v0.1-transformer"}),
                "audio_file": ("STRING", {"default": ""}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"

    def generate_speech(self, text, language, model_name, audio_file, cfg_scale):
        if self.model is None or self.current_model_name != model_name:
            self.model = self.load_model(model_name)
            self.current_model_name = model_name
        
        # Process speaker embedding if audio file is provided
        if audio_file and os.path.exists(audio_file):
            wav, sampling_rate = torchaudio.load(audio_file)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
        else:
            speaker = None
        
        # Create conditioning
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = self.model.prepare_conditioning(cond_dict)
        
        # Generate audio
        codes = self.model.generate(conditioning, cfg_scale=cfg_scale)
        wavs = self.model.autoencoder.decode(codes).cpu()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"zonos_{timestamp}_{unique_id}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Save the generated audio
        torchaudio.save(output_path, wavs[0], self.model.autoencoder.sampling_rate)
        
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "ZonosTextToSpeech": ZonosTextToSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZonosTextToSpeech": "Lisa Zonos Text to Speech",
} 