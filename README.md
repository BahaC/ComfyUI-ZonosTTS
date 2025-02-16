# ComfyUI Zonos TTS Node

A ComfyUI custom node that brings Zonos Text-to-Speech capabilities to your workflows, featuring high-quality speech synthesis and voice cloning.

## Features

- 🎯 High-quality text-to-speech synthesis
- 🗣️ Voice cloning from reference audio
- 💾 Local model caching for faster loading
- 🎚️ Advanced parameter control for speech generation
- 🌍 Support for English and Japanese
- ⚡ Multiple model architectures (Transformer/Hybrid)

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/BahaC/ComfyUI-ZonosTTS.git Zonos
```

2. Install the requirements:
```bash
cd ComfyUI-ZonosTTS
pip install -r requirements.txt
```

## Node Usage

### Zonos Text to Speech
The node provides a simple interface for text-to-speech conversion with advanced options:

#### Inputs
- `text`: Input text to synthesize (String)
- `language`: Language code selection (en-us, ja-jp)
- `model_name`: Choice of model architecture:
  - `Zyphra/Zonos-v0.1-transformer`: Faster, lighter model
  - `Zyphra/Zonos-v0.1-hybrid`: Higher quality (requires additional dependencies)
- `audio_file`: Reference audio for voice cloning (optional)
- `cfg_scale`: Control over generation quality (1.0 - 10.0)

#### Output
- `audio_path`: Path to the generated WAV file

## Model Management

Models are automatically downloaded and cached in:
```
/workspace/ComfyUI/models/TTS/Zonos/
```

The node implements smart model caching:
- First run: Downloads and caches the model
- Subsequent runs: Uses cached model for faster loading
- Automatic model switching when changing architectures

## Example Workflows

### Basic Text to Speech
```
[Text Input] -> [Zonos TTS] -> [Audio Output]
```

### Voice Cloning
```
[Text Input] -> [Zonos TTS] <- [Audio File] == [Audio File]
```

## Configuration

### Audio Output
Generated audio files are saved with unique timestamps:
```
output/zonos_YYYYMMDD-HHMMSS_UUID.wav
```

### Model Settings
- **Transformer Model**
  - Faster inference
  - Lower resource requirements
  - Good for most use cases

- **Hybrid Model**
  - Higher quality output
  - Requires additional dependencies
  - More resource intensive

## Requirements

- Python >= 3.10
- torch >= 2.0.0
- torchaudio >= 2.0.0
- safetensors >= 0.3.0
- huggingface_hub >= 0.16.0
- Additional dependencies in requirements.txt

## Troubleshooting

### Common Issues
1. **Model Download Fails**
   - Check your internet connection
   - Ensure you have sufficient disk space
   - Try manually downloading to the models directory

2. **Voice Cloning Issues**
   - Ensure reference audio is clean and contains only speech
   - Use WAV format for reference audio
   - Keep reference audio under 30 seconds

3. **CUDA Out of Memory**
   - Try using the transformer model instead of hybrid
   - Reduce batch size or audio length
   - Free up GPU memory from other applications

## Credits

- Zonos TTS by [Zyphra](https://github.com/Zyphra/Zonos)

## License

This project is licensed under the terms of the LICENSE file included in the repository. 
