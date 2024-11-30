# Real-Time Audio Transcription System

A Python-based real-time audio transcription system that captures system audio via BlackHole and provides high-quality speech-to-text conversion using OpenAI's Whisper models.

## Features

- Real-time audio transcription with minimal latency
- Multiple quality levels based on available hardware
- Support for CUDA GPUs and Apple Silicon (MPS)
- Sophisticated audio processing pipeline
- Advanced logging system with visual feedback
- Automatic model selection based on system capabilities
- Voice activity detection to minimize processing
- Clean, formatted transcription output

## Prerequisites

### Hardware Requirements
- Minimum 4GB RAM (8GB+ recommended)
- CUDA-capable GPU, Apple Silicon, or modern CPU
- Audio input/output capability

### Software Requirements
- Python 3.8 or higher
- BlackHole 2ch virtual audio interface
- pip (Python package manager)

## Installation

1. Install BlackHole 2ch:
   ```bash
   brew install blackhole-2ch  # macOS with Homebrew
   # Or download from https://existential.audio/blackhole/
   ```

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

### Audio Setup
1. Open System Settings > Sound
2. Set output device to "BlackHole 2ch"
3. Configure your audio application to output to BlackHole
4. Verify audio routing using the built-in level monitor

### Model Selection
The system automatically selects the appropriate Whisper model based on your hardware:

- High Quality (12GB+ memory):
  - Model: whisper-medium
  - Best accuracy, larger vocabulary
  - Suitable for CUDA GPUs with 12GB+ VRAM

- Medium Quality (8GB+ memory):
  - Model: whisper-small
  - Good balance of accuracy and speed
  - Suitable for CUDA GPUs with 8GB+ VRAM or Apple Silicon

- Base Quality (4GB+ memory):
  - Model: whisper-base
  - Fast, reasonable accuracy
  - Suitable for lower-end GPUs or modern CPUs

- Minimal Quality (2GB+ memory):
  - Model: whisper-tiny
  - Fastest, basic accuracy
  - Suitable for any system

## Usage

1. Start the transcription system:
   ```bash
   python main.py
   ```

2. The system will:
   - Detect available hardware
   - Select appropriate model
   - Initialize audio pipeline
   - Begin displaying transcriptions

3. Monitor the output:
   - Real-time audio levels
   - Transcribed text with confidence scores
   - System status and errors

4. Logs are saved to:
   - `logs/transcription_[timestamp].log` - Human-readable transcripts
   - `logs/transcription_[timestamp].jsonl` - Machine-parseable format
   - `logs/system_[timestamp].log` - System events and errors

## Advanced Configuration

### Environment Variables
- `TRANSCRIPTION_MODEL`: Override automatic model selection
- `DEVICE`: Force specific device (cuda/mps/cpu)
- `LOG_LEVEL`: Set logging verbosity (DEBUG/INFO/WARNING/ERROR)

### Audio Parameters
- Sample Rate: 16kHz
- Chunk Size: Varies by model (1.5-5 seconds)
- Overlap: 1 second
- VAD Aggressiveness: 1 (adjustable 0-3)

## Known Issues

- First-time model download may be slow
- MPS (Apple Silicon) performance varies by model size
- High CPU usage with larger models on CPU-only systems
- Occasional audio buffer underruns with small chunk sizes
- Some accents may require model fine-tuning

## To Do

### High Priority
- [ ] Implement pause/resume functionality
- [ ] Add speaker diarization
- [ ] Improve error recovery
- [ ] Add model caching
- [ ] Optimize memory usage

### Medium Priority
- [ ] Add GUI interface
- [ ] Support more audio interfaces
- [ ] Add custom vocabulary support
- [ ] Implement batch processing
- [ ] Add export functionality

### Low Priority
- [ ] Add more language support
- [ ] Create model fine-tuning pipeline
- [ ] Add audio preprocessing options
- [ ] Implement streaming output
- [ ] Create comprehensive tests

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper models
- ExistentialAudio for BlackHole
- The PyTorch and transformers communities

## Support

For issues and questions:
1. Check the logs in the `logs` directory
2. Review known issues above
3. Open an issue on GitHub with:
   - System specifications
   - Log files
   - Steps to reproduce 