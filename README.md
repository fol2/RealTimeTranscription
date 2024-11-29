# Audio Monitor and Speech Recognition Tool

A Python-based tool that combines BlackHole audio monitoring with SpeechBrain speech recognition capabilities. This utility helps verify audio routing and process real-time speech input.

## Introduction

This tool provides a comprehensive audio processing pipeline:
- Detect and monitor BlackHole 2ch audio interface
- Route system audio for speech recognition
- Process audio using SpeechBrain for speech-to-text conversion
- Monitor real-time audio levels and quality

## Prerequisites

- Python 3.7 or higher
- BlackHole 2ch virtual audio interface installed
- pip (Python package manager)
- Sufficient disk space for SpeechBrain models

## Installation

1. Install BlackHole 2ch from [ExistentialAudio/BlackHole](https://github.com/ExistentialAudio/BlackHole)

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required SpeechBrain models (will be downloaded automatically on first run)

## Components

### Audio Monitoring (`test_blackhole.py`)
- Lists available audio devices
- Detects BlackHole 2ch interface
- Displays real-time audio levels

### Speech Recognition
- Uses SpeechBrain for speech-to-text conversion
- Processes audio input from BlackHole
- Supports multiple languages and models

## Usage

1. Configure BlackHole as your system audio output:
   - Open System Preferences > Sound
   - Set output device to "BlackHole 2ch"

2. Run the audio monitor to verify routing:
   ```bash
   python test_blackhole.py
   ```

3. Run speech recognition (once audio routing is confirmed):
   ```bash
   python test_audio.py
   ```

4. The system will:
   - Monitor audio levels
   - Process speech through SpeechBrain
   - Output recognized text

## Audio Pipeline

```
System Audio → BlackHole 2ch → Audio Monitor → SpeechBrain → Text Output
```

## Known Issues

- BlackHole device might not be detected if not properly installed
- Audio permissions might need to be granted on first run
- Sample rate mismatches can occur if system audio is configured differently
- Volume normalization might need adjustment for different audio sources
- SpeechBrain model download might fail on slow connections
- Speech recognition accuracy depends on audio quality

## To Do

High Priority:
- [ ] Implement real-time speech-to-text translation
  - [ ] Continuous audio stream processing
  - [ ] Optimize SpeechBrain inference for real-time performance
  - [ ] Add support for multiple languages
  - [ ] Minimize latency in translation pipeline
  - [ ] Add buffer management for audio stream
  - [ ] Implement threading for non-blocking audio processing

Medium Priority:
- [ ] Add GUI interface
- [ ] Add configuration file support
- [ ] Implement logging functionality
- [ ] Add audio quality preprocessing
- [ ] Create comprehensive testing suite

Lower Priority:
- [ ] Support for additional virtual audio interfaces
- [ ] Add tests for different audio scenarios
- [ ] Improve error messages and handling
- [ ] Add support for custom sample rates
- [ ] Create installation verification tool

## Current Development Focus

The main focus is on implementing real-time translation with the following goals:
1. Achieve low-latency translation (target < 500ms)
2. Support multiple languages
3. Maintain high accuracy while processing continuous audio
4. Provide clear visual feedback of translation status
5. Allow easy language switching during runtime

## Roadmap for Real-Time Translation

1. Phase 1: Basic Integration
   - Connect audio stream to SpeechBrain
   - Implement basic real-time processing
   - Add simple text output

2. Phase 2: Performance Optimization
   - Optimize buffer sizes
   - Implement threading
   - Reduce latency

3. Phase 3: Features
   - Add multiple language support
   - Implement language switching
   - Add confidence scores

4. Phase 4: UI/UX
   - Add progress indicators
   - Implement status monitoring
   - Create user controls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here] 