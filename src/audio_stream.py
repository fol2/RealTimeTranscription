import queue
import threading
import numpy as np
import logging
import pyaudio
import time
from typing import Optional, Tuple
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, sample_rate=None, channels=None, chunk_size=None):
        """Initialize audio stream with configurable settings."""
        # Load configuration
        self.config = ConfigManager().get_audio_stream_config()
        
        # Use provided values or defaults from config
        self.sample_rate = sample_rate or self.config['default_sample_rate']
        self.channels = channels or self.config['default_channels']
        self.chunk_size = chunk_size or self.config['default_chunk_size']
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self._stream = None
        self._audio = None
        self._stream_thread = None
        self.audio_detected = False
        
    def _find_blackhole(self) -> Tuple[int, dict]:
        """Find BlackHole 2ch device and return its index and info."""
        audio = pyaudio.PyAudio()
        device_index = None
        device_info = None
        
        # List all audio devices
        if self.config['show_device_scan']:
            logger.debug("Scanning for audio devices...")
            
        # Look for configured device name
        target_device = self.config['device_name']
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if target_device in dev_info["name"]:
                device_index = i
                device_info = dev_info
                if self.config['show_device_scan']:
                    logger.info(f"✓ Found {target_device}")
                    logger.debug(f"Device info: {dev_info}")
                break
            
        if device_index is None:
            raise RuntimeError(f"Could not find {target_device} device")
            
        return device_index, device_info
        
    def start(self, device=None):
        """Start the audio streaming."""
        if self.is_running:
            return
            
        try:
            # Initialize PyAudio
            self._audio = pyaudio.PyAudio()
            
            # Find audio device
            device_index, device_info = self._find_blackhole()
            
            # Get format from config
            audio_format = getattr(pyaudio, f'pa{self.config["format"].capitalize()}')
            
            # Open stream with configured settings
            self._stream = self._audio.open(
                format=audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self._stream.start_stream()
            
            # Print status if enabled
            if self.config['show_status']:
                print("\n=== Audio Stream Status ===")
                print("✓ Audio stream started")
                print(f"✓ Sample rate: {self.sample_rate} Hz")
                print(f"✓ Channels: {self.channels}")
                print(f"✓ Input device: {device_info['name']}")
                print(f"✓ Latency: {device_info['defaultLowInputLatency']:.3f}s")
                print("========================\n")
            
        except Exception as e:
            print(f"\n❌ Error starting audio stream: {str(e)}")
            self.stop()
            raise
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data."""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Check audio levels but only show initial detection
            level = np.abs(audio_data).mean()
            if level > self.config['audio_level_threshold'] and not self.audio_detected:
                # Only show detection message once at startup
                if not hasattr(self, '_shown_detection') and self.config['show_detection']:
                    logger.info("✓ Audio input detected")
                    self._shown_detection = True
                self.audio_detected = True
            elif level <= self.config['audio_level_threshold']:
                self.audio_detected = False
            
            self.audio_queue.put(audio_data)
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paAbort)
            
    def stop(self):
        """Stop the audio streaming."""
        self.is_running = False
        
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None
            
        logger.info("✓ Audio stream stopped")
        
    def get_audio_chunk(self, timeout: Optional[float] = None):
        """Get an audio chunk from the queue."""
        try:
            timeout = timeout if timeout is not None else self.config['queue_timeout']
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
