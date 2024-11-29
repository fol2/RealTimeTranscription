import queue
import threading
import numpy as np
import logging
import pyaudio
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, sample_rate=48000, channels=2, chunk_size=4000):
        """Initialize audio stream with BlackHole's native settings."""
        self.sample_rate = sample_rate  # BlackHole's native rate
        self.channels = channels        # Stereo
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_running = False
        self._stream = None
        self._audio = None
        self._stream_thread = None
        self.debug_counter = 0
        self.audio_detected = False
        
    def _find_blackhole(self) -> Tuple[int, dict]:
        """Find BlackHole 2ch device and return its index and info."""
        audio = pyaudio.PyAudio()
        device_index = None
        device_info = None
        
        # List all audio devices
        print("\n=== Available Audio Devices ===")
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if "BlackHole 2ch" in dev_info["name"]:
                device_index = i
                device_info = dev_info
                print(f"\n✓ Found BlackHole 2ch at index {i}")
                print("Device info:")
                for key, value in dev_info.items():
                    print(f"  {key}: {value}")
                break
            
        if device_index is None:
            raise RuntimeError("❌ Could not find BlackHole 2ch device")
            
        return device_index, device_info
        
    def start(self, device="BlackHole 2ch"):
        """Start the audio streaming from BlackHole."""
        if self.is_running:
            return
            
        try:
            # Initialize PyAudio
            self._audio = pyaudio.PyAudio()
            
            # Find BlackHole device
            device_index, device_info = self._find_blackhole()
            
            # Open stream with BlackHole's native settings
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,  # 32-bit float
                channels=self.channels,    # Stereo
                rate=self.sample_rate,     # 48kHz
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self._stream.start_stream()
            
            # Print status
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
            # Convert bytes to numpy array (32-bit float)
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Check audio levels periodically
            self.debug_counter += 1
            if self.debug_counter % 10 == 0:
                level = np.abs(audio_data).mean()
                bars = int(50 * level)
                print(f"Audio Level: {'=' * bars}{' ' * (50 - bars)} [{level:.4f}]")
                
                # Update audio detection status
                if level > 0.01:  # Adjust threshold as needed
                    if not self.audio_detected:
                        print("\n✓ Audio input detected!")
                        self.audio_detected = True
                else:
                    self.audio_detected = False
            
            # Put the audio data in the queue
            self.audio_queue.put(audio_data)
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
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
            
        print("\n✓ Audio stream stopped cleanly")
        
    def get_audio_chunk(self, timeout: Optional[float] = 0.1):
        """Get an audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
