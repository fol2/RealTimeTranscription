import numpy as np
import librosa
import webrtcvad
import struct
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Container for audio data with metadata"""
    data: np.ndarray
    timestamp: float
    is_speech: bool = False

class AudioPreprocessor:
    """Handles audio preprocessing with minimal signal modification"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ConfigManager().get_audio_processing_config()
        self.target_sample_rate = self.config['sample_rate']
        
    def process(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Apply minimal preprocessing pipeline to audio chunk"""
        try:
            sample_rate = sample_rate or self.config['sample_rate']
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sample_rate
                )
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Remove DC offset if enabled
            if self.config['dc_offset_removal']:
                audio = self._remove_dc_offset(audio)
            
            # Normalize if enabled
            if self.config['normalization_enabled']:
                audio = self._normalize_audio(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio  # Return original audio if processing fails
            
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal"""
        return audio - np.mean(audio)
        
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Simple peak normalization to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

class VoiceActivityDetector:
    """Robust voice activity detection using webrtcvad"""
    def __init__(self, config: Optional[Dict] = None):
        # Get both audio and VAD configs
        self.config = config or ConfigManager().get_audio_config()
        self.vad_config = ConfigManager().get_vad_config()
        
        self.sample_rate = self.config['sample_rate']
        self.frame_duration_ms = self.vad_config['frame_duration_ms']
        self.padding_duration_ms = self.vad_config['padding_duration_ms']
        
        # Use configuration factors for calculations
        ms_to_sec = self.config['ms_to_sec_factor']
        self.frame_size = int(self.sample_rate * self.frame_duration_ms * 
                            self.config['vad_frame_size_factor'] / ms_to_sec)
        self.padding_size = int(self.sample_rate * self.padding_duration_ms * 
                              self.config['vad_padding_size_factor'] / ms_to_sec)
        self.min_silence_frames = int(self.padding_duration_ms * 
                                    self.config['vad_silence_frames_factor'] / 
                                    self.frame_duration_ms)
        
        self.vad = webrtcvad.Vad(self.vad_config['aggressiveness'])
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.triggered = False
        self.min_speech_frames = self.config['vad_min_speech_frames']
        self.speech_start_time = None
        self.last_speech_time = None
        self.speech_probability = self.vad_config['initial_probability']
        self.speech_probability_threshold = self.vad_config['probability_threshold']
        
    def process_audio(self, audio: np.ndarray) -> tuple[bool, list[float]]:
        """Process audio chunk with improved speech detection"""
        current_time = time.time()
        
        if audio.dtype != np.int16:
            audio = (audio * self.config['audio_scaling_factor']).astype(np.int16)
            
        frames = self._frame_generator(audio)
        confidences = []
        speech_detected = False
        frame_speech_count = 0
        total_frames = 0
        
        for frame in frames:
            total_frames += 1
            is_speech = self._process_frame(frame)
            if is_speech:
                frame_speech_count += 1
            confidences.append(
                self.vad_config['speech_confidence'] if is_speech 
                else self.vad_config['silence_confidence']
            )
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = max(0, self.silence_frames - self.vad_config['probability_decay'])
                self.last_speech_time = current_time
                speech_detected = True
            else:
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - self.vad_config['probability_update'])
            
            if total_frames > 0:
                self.speech_probability = (self.vad_config['probability_decay'] * self.speech_probability + 
                                         self.vad_config['probability_update'] * (frame_speech_count / total_frames))
            
            if not self.triggered:
                if self.speech_frames >= self.min_speech_frames or self.speech_probability > self.speech_probability_threshold:
                    self.triggered = True
                    self.is_speaking = True
                    self.speech_start_time = current_time
            else:
                if (self.silence_frames >= self.min_silence_frames and 
                    self.speech_probability < self.vad_config['silence_probability'] and 
                    (current_time - self.speech_start_time) > self.vad_config['min_speech_duration_ms'] / 1000):
                    self.triggered = False
                    self.is_speaking = False
                    self.speech_frames = 0
                    
        if self.last_speech_time and (current_time - self.last_speech_time < self.vad_config['max_silence_duration_ms'] / 1000):
            self.is_speaking = True
            
        return self.is_speaking, confidences
        
    def _frame_generator(self, audio: np.ndarray) -> list[bytes]:
        """Generate frames of appropriate size for webrtcvad"""
        n = len(audio)
        offset = 0
        frames = []
        
        while offset + self.frame_size <= n:
            frame = audio[offset:offset + self.frame_size]
            frames.append(self._frame_to_bytes(frame))
            offset += self.frame_size
            
        return frames
        
    def _frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert numpy array to bytes in required format"""
        format_str = f"%d{self.config['frame_pack_format']}"
        return struct.pack(format_str % len(frame), *frame)
        
    def _process_frame(self, frame: bytes) -> bool:
        """Process a single frame and return speech detection result"""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return False 