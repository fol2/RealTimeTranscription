import torch
import logging
from typing import Optional, List, Dict, Tuple
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import numpy as np
from collections import deque
import threading
import queue
from dataclasses import dataclass
import time
import re
from difflib import SequenceMatcher
import json
from datetime import datetime
import librosa
import scipy.signal as signal
from scipy.io import wavfile
import noisereduce as nr
import webrtcvad
from typing import List, Tuple
import struct
import psutil
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import colorama
from colorama import Fore, Style
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from .logging_utils import setup_logging, logger, transcription_logger
from .audio_processing import AudioPreprocessor, VoiceActivityDetector, AudioSegment
from .text_processing import TranscriptionResult, TranscriptionLogger
from .buffer_management import StreamBuffer
from .model_management import ModelConfig, ModelManager
from .realtime_handler import RealTimeTranscriptionHandler
from .pipeline_workers import TranscriptionWorker, TextProcessingWorker
from .config_manager import ConfigManager

# Initialize colorama for cross-platform color support
colorama.init()

class TranscriptionPipeline:
    """Manages the parallel processing pipeline"""
    def __init__(self, model_config: Optional[Dict] = None):
        # Initialize logging first
        self.session_id = setup_logging()
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Load configurations
        base_config = {}
        if model_config:
            base_config.update(model_config)
            
        # Add other configurations
        base_config.update(self.config_manager.get_audio_config())
        base_config.update(self.config_manager.get_buffer_config())
        base_config.update(self.config_manager.get_transcription_config())
        
        self.config = base_config
        
        # Initialize model manager with the complete config
        try:
            self.model_manager = ModelManager(self.config)
            self.model_manager.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {str(e)}")
            logger.error(f"Config used: {self.config}")
            raise
        
        # Use model manager's attributes
        self.model = self.model_manager.model
        self.processor = self.model_manager.processor
        self.is_whisper = self.model_manager.is_whisper
        
        # Get buffer config for queue sizes
        buffer_config = self.config_manager.get_buffer_config()
        
        # Initialize queues with configured sizes
        self.transcription_queue = queue.Queue(maxsize=buffer_config['transcription_queue_size'])
        self.output_queue = queue.Queue(maxsize=buffer_config['output_queue_size'])
        
        # Initialize components with new chunk sizes
        self.audio_buffer = StreamBuffer(
            max_size=self.config["max_size"],
            min_size=self.config["min_size"],
            overlap=self.config["overlap"],
            max_chunks=self.config["max_chunks"]
        )
        # Initialize preprocessor with config
        audio_config = self.config_manager.get_audio_config()
        self.preprocessor = AudioPreprocessor(config=audio_config)
        self.vad = VoiceActivityDetector(config=audio_config)
        self.logger = TranscriptionLogger()
        
        # Initialize silence tracking
        self._accumulated_silence = 0
        self._last_speech_state = False
        self._speech_start_time = None
        
        # Initialize thread control
        self.running = False
        self.threads = []
        
        # Add sentence tracking
        self.current_sentence = ""
        self.sentence_end_patterns = ['.', '!', '?', ';']
        
        # Initialize transcription handler
        self.transcription_handler = RealTimeTranscriptionHandler()
        
        # Initialize workers
        self.transcription_worker = TranscriptionWorker(self)
        self.text_processing_worker = TextProcessingWorker(self)
        
        # Load processing config
        processing_config = self.config_manager.get_processing_config()
        self.buffer_time = processing_config['buffer_time']
        self.min_buffer_size = processing_config['min_buffer_size']
        
        # Load display config
        display_config = self.config_manager.get_display_config()
        self.show_confidence = display_config['show_confidence']
        self.show_timestamps = display_config['show_timestamps']
        self.colored_output = display_config['colored_output']
        
    def start(self):
        """Start all pipeline threads"""
        self.running = True
        
        # Start transcription thread
        t_transcribe = threading.Thread(target=self.transcription_worker.run)
        t_transcribe.daemon = True
        t_transcribe.start()
        self.threads.append(t_transcribe)
        
        # Start text processing thread
        t_process = threading.Thread(target=self.text_processing_worker.run)
        t_process.daemon = True
        t_process.start()
        self.threads.append(t_process)
        
    def stop(self):
        """Stop all pipeline threads"""
        self.running = False
        for thread in self.threads:
            thread.join()
            
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Add audio chunk to pipeline with enhanced error handling"""
        try:
            if audio_chunk is None or len(audio_chunk) == 0:
                logger.warning("Received empty audio chunk")
                return None
                
            # Log audio chunk statistics
            logger.debug(f"Processing chunk: shape={audio_chunk.shape}, "
                        f"min={np.min(audio_chunk):.3f}, max={np.max(audio_chunk):.3f}, "
                        f"mean={np.mean(np.abs(audio_chunk)):.3f}")
            
            try:
                # Preprocess audio
                processed_audio = self.preprocessor.process(audio_chunk)
            except Exception as e:
                logger.error(f"Audio preprocessing failed: {e}")
                logger.exception("Preprocessing traceback:")
                return None
                
            try:
                # Check for voice activity
                is_speech, confidences = self.vad.process_audio(processed_audio)
            except Exception as e:
                logger.error(f"VAD processing failed: {e}")
                logger.exception("VAD traceback:")
                return None
            
            # Track speech state changes with improved error handling
            current_time = time.time()
            
            try:
                if is_speech and not self._last_speech_state:
                    logger.info("Speech detected")
                    self._speech_start_time = current_time
                    self._accumulated_silence = 0
                elif not is_speech:
                    if self._last_speech_state:
                        self._accumulated_silence += len(audio_chunk) / self.config['sample_rate']
                        if self._accumulated_silence >= self.config['silence_threshold_seconds']:
                            logger.info(f"Speech ended (silence: {self._accumulated_silence:.2f}s)")
                            try:
                                # Clear buffers
                                self.audio_buffer.reset()
                                self._accumulated_silence = 0
                                self.transcription_queue.queue.clear()
                                self.output_queue.queue.clear()
                            except Exception as e:
                                logger.error(f"Failed to clear buffers: {e}")
                                logger.exception("Buffer clearing traceback:")
                    else:
                        self._accumulated_silence += len(audio_chunk) / self.config['sample_rate']
            except Exception as e:
                logger.error(f"Speech state tracking failed: {e}")
                logger.exception("State tracking traceback:")
            
            # Update speech state
            self._last_speech_state = is_speech
            
            try:
                # Create audio segment
                segment = AudioSegment(
                    data=processed_audio,
                    timestamp=current_time,
                    is_speech=is_speech
                )
                
                # Only add to buffer if there's potential speech
                if is_speech:
                    logger.debug(f"Adding speech segment (length: {len(processed_audio)})")
                    self.audio_buffer.add(segment)
            except Exception as e:
                logger.error(f"Failed to handle audio segment: {e}")
                logger.exception("Segment handling traceback:")
            
            # Check output queue for results with error handling
            try:
                result = self.output_queue.get_nowait()
                if result and result.text:
                    logger.info(f"Transcribed: {result.text}")
                    if len(result.text.split()) >= 2:
                        try:
                            formatted_text = self.logger.format_transcription(result.text)
                            if formatted_text:
                                logger.debug(f"Formatted transcription: {formatted_text}")
                                return formatted_text
                        except Exception as e:
                            logger.error(f"Text formatting failed: {e}")
                            logger.exception("Formatting traceback:")
                            return result.text  # Return unformatted text as fallback
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Queue processing failed: {e}")
                logger.exception("Queue processing traceback:")
            
            return None
            
        except Exception as e:
            logger.error(f"Critical error in process_chunk: {e}")
            logger.exception("Full process_chunk traceback:")
            # Try to recover by resetting state
            self._last_speech_state = False
            self._accumulated_silence = 0
            return None
            
    def _handle_error(self, error: Exception, context: str) -> None:
        """Centralized error handling with context"""
        logger.error(f"Error in {context}: {str(error)}")
        logger.exception(f"{context} traceback:")
        
        # Log additional diagnostic information
        logger.error(f"Current pipeline state:")
        logger.error(f"  Speech state: {self._last_speech_state}")
        logger.error(f"  Accumulated silence: {self._accumulated_silence:.2f}s")
        logger.error(f"  Queue sizes:")
        logger.error(f"    Transcription queue: {self.transcription_queue.qsize()}")
        logger.error(f"    Output queue: {self.output_queue.qsize()}")
        
        # Try to recover based on context
        if context == "audio_processing":
            self.audio_buffer.reset()
        elif context == "transcription":
            self.transcription_handler.reset()
        elif context == "output":
            self.output_queue.queue.clear()
        
    def _detect_speech(self, audio: np.ndarray) -> bool:
        """Simple voice activity detection"""
        return np.abs(audio).mean() > self.config['vad_speech_threshold']
        
class Transcriber:
    """Wrapper class for backwards compatibility with existing code"""
    def __init__(self, model_path=None):
        # Initialize logging if not already done
        if not logging.getLogger().handlers:
            self.session_id = setup_logging()
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Create ModelConfig instance and get configuration
        model_config = ModelConfig()
        # Get model configuration (this will print device info once)
        config = model_config.get_model_config(model_path)
        
        # Initialize pipeline with configuration
        try:
            self.pipeline = TranscriptionPipeline(config)
            self.pipeline.start()
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Process audio chunk using the pipeline"""
        return self.pipeline.process_chunk(audio_chunk)
        
    def __del__(self):
        """Ensure pipeline is stopped when object is destroyed"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
