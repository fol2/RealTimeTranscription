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

# Initialize colorama for cross-platform color support
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Enhanced custom formatter with better visual hierarchy"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    CATEGORIES = {
        'transcription': f"{Fore.BLUE}█{Style.RESET_ALL}",
        'model': f"{Fore.MAGENTA}◆{Style.RESET_ALL}",
        'system': f"{Fore.GREEN}●{Style.RESET_ALL}",
        'error': f"{Fore.RED}✖{Style.RESET_ALL}",
    }
    
    def format(self, record):
        # Add color to level name if it's a terminal
        if record.levelname in self.COLORS and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname:8}{Style.RESET_ALL}"
        
        # Format based on message type
        msg = record.msg
        if 'transcription' in msg.lower():
            # Format transcription with box
            text = msg.split(':', 1)[1].strip()
            record.msg = (
                f"\n{self.CATEGORIES['transcription']} Transcription "
                f"{Fore.BLUE}{'─' * 60}{Style.RESET_ALL}\n"
                f"  {text}\n"
                f"{Fore.BLUE}{'─' * 72}{Style.RESET_ALL}\n"
            )
        elif 'error' in msg.lower():
            # Format errors with clear marker
            record.msg = f"{self.CATEGORIES['error']} {msg}"
        elif any(word in msg.lower() for word in ['model', 'loading', 'initialized']):
            # Format model-related messages
            record.msg = f"{self.CATEGORIES['model']} {msg}"
        else:
            # Format system messages
            record.msg = f"{self.CATEGORIES['system']} {msg}"
        
        return super().format(record)

class TranscriptionFormatter(logging.Formatter):
    """Dedicated formatter for transcription output"""
    def format(self, record):
        # Format transcription with timestamp and clean styling
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = record.msg
        
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return (
                f"\n{Fore.BLUE}[{timestamp}] Transcription {Style.RESET_ALL}\n"
                f"{Fore.BLUE}{'─' * 72}{Style.RESET_ALL}\n"
                f"  {text}\n"
                f"{Fore.BLUE}{'─' * 72}{Style.RESET_ALL}\n"
            )
        else:
            return f"[{timestamp}] {text}"

class TranscriptionFileFormatter(logging.Formatter):
    """Formatter for transcription log files with JSON support"""
    def __init__(self, use_json=False):
        super().__init__()
        self.use_json = use_json
        
    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.use_json:
            # JSON format for easy parsing
            data = {
                'timestamp': timestamp,
                'text': record.msg,
                'session_id': getattr(record, 'session_id', ''),
                'confidence': getattr(record, 'confidence', None)
            }
            return json.dumps(data)
        else:
            # Human-readable format
            return f"[{timestamp}] {record.msg}"

def setup_logging():
    """Configure enhanced logging system with improved file output"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Generate session ID and log filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{timestamp}"
    
    system_log_file = f"logs/system_{timestamp}.log"
    transcription_log_file = f"logs/transcription_{timestamp}.log"
    transcription_json_file = f"logs/transcription_{timestamp}.jsonl"
    
    # Setup system logger (unchanged)
    console_handler = logging.StreamHandler()
    system_file_handler = RotatingFileHandler(
        system_log_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(console_formatter)
    system_file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    root_logger.addHandler(system_file_handler)
    
    # Setup transcription logger with multiple outputs
    transcription_logger = logging.getLogger('transcription')
    transcription_logger.setLevel(logging.INFO)
    transcription_logger.propagate = False
    
    # Console handler (unchanged)
    trans_console_handler = logging.StreamHandler()
    trans_console_handler.setFormatter(TranscriptionFormatter())
    
    # Human-readable file handler
    trans_file_handler = RotatingFileHandler(
        transcription_log_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    trans_file_handler.setFormatter(TranscriptionFileFormatter(use_json=False))
    
    # JSON file handler for machine processing
    trans_json_handler = RotatingFileHandler(
        transcription_json_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    trans_json_handler.setFormatter(TranscriptionFileFormatter(use_json=True))
    
    transcription_logger.addHandler(trans_console_handler)
    transcription_logger.addHandler(trans_file_handler)
    transcription_logger.addHandler(trans_json_handler)
    
    # Store session info
    with open(f"logs/session_{timestamp}_info.json", 'w') as f:
        json.dump({
            'session_id': session_id,
            'timestamp': timestamp,
            'files': {
                'system_log': os.path.abspath(system_log_file),
                'transcription_log': os.path.abspath(transcription_log_file),
                'transcription_json': os.path.abspath(transcription_json_file)
            }
        }, f, indent=2)
    
    return session_id

# Get loggers
logger = logging.getLogger(__name__)
transcription_logger = logging.getLogger('transcription')

@dataclass
class AudioSegment:
    """Container for audio data with metadata"""
    data: np.ndarray
    timestamp: float
    is_speech: bool = False

@dataclass
class TranscriptionResult:
    """Container for transcription results with metadata"""
    text: str
    timestamp: float
    confidence: float

class TranscriptionLogger:
    """Handles logging and formatting of transcriptions"""
    def __init__(self):
        self.current_sentence = []
        self.last_phrase = ""
        self.similarity_threshold = 0.85
        
    def format_transcription(self, text: str) -> Optional[str]:
        """Format transcription for output, handling duplicates and partial phrases"""
        # Skip if too similar to last phrase
        if self._is_similar(text, self.last_phrase):
            return None
            
        # Update last phrase
        self.last_phrase = text
        
        # Clean up the text
        cleaned = self._clean_text(text)
        if not cleaned:
            return None
            
        # Add to current sentence
        self.current_sentence.append(cleaned)
        
        # Check if we should output the sentence
        if self._is_sentence_complete(cleaned):
            result = self._format_sentence()
            self.current_sentence = []
            return result
            
        return None
        
    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are too similar"""
        if not text1 or not text2:
            return False
        return SequenceMatcher(None, text1, text2).ratio() > self.similarity_threshold
        
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better handling of repeated phrases"""
        # Remove the transcription prefix
        text = re.sub(r"Transcription:\s*", "", text, flags=re.IGNORECASE)
        
        # Convert to uppercase and strip
        text = text.upper().strip()
        
        # Remove non-alphabetic characters except apostrophes and hyphens
        text = re.sub(r'[^A-Z\'\-\s]', '', text)
        
        # Handle contractions and hyphenated words
        text = text.replace("'", "'")  # Normalize apostrophes
        
        # Split into words while preserving contractions and hyphens
        words = text.split()
        cleaned_words = []
        prev_word = None
        repetition_count = 0
        
        for word in words:
            # Skip single-letter words except 'A' and 'I'
            if len(word) <= 1 and word not in ['A', 'I']:
                continue
                
            # Check for repetition
            if word == prev_word:
                repetition_count += 1
                # Only keep repetition if it's likely intentional
                if repetition_count <= 1:
                    cleaned_words.append(word)
            else:
                # Check if this word is part of a stutter
                if prev_word and prev_word.startswith(word[:2]):
                    continue  # Skip potential stutter
                cleaned_words.append(word)
                repetition_count = 0
                prev_word = word
        
        # Join words
        cleaned_text = ' '.join(cleaned_words)
        
        # Remove common transcription artifacts
        artifacts = {
            r'\bUM\b': '',
            r'\bUH\b': '',
            r'\bHMM\b': '',
            r'\bAH\b': '',
            r'\bER\b': '',
            r'\bLIKE\b\s+(?=\bLIKE\b)': '',  # Remove repeated "like"
        }
        
        for pattern, replacement in artifacts.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Clean up any resulting multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if current text forms a complete sentence"""
        # Check for ending punctuation
        if any(text.rstrip().endswith(p) for p in '.!?'):
            return True
            
        # Check for common sentence-ending words
        ending_words = {'SO', 'AND', 'BUT', 'OR', 'BECAUSE', 'HOWEVER'}
        words = text.split()
        if words and words[-1] in ending_words:
            return True
            
        # Check length
        if len(" ".join(self.current_sentence).split()) > 20:
            return True
            
        return False
        
    def _format_sentence(self) -> str:
        """Format the complete sentence for output"""
        # Combine all phrases
        full_text = " ".join(self.current_sentence)
        
        # Remove redundant phrases
        phrases = full_text.split()
        unique_phrases = []
        for phrase in phrases:
            if not unique_phrases or phrase != unique_phrases[-1]:
                unique_phrases.append(phrase)
                
        # Format final text
        final_text = " ".join(unique_phrases)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        return f"[{timestamp}] {final_text}"

class StreamBuffer:
    """Enhanced thread-safe buffer with smart chunking strategy"""
    def __init__(self, 
                 max_size: int = 80000,    # 5 seconds at 16kHz
                 min_size: int = 40000,    # 2.5 seconds minimum
                 overlap: int = 16000,     # 1 second overlap
                 max_chunks: int = 3):     # Keep 3 chunks in memory
        self.max_size = max_size
        self.min_size = min_size
        self.overlap = overlap
        self.buffer = deque(maxlen=max_size)
        self.chunks = deque(maxlen=max_chunks)
        self.lock = threading.Lock()
        self.last_chunk_time = 0
        
    def add(self, segment: AudioSegment):
        """Add audio segment to buffer with improved chunk management"""
        with self.lock:
            # Add to main buffer
            if isinstance(self.buffer, deque):
                self.buffer.extend(segment.data)
            else:
                self.buffer = np.concatenate([self.buffer, segment.data])
            
            current_time = time.time()
            buffer_duration = len(self.buffer) / 16000  # Duration in seconds
            
            # Create new chunk if we have enough data and enough time has passed
            if (len(self.buffer) >= self.min_size and 
                (current_time - self.last_chunk_time) >= 0.5):  # At least 0.5s between chunks
                
                # Create new chunk with overlap
                chunk_data = np.array(list(self.buffer))
                self.chunks.append(
                    AudioSegment(
                        data=chunk_data,
                        timestamp=segment.timestamp,
                        is_speech=segment.is_speech
                    )
                )
                
                # Keep overlap portion
                if len(self.buffer) > self.overlap:
                    self.buffer = deque(
                        list(self.buffer)[-self.overlap:],
                        maxlen=self.max_size
                    )
                
                self.last_chunk_time = current_time
                
    def get_chunks(self) -> List[AudioSegment]:
        """Get all available chunks"""
        with self.lock:
            return list(self.chunks)
            
    def clear_processed_chunks(self):
        """Clear processed chunks but maintain overlap"""
        with self.lock:
            self.chunks.clear()

class AudioPreprocessor:
    """Handles audio preprocessing with minimal signal modification"""
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply minimal preprocessing pipeline to audio chunk"""
        try:
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
            
            # Remove DC offset
            audio = self._remove_dc_offset(audio)
            
            # Simple normalization
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
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 padding_duration_ms: int = 500,  # Increased from 300ms to 500ms
                 aggressiveness: int = 1):        # Reduced from 3 to 1
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.padding_size = int(sample_rate * padding_duration_ms / 1000)
        
        # Initialize webrtcvad with lower aggressiveness
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Improved state tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.triggered = False
        self.min_speech_frames = 3  # Reduced from 5 to be more responsive
        self.min_silence_frames = int(padding_duration_ms / frame_duration_ms)  # Dynamic calculation
        self.speech_start_time = None
        self.last_speech_time = None
        self.speech_probability = 0.0  # Add probability tracking
        
    def process_audio(self, audio: np.ndarray) -> Tuple[bool, List[float]]:
        """Process audio chunk with improved speech detection"""
        current_time = time.time()
        
        # Ensure audio is in the correct format (16-bit PCM)
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
            
        # Split audio into frames
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
            confidences.append(1.0 if is_speech else 0.0)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = max(0, self.silence_frames - 1)  # Gradual reduction
                self.last_speech_time = current_time
                speech_detected = True
            else:
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - 0.5)  # Slower decay
            
            # Update speech probability
            if total_frames > 0:
                self.speech_probability = 0.7 * self.speech_probability + 0.3 * (frame_speech_count / total_frames)
            
            # State machine for speech detection with hysteresis
            if not self.triggered:
                # More lenient trigger condition
                if self.speech_frames >= self.min_speech_frames or self.speech_probability > 0.6:
                    self.triggered = True
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    logger.debug("Speech started")
            else:
                # More conservative end condition
                if (self.silence_frames >= self.min_silence_frames and 
                    self.speech_probability < 0.3 and 
                    (current_time - self.speech_start_time) > 0.5):  # Minimum duration
                    self.triggered = False
                    self.is_speaking = False
                    self.speech_frames = 0
                    logger.debug("Speech ended")
                    
        # Maintain speech state for short gaps
        if self.last_speech_time and (current_time - self.last_speech_time < 0.3):  # 300ms gap tolerance
            self.is_speaking = True
            
        return self.is_speaking, confidences
        
    def _frame_generator(self, audio: np.ndarray) -> List[bytes]:
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
        return struct.pack("%dh" % len(frame), *frame)
        
    def _process_frame(self, frame: bytes) -> bool:
        """Process a single frame and return speech detection result"""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return False

class TranscriptionPipeline:
    """Manages the parallel processing pipeline"""
    def __init__(self, model_config: Optional[Dict] = None):
        # Initialize logging first
        self.session_id = setup_logging()
        
        # Configure logging level
        logging.getLogger('src.transcriber').setLevel(logging.INFO)
        
        # Initialize configuration with no default model
        self.config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "chunk_size": 80000,      # 5 seconds at 16kHz
            "min_chunk_size": 40000,  # 2.5 seconds minimum
            "overlap": 16000,         # 1 second overlap
            "max_chunks": 3,          # Keep 3 chunks in memory
            "silence_threshold": 0.0005,
            "confidence_threshold": 0.5,
            "max_segment_duration": 10.0
        }
        
        # Update with provided config (which includes model selection)
        if model_config:
            self.config.update(model_config)
        else:
            raise ValueError("model_config is required and must include model_name")
        
        # Initialize model first
        self.load_model()
        
        # Initialize queues with larger sizes
        self.transcription_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Initialize components with new chunk sizes
        self.audio_buffer = StreamBuffer(
            max_size=self.config["chunk_size"],
            min_size=self.config["min_chunk_size"],
            overlap=self.config["overlap"],
            max_chunks=self.config["max_chunks"]
        )
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000)
        self.vad = VoiceActivityDetector(
            sample_rate=16000,
            frame_duration_ms=30,
            padding_duration_ms=500,
            aggressiveness=1
        )
        self.logger = TranscriptionLogger()
        
        # Initialize silence tracking
        self._accumulated_silence = 0
        self._last_speech_state = False
        self._speech_start_time = None
        
        # Initialize thread control
        self.running = False
        self.threads = []
        
    def load_model(self):
        """Load ASR model with Whisper support"""
        try:
            logger.info(f"Loading model: {self.config['model_name']}")
            
            # Check if using Whisper model
            if "whisper" in self.config["model_name"].lower():
                self.processor = WhisperProcessor.from_pretrained(
                    self.config["model_name"],
                    cache_dir="models",
                    language="en",  # Set default language
                    task="transcribe"  # Set default task
                )
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.config["model_name"],
                    cache_dir="models",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True  # Use safetensors for better memory efficiency
                )
                
                # Don't set forced_decoder_ids here since we're using language parameter
                self.model.config.suppress_tokens = []
                self.is_whisper = True
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    self.config["model_name"],
                    cache_dir="models"
                )
                self.model = Wav2Vec2ForCTC.from_pretrained(
                    self.config["model_name"],
                    cache_dir="models",
                    torch_dtype=torch.float32
                )
                self.is_whisper = False
            
            # Move to device and optimize
            self.model = self.model.to(self.config["device"])
            self.model.eval()
            
            logger.info(f"Model loaded successfully: {'Whisper' if self.is_whisper else 'Wav2Vec2'}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def start(self):
        """Start all pipeline threads"""
        self.running = True
        
        # Start transcription thread
        t_transcribe = threading.Thread(target=self._transcription_worker)
        t_transcribe.daemon = True
        t_transcribe.start()
        self.threads.append(t_transcribe)
        
        # Start text processing thread
        t_process = threading.Thread(target=self._text_processing_worker)
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
                        self._accumulated_silence += len(audio_chunk) / 16000
                        if self._accumulated_silence >= 0.3:
                            logger.info(f"Speech ended (silence: {self._accumulated_silence:.2f}s)")
                            try:
                                # Clear buffers
                                self.audio_buffer = StreamBuffer(
                                    max_size=self.config["chunk_size"],
                                    min_size=self.config["min_chunk_size"],
                                    overlap=self.config["overlap"],
                                    max_chunks=self.config["max_chunks"]
                                )
                                self._accumulated_silence = 0
                                self.transcription_queue.queue.clear()
                                self.output_queue.queue.clear()
                            except Exception as e:
                                logger.error(f"Failed to clear buffers: {e}")
                                logger.exception("Buffer clearing traceback:")
                    else:
                        self._accumulated_silence += len(audio_chunk) / 16000
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
            self.audio_buffer = StreamBuffer(
                max_size=self.config["chunk_size"],
                min_size=self.config["min_chunk_size"],
                overlap=self.config["overlap"],
                max_chunks=self.config["max_chunks"]
            )
        elif context == "transcription":
            self.transcription_queue.queue.clear()
        elif context == "output":
            self.output_queue.queue.clear()
        
    def _detect_speech(self, audio: np.ndarray) -> bool:
        """Simple voice activity detection"""
        return np.abs(audio).mean() > 0.0005
        
    def _clean_and_combine_text(self, text_buffer: List[TranscriptionResult]) -> Optional[str]:
        """Clean and combine transcription results with improved text processing"""
        if not text_buffer:
            return None
            
        # Sort by timestamp
        sorted_results = sorted(text_buffer, key=lambda x: x.timestamp)
        
        # Combine texts with confidence weighting
        weighted_texts = []
        for result in sorted_results:
            # Only include text with reasonable confidence
            if result.confidence > self.config["confidence_threshold"]:
                weighted_texts.append(result.text)
        
        if not weighted_texts:
            return None
            
        # Join texts
        combined_text = " ".join(weighted_texts)
        
        # Basic cleaning
        combined_text = re.sub(r'[^a-zA-Z\s]', '', combined_text)  # Remove non-alphabetic chars
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()  # Remove extra spaces
        
        # Convert to uppercase
        combined_text = combined_text.upper()
        
        # Remove word repetitions while preserving meaningful duplicates
        words = combined_text.split()
        deduped_words = []
        repetition_count = 0
        prev_word = None
        
        for word in words:
            # Skip single-letter words
            if len(word) <= 1:
                continue
                
            # Check for repetition
            if word == prev_word:
                repetition_count += 1
                # Only keep repetition if it's likely intentional (max 2 times)
                if repetition_count <= 1:
                    deduped_words.append(word)
            else:
                deduped_words.append(word)
                repetition_count = 0
                prev_word = word
        
        # Join words back together
        final_text = ' '.join(deduped_words)
        
        # Remove common transcription artifacts
        artifacts = {
            r'\bAH\b': '',
            r'\bUM\b': '',
            r'\bUH\b': '',
            r'\bMM\b': '',
            r'\bHMM\b': '',
        }
        
        for pattern, replacement in artifacts.items():
            final_text = re.sub(pattern, replacement, final_text)
        
        # Remove any resulting double spaces
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        return final_text if final_text else None
        
    def _transcription_worker(self):
        """Worker thread for ASR processing with enhanced logging"""
        while self.running:
            try:
                chunks = self.audio_buffer.get_chunks()
                if chunks:
                    logger.debug(f"Processing {len(chunks)} audio chunks")
                
                for chunk in chunks:
                    if chunk.is_speech:
                        try:
                            if self.is_whisper:
                                # Normalize audio to prevent low levels
                                audio_data = chunk.data
                                if np.abs(audio_data).max() > 0:
                                    audio_data = audio_data / np.abs(audio_data).max()
                                
                                # Process audio for Whisper with proper attention mask
                                inputs = self.processor(
                                    audio_data,
                                    sampling_rate=16000,
                                    return_tensors="pt",
                                    return_attention_mask=True  # Explicitly request attention mask
                                )
                                
                                # Move all inputs to device
                                input_features = inputs.input_features.to(self.config["device"])
                                attention_mask = inputs.attention_mask.to(self.config["device"]) if hasattr(inputs, 'attention_mask') else None
                                
                                with torch.no_grad():
                                    # Generate without explicit task setting to avoid conflicts
                                    generated_ids = self.model.generate(
                                        input_features,
                                        attention_mask=attention_mask,
                                        num_beams=5,
                                        max_length=448,
                                        suppress_tokens=[1],
                                        return_timestamps=False,
                                        task="transcribe",  # Explicitly set task
                                        language="en",      # Force English
                                        no_repeat_ngram_size=3  # Reduce repetition
                                    )
                                    
                                    transcription = self.processor.batch_decode(
                                        generated_ids, 
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True
                                    )
                                    
                                    # Whisper provides high-quality transcriptions by default
                                    confidence = 0.95
                            else:
                                # Existing Wav2Vec2 processing
                                inputs = self.processor(
                                    chunk.data,
                                    sampling_rate=16000,
                                    return_tensors="pt",
                                    padding=True
                                ).to(self.config["device"])
                                
                                with torch.no_grad():
                                    logits = self.model(inputs.input_values).logits
                                    predicted_ids = torch.argmax(logits, dim=-1)
                                    transcription = self.processor.batch_decode(predicted_ids)
                                    confidence = torch.max(torch.softmax(logits, dim=-1)).item()
                            
                            if transcription and transcription[0].strip():
                                text = transcription[0].strip()
                                if self.is_whisper:
                                    text = re.sub(r'\[.*?\]', '', text)
                                    text = re.sub(r'\(.*?\)', '', text)
                                    text = re.sub(r'\s+', ' ', text).strip()
                                
                                # Create log record with extra info
                                record = logging.LogRecord(
                                    name='transcription',
                                    level=logging.INFO,
                                    pathname='',
                                    lineno=0,
                                    msg=text,
                                    args=(),
                                    exc_info=None
                                )
                                record.confidence = confidence
                                
                                # Log with enhanced information
                                transcription_logger.handle(record)
                                
                                # Debug log
                                logger.debug(f"Raw transcription: {text}")
                                
                                self.transcription_queue.put(
                                    TranscriptionResult(
                                        text=text,
                                        timestamp=chunk.timestamp,
                                        confidence=confidence
                                    )
                                )
                                
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.exception("Full traceback:")
                
                # Clear processed chunks
                self.audio_buffer.clear_processed_chunks()
                
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Full traceback:")
                
            time.sleep(0.01)
            
    def _text_processing_worker(self):
        """Worker thread for text processing with improved accumulation"""
        text_buffer = []
        last_output_time = time.time()
        buffer_time = 2.0  # Accumulate for 2 seconds
        min_buffer_size = 5  # Minimum number of transcriptions to process
        
        while self.running:
            try:
                # Get new transcription with longer timeout
                try:
                    result = self.transcription_queue.get(timeout=0.1)
                    logger.debug(f"Added to buffer: {result.text}")
                    text_buffer.append(result)
                except queue.Empty:
                    # Process buffer if we've waited long enough without new input
                    current_time = time.time()
                    if text_buffer and (current_time - last_output_time) > buffer_time:
                        logger.debug("Processing buffer due to timeout")
                        self._process_buffer(text_buffer)
                        text_buffer = []
                        last_output_time = current_time
                    continue
                
                # Process buffer if we have enough transcriptions or enough time has passed
                current_time = time.time()
                if len(text_buffer) >= min_buffer_size or (current_time - last_output_time) > buffer_time:
                    logger.debug(f"Processing buffer with {len(text_buffer)} items")
                    self._process_buffer(text_buffer)
                    text_buffer = []
                    last_output_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in text processing worker: {e}")
                logger.exception("Full traceback:")
                text_buffer = []
                
    def _process_buffer(self, text_buffer: List[TranscriptionResult]) -> None:
        """Process accumulated transcriptions"""
        if not text_buffer:
            return
            
        # Sort by timestamp to maintain order
        text_buffer.sort(key=lambda x: x.timestamp)
        
        # Group transcriptions by time proximity
        groups = []
        current_group = [text_buffer[0]]
        
        for i in range(1, len(text_buffer)):
            current = text_buffer[i]
            previous = text_buffer[i-1]
            
            # If transcriptions are close in time, group them
            if current.timestamp - previous.timestamp < 0.5:  # 500ms threshold
                current_group.append(current)
            else:
                groups.append(current_group)
                current_group = [current]
        groups.append(current_group)
        
        # Process each group
        for group in groups:
            # Combine and clean text
            combined_text = self._clean_and_combine_text(group)
            
            if combined_text:
                logger.info(f"Combined text from {len(group)} transcriptions: {combined_text}")
                
                # Calculate average confidence
                avg_confidence = sum(r.confidence for r in group) / len(group)
                
                # Only output if confidence is high enough
                if avg_confidence > self.config["confidence_threshold"]:
                    self.output_queue.put(
                        TranscriptionResult(
                            text=combined_text,
                            timestamp=group[-1].timestamp,
                            confidence=avg_confidence
                        )
                    )
                    logger.debug("Added to output queue")

class Transcriber:
    """Wrapper class for backwards compatibility with existing code"""
    def __init__(self, model_path=None):
        # Initialize logging if not already done
        if not logging.getLogger().handlers:
            self.session_id = setup_logging()
            
        # Define model hierarchy based on quality and available memory
        model_hierarchy = {
            "high": {
                "name": "openai/whisper-medium",  # ~2.6GB
                "min_memory": 12e9,
                "chunk_size": 80000,
                "min_chunk_size": 40000
            },
            "medium": {
                "name": "openai/whisper-small",  # ~900MB
                "min_memory": 8e9,
                "chunk_size": 48000,
                "min_chunk_size": 24000
            },
            "low": {
                "name": "openai/whisper-base",  # ~400MB
                "min_memory": 4e9,
                "chunk_size": 32000,
                "min_chunk_size": 16000
            },
            "tiny": {
                "name": "openai/whisper-tiny",  # ~150MB
                "min_memory": 2e9,
                "chunk_size": 24000,
                "min_chunk_size": 12000
            }
        }
        
        # Device detection with MPS support
        if torch.cuda.is_available():
            device = "cuda"
            memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"Using CUDA GPU with {memory / 1e9:.1f}GB VRAM")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            # Apple Silicon typically has unified memory
            memory = psutil.virtual_memory().total * 0.7  # Use 70% of system RAM
            logger.info(f"Using Apple Silicon GPU (MPS) with {memory / 1e9:.1f}GB available memory")
        else:
            device = "cpu"
            memory = psutil.virtual_memory().total * 0.5  # Use 50% of system RAM
            logger.info(f"Using CPU with {memory / 1e9:.1f}GB available memory")
        
        # Select model based on available memory
        if memory >= model_hierarchy["high"]["min_memory"]:
            logger.info("Using high-quality model (Whisper Medium)")
            model_config = model_hierarchy["high"]
        elif memory >= model_hierarchy["medium"]["min_memory"]:
            logger.info("Using medium-quality model (Whisper Small)")
            model_config = model_hierarchy["medium"]
        elif memory >= model_hierarchy["low"]["min_memory"]:
            logger.info("Using base-quality model (Whisper Base)")
            model_config = model_hierarchy["low"]
        else:
            logger.info("Using tiny model (Whisper Tiny)")
            model_config = model_hierarchy["tiny"]
            
        # Configure pipeline
        config = {
            "model_name": model_path or model_config["name"],
            "device": device,
            "chunk_size": model_config["chunk_size"],
            "min_chunk_size": model_config["min_chunk_size"],
            "overlap": 16000,        # 1 second overlap
            "max_chunks": 3,
            "confidence_threshold": 0.5
        }
        
        logger.info(f"Selected model: {config['model_name']}")
        logger.info(f"Using device: {config['device']}")
        logger.info(f"Chunk size: {config['chunk_size'] / 16000:.1f}s")
        logger.info(f"Min chunk size: {config['min_chunk_size'] / 16000:.1f}s")
        
        self.pipeline = TranscriptionPipeline(config)
        self.pipeline.start()
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Process audio chunk using the pipeline"""
        return self.pipeline.process_chunk(audio_chunk)
        
    def __del__(self):
        """Ensure pipeline is stopped when object is destroyed"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
