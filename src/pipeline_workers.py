import torch
import numpy as np
import re
import logging
import time
import queue
import difflib
from typing import List, Optional
from datetime import datetime
from .text_processing import TranscriptionResult
from .config_manager import ConfigManager
import threading

logger = logging.getLogger(__name__)
transcription_logger = logging.getLogger('transcription')

class TranscriptionWorker:
    """Handles transcription processing in a separate thread"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = ConfigManager().get_pipeline_worker_config()
        self.current_line_text = ""  # Add this line
        self.previous_text = ""  # Initialize previous transcription
        self.lock = threading.Lock()  # Add this line

    def run(self):
        """Main transcription worker loop"""
        while self.pipeline.running:
            try:
                chunks = self.pipeline.audio_buffer.get_chunks()
                
                for chunk in chunks:
                    if chunk.is_speech:
                        try:
                            transcription, confidence = self._process_chunk(chunk)
                            
                            if transcription and transcription[0].strip():
                                text = self._clean_transcription(transcription[0])
                                
                                # Update transcription with sentence handling
                                current_text, is_complete = self.pipeline.transcription_handler.update_transcription(text)
                                
                                if current_text:
                                    new_words = self._get_new_words(self.previous_text, current_text)
                                    if new_words:
                                        # Create log record with only the new words
                                        record = logging.LogRecord(
                                            name='transcription',
                                            level=logging.INFO,
                                            pathname='',
                                            lineno=0,
                                            msg=new_words,
                                            args=(),
                                            exc_info=None
                                        )
                                        transcription_logger.handle(record)
                                    self._update_previous_text(current_text)  # Update previous_text
                                
                        except Exception as e:
                            logger.error(f"Transcription error: {str(e)}")
            
                self.pipeline.audio_buffer.clear_processed_chunks()
                
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
            
    def _process_chunk(self, chunk) -> tuple[Optional[List[str]], float]:
        """Process a single audio chunk"""
        if self.pipeline.is_whisper:
            # Normalize audio if configured
            audio_data = chunk.data
            if self.config['normalize_audio'] and np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Process audio for Whisper
            inputs = self.pipeline.processor(
                audio_data,
                sampling_rate=self.config['sampling_rate'],
                return_tensors="pt",
                return_attention_mask=True
            )
            
            input_features = inputs.input_features.to(self.pipeline.config["device"])
            attention_mask = inputs.attention_mask.to(self.pipeline.config["device"]) if hasattr(inputs, 'attention_mask') else None
            
            with torch.no_grad():
                generated_ids = self.pipeline.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    num_beams=self.config['num_beams'],
                    max_length=self.config['max_length'],
                    suppress_tokens=[1],
                    return_timestamps=False,
                    task="transcribe",
                    language="en",
                    no_repeat_ngram_size=self.config['no_repeat_ngram_size']
                )
                
                transcription = self.pipeline.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                confidence = self.config['default_confidence']
                
            return transcription, confidence
        return None, 0.0

    def _clean_transcription(self, text: str) -> str:
        """Clean up transcription text"""
        if not self.config['clean_text']:
            return text
            
        text = text.strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _log_transcription(self, text: str):
        """Log transcription to file"""
        if self.config['log_transcriptions']:
            with open(f"logs/transcription_{self.pipeline.session_id}.log", 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {text}\n")

    def _get_new_words(self, old_text, new_text):
        """Extract new words added to the transcription."""
        old_words = old_text.strip().split()
        new_words = new_text.strip().split()

        # Reverse the word lists to compare from the end
        index = 0
        while (index < len(old_words) and index < len(new_words) and
               old_words[-(index + 1)] == new_words[-(index + 1)]):
            index += 1

        # New words are from start to len(new_words) - index
        new_words = ' '.join(new_words[:len(new_words) - index]).strip()

        # If no new words are found, return an empty string
        return new_words
        # Implement the logic to process the text buffer if needed.
        # If not required, you can leave it as a pass.
        pass

    def _update_previous_text(self, new_text):
        with self.lock:
            self.previous_text = new_text

class TextProcessingWorker:
    """Handles text processing in a separate thread"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        """Process the accumulated text buffer."""
        # Implement the logic to process the text buffer if needed.
        # If not required, you can leave it as a pass.
        pass

class TextProcessingWorker:
    """Handles text processing in a separate thread"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = ConfigManager().get_pipeline_worker_config()
        self.text_buffer = []
        self.last_output_time = time.time()
        
    def run(self):
        """Main text processing worker loop"""
        while self.pipeline.running:
            try:
                try:
                    result = self.pipeline.transcription_queue.get(
                        timeout=self.config['queue_timeout']
                    )
                    self.text_buffer.append(result)
                except queue.Empty:
                    current_time = time.time()
                    if self.text_buffer and (current_time - self.last_output_time) > self.config['buffer_time']:
                        self._process_buffer()
                        self.text_buffer = []
                        self.last_output_time = current_time
                    continue
                
                current_time = time.time()
                if (len(self.text_buffer) >= self.config['min_buffer_size'] or 
                    (current_time - self.last_output_time) > self.config['buffer_time']):
                    self._process_buffer()
                    self.text_buffer = []
                    self.last_output_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in text processing: {e}")
                logger.exception("Full traceback:")
                self.text_buffer = []
        self.pipeline._process_buffer(self.text_buffer)
    def _process_buffer(self):
        """Process the accumulated text buffer."""
        # Implement the logic to process the text buffer if needed.
        # If not required, you can leave it as a pass.
        self.pipeline._process_buffer(self.text_buffer)