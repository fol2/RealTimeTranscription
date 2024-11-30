import re
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import logging
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Container for transcription results with metadata"""
    text: str
    timestamp: float
    confidence: float

class TranscriptionLogger:
    """Handles logging and formatting of transcriptions"""
    def __init__(self):
        self.config = ConfigManager().get_text_processing_config()
        self.current_sentence = []
        self.last_phrase = ""
        
    def format_transcription(self, text: str) -> Optional[str]:
        """Format transcription for output, handling duplicates and partial phrases"""
        if self._is_similar(text, self.last_phrase):
            return None
            
        self.last_phrase = text
        cleaned = self._clean_text(text)
        if not cleaned:
            return None
            
        self.current_sentence.append(cleaned)
        
        if self._is_sentence_complete(cleaned):
            result = self._format_sentence()
            self.current_sentence = []
            return result
            
        return None
        
    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar using sequence matching"""
        if not text1 or not text2:
            return False
        return SequenceMatcher(None, text1, text2).ratio() > self.config['similarity_threshold']
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize transcribed text"""
        if self.config['strip_transcription_prefix']:
            text = re.sub(r"Transcription:\s*", "", text, flags=re.IGNORECASE)
            
        if self.config['uppercase_text']:
            text = text.upper()
            
        text = text.strip()
        text = re.sub(f'[^{self.config["preserve_chars"]}]', '', text)
        text = text.replace("'", "'")
        
        words = text.split()
        cleaned_words = []
        prev_word = None
        repetition_count = 0
        
        for word in words:
            if (len(word) < self.config['min_word_length'] and 
                word not in self.config['allowed_single_chars']):
                continue
                
            if word == prev_word:
                repetition_count += 1
                if repetition_count <= self.config['max_repetitions']:
                    cleaned_words.append(word)
            else:
                if prev_word and prev_word.startswith(word[:2]):
                    continue
                cleaned_words.append(word)
                repetition_count = 0
                prev_word = word
        
        cleaned_text = ' '.join(cleaned_words)
        
        # Remove configured artifacts
        for pattern, replacement in self.config['artifacts'].items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        return re.sub(r'\s+', ' ', cleaned_text).strip()
        
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text represents a complete sentence"""
        # Check for ending punctuation
        if any(text.strip().endswith(p) for p in self.config['sentence_end_chars']):
            return True
        
        # Check for configured ending phrases
        return any(text.strip().endswith(phrase) for phrase in self.config['ending_phrases'])
        
    def _format_sentence(self) -> str:
        """Format the complete sentence with timestamp"""
        full_text = " ".join(self.current_sentence)
        phrases = full_text.split()
        unique_phrases = []
        for phrase in phrases:
            if not unique_phrases or phrase != unique_phrases[-1]:
                unique_phrases.append(phrase)
                
        final_text = " ".join(unique_phrases)
        timestamp = datetime.now().strftime(self.config['timestamp_format'])
        
        return f"[{timestamp}] {final_text}" 