import spacy
from spacy.language import Language
from spacy.tokens import Doc
import logging
from typing import Tuple, Set
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class RealTimeTranscriptionHandler:
    """Handles real-time transcription with improved word handling"""
    def __init__(self):
        # Load configuration
        self.config = ConfigManager().get_realtime_handler_config()
        
        # Load spaCy with configured components
        self.nlp = spacy.load(
            self.config['spacy_model'], 
            disable=self.config['disable_components']
        )
        
        if self.config['enable_sentencizer'] and "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        
        self.current_sentence = ""
        self.word_buffer = []  # Store words with their raw form
        self.partial_word = ""  # Store incomplete word fragments
        
        # Load common English words for validation
        self.vocab = set(word.lower() for word in self.nlp.vocab.strings)
        self.common_prefixes = self.config['common_prefixes']
        self.common_suffixes = self.config['common_suffixes']
        
    def _is_valid_word_part(self, text: str) -> bool:
        """Check if text could be part of a valid word"""
        if not self.config['enable_word_validation']:
            return True
            
        if len(text) < self.config['min_word_length']:
            return False
            
        text = text.lower()
        # Check if it's a complete word
        if text in self.vocab:
            return True
        # Check if it's a common prefix
        if any(text.startswith(prefix) for prefix in self.common_prefixes):
            return True
        # Check if it could become a word with common suffixes
        if any(text + suffix in self.vocab for suffix in self.common_suffixes):
            return True
        # Check if it's the start of any word in vocabulary
        return any(word.startswith(text) for word in self.vocab if len(word) > len(text))
    
    def update_transcription(self, text: str) -> Tuple[str, bool]:
        """Update transcription with improved word handling"""
        # Clean the text if configured
        if self.config['strip_text']:
            text = text.strip()
        if not text:
            return "", False
        
        # Split into words, keeping potential partial words
        words = text.split()
        processed_words = []
        
        for word in words:
            # If we have a partial word and combining is enabled, try to combine
            if self.partial_word and self.config['combine_partial_words']:
                combined = self.partial_word + word
                if combined.lower() in self.vocab:
                    # Found a complete word
                    processed_words.append(combined)
                    self.partial_word = ""
                elif self._is_valid_word_part(combined):
                    # Still could be part of a word
                    self.partial_word = combined
                else:
                    # Not a valid combination, treat as separate
                    if self._is_valid_word_part(self.partial_word):
                        processed_words.append(self.partial_word)
                    if self._is_valid_word_part(word):
                        processed_words.append(word)
                    self.partial_word = ""
            else:
                if self._is_valid_word_part(word):
                    if word.lower() in self.vocab:
                        processed_words.append(word)
                    else:
                        self.partial_word = word
                
        # Build the current sentence
        if processed_words:
            combined_text = ' '.join(processed_words)
            if self.current_sentence:
                self.current_sentence = f"{self.current_sentence} {combined_text}".strip()
            else:
                self.current_sentence = combined_text

            # Use spaCy to detect sentences
            doc = self.nlp(self.current_sentence)
            sentences = list(doc.sents)

            if sentences:
                # Check if there's at least one complete sentence
                completed_sentences = [sent.text.strip() for sent in sentences[:-1]]
                self.current_sentence = sentences[-1].text.strip()

                if completed_sentences:
                    completed_text = ' '.join(completed_sentences)
                    return completed_text, True

        return self.current_sentence, False

    def reset(self):
        """Reset the handler state"""
        self.current_sentence = ""
        self.word_buffer = []
        self.partial_word = ""
