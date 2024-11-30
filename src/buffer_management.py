import threading
from collections import deque
import time
import numpy as np
from typing import List, Optional
from .audio_processing import AudioSegment
from .config_manager import ConfigManager

class StreamBuffer:
    """Enhanced thread-safe buffer with smart chunking strategy"""
    def __init__(self, 
                 max_size: Optional[int] = None,
                 min_size: Optional[int] = None,
                 overlap: Optional[int] = None,
                 max_chunks: Optional[int] = None):
        """Initialize buffer with configurable settings"""
        # Load configuration
        self.config = ConfigManager().get_buffer_management_config()
        
        # Use provided values or defaults from config
        self.max_size = max_size or self.config['buffer_maxlen']
        self.min_size = min_size or (self.max_size // 2)  # Default to half max_size if not provided
        self.overlap = overlap or self.config['overlap_size']
        self.max_chunks = max_chunks or self.config['max_chunks_memory']
        
        # Initialize buffers
        self.buffer = deque(maxlen=self.max_size)
        self.chunks = deque(maxlen=self.max_chunks)
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
            min_chunk_interval = self.config['min_chunk_interval']
            
            # Create new chunk if we have enough data and enough time has passed
            if (len(self.buffer) >= self.min_size and 
                (current_time - self.last_chunk_time) >= min_chunk_interval):
                
                # Create new chunk with overlap
                chunk_data = np.array(list(self.buffer))
                self.chunks.append(
                    AudioSegment(
                        data=chunk_data,
                        timestamp=segment.timestamp,
                        is_speech=segment.is_speech
                    )
                )
                
                # Keep overlap portion if configured
                if self.config['maintain_overlap'] and len(self.buffer) > self.overlap:
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
        """Clear processed chunks but maintain overlap if configured"""
        with self.lock:
            if self.config['maintain_overlap']:
                # Keep the last chunk for overlap
                last_chunk = self.chunks[-1] if self.chunks else None
                self.chunks.clear()
                if last_chunk:
                    self.chunks.append(last_chunk)
            else:
                self.chunks.clear()

    def reset(self):
        """Reset buffer state"""
        with self.lock:
            if self.config['clear_on_reset']:
                self.buffer.clear()
                self.chunks.clear()
            self.last_chunk_time = 0
        