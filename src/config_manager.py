# src/config_manager.py

import configparser
import os
import logging
import torch
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration with INI file support"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
        
        # Set up attributes for commonly accessed configurations with alias mappings
        buffer_config = self.get_buffer_config()
        self.max_size = buffer_config['max_size'] 
        self.min_size = buffer_config['min_size']                 # Alias for 'min_buffer_size'
        self.overlap = buffer_config['overlap_size']              # Alias for 'overlap_size'
        self.overlap_size = buffer_config['overlap_size']         # Direct mapping
        self.chunk_size = buffer_config['chunk_size']             # Direct mapping
        self.min_chunk_size = buffer_config['min_chunk_size']     # Direct mapping
        self.max_chunks_memory = buffer_config['max_chunks_memory']  # Direct mapping
        
        # Transcription settings
        transcription_config = self.get_transcription_config()
        self.confidence_threshold = transcription_config['confidence_threshold']
        self.max_segment_duration = transcription_config['max_segment_duration']
        self.similarity_threshold = transcription_config['similarity_threshold']
        self.min_words_per_segment = transcription_config['min_words_per_segment']
        self.max_words_per_segment = transcription_config['max_words_per_segment']
        
        # Add more attribute assignments as needed based on your codebase's requirements
        
    def load_config(self) -> None:
        """Load configuration from INI file with comment handling"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._create_default_config()
        else:
            # Create a temporary string without comments for parsing
            temp_config = []
            with open(self.config_path, 'r') as f:
                for line in f:
                    # Remove inline comments but keep the value
                    if '=' in line and not line.strip().startswith('#'):
                        key_value = line.split('#')[0].strip()
                        temp_config.append(key_value)
                    # Keep section headers
                    elif line.strip().startswith('['):
                        temp_config.append(line)
                        
            # Create a temporary string with cleaned config
            clean_config = '\n'.join(temp_config)
            
            # Use StringIO to parse the cleaned config
            from io import StringIO
            self.config.read_string(clean_config)
            
    def _create_default_config(self) -> None:
        """Create default configuration file"""
        # Define default configuration without comments
        self.config.read_dict({
            'model': {
                'model_path': 'auto',
                'device': 'auto',
                'beam_size': '5',
                'language': 'en',
                'task': 'transcribe',
                'compute_type': 'float32',
                'cache_dir': 'models',
                'use_cached': 'true',
                'force_download': 'false',
                'show_chunk_info': 'true',
                'show_hardware_info': 'true',
                'high_model_name': 'openai/whisper-medium',
                'high_min_memory': '12000000000',
                'high_chunk_size': '32000',
                'high_min_chunk_size': '4000',
                'medium_model_name': 'openai/whisper-small',
                'medium_min_memory': '8000000000',
                'medium_chunk_size': '24000',
                'medium_min_chunk_size': '4000',
                'low_model_name': 'openai/whisper-base',
                'low_min_memory': '4000000000',
                'low_chunk_size': '16000',
                'low_min_chunk_size': '4000',
                'tiny_model_name': 'openai/whisper-tiny',
                'tiny_min_memory': '2000000000',
                'tiny_chunk_size': '8000',
                'tiny_min_chunk_size': '4000'
            },
            'audio': {
                'sample_rate': self.config.getint('audio', 'processing_sample_rate'),
                'channels': '1',
                'chunk_size': '4000',
                'input_device': 'default',
                'audio_scaling_factor': '32767',
                'normalization_enabled': 'true',
                'dc_offset_removal': 'true',
                'enabled': 'true',
                'aggressiveness': '1',
                'frame_duration_ms': '30',
                'padding_duration_ms': '500',
                'speech_threshold': '0.0005',
                'min_speech_duration_ms': '100',
                'max_silence_duration_ms': '1000',
                'vad_min_speech_frames': '2',
                'vad_initial_speech_probability': '0.0',
                'vad_speech_probability_threshold': '0.4',
                'vad_probability_decay': '0.7',
                'vad_probability_update': '0.3',
                'vad_silence_threshold_ratio': '0.3',
                'vad_min_speech_duration': '0.5',
                'ms_to_sec_factor': '1000',
                'vad_frame_size_factor': '1.0',
                'vad_padding_size_factor': '1.0',
                'vad_silence_frames_factor': '1.0',
                'frame_pack_format': 'h',
                'vad_speech_confidence': '1.0',
                'vad_silence_confidence': '0.0',
                'vad_speech_frame_decrease': '1.0',
                'vad_silence_frame_decrease': '1.0',
                'input_sample_rate': '48000',
                'processing_sample_rate': self.config.getint('audio', 'processing_sample_rate'),
                'resampling_enabled': 'true'
            },
            'buffer': {
                'max_buffer_size': self.config.getint('buffer', 'max_buffer_size'),
                'min_buffer_size': '80000',
                'chunk_size': '4000',
                'min_chunk_size': '2000',
                'overlap_size': '16000',
                'max_queue_size': '10',
                'transcription_queue_size': '10',
                'output_queue_size': '10',
                'silence_threshold_seconds': '0.3',
                'min_chunk_interval': '0.5',
                'buffer_maxlen': '80000',
                'max_chunks_memory': '3',
                'maintain_overlap': 'true',
                'clear_on_reset': 'true'
            },
            'transcription': {
                'confidence_threshold': '0.5',
                'max_segment_duration': '10.0',
                'similarity_threshold': '0.85',
                'min_words_per_segment': '3',
                'max_words_per_segment': '50',
                'remove_filler_words': 'true',
                'capitalize_sentences': 'true',
                'add_punctuation': 'true',
                'merge_short_segments': 'true'
            },
            'processing': {
                'buffer_time': '2.0',
                'min_buffer_size': '5',
                'max_buffer_size': '20',
                'parallel_processing': 'true',
                'num_workers': '2'
            },
            'logging': {
                'level': 'INFO',
                'max_file_size': '10',
                'backup_count': '5',
                'log_format': 'detailed',
                'include_timestamps': 'true',
                'log_to_console': 'true',
                'log_to_file': 'true'
            },
            'display': {
                'show_confidence': 'false',
                'show_timestamps': 'true',
                'colored_output': 'true',
                'live_wpm': 'true',
                'show_speaker_changes': 'true',
                'progress_bar': 'true'
            },
            'advanced': {
                'use_gpu_acceleration': 'auto',
                'gpu_memory_fraction': '0.7',
                'batch_size': '1',
                'thread_priority': 'normal',
                'enable_optimization': 'true',
                'debug_mode': 'false'
            },
            'device': {
                'cpu_memory_fraction': '0.5',
                'mps_memory_fraction': '0.7',
                'cuda_memory_limit': '0.9'
            },
            'audio_stream': {
                'default_sample_rate': self.config.getint('audio', 'input_sample_rate'),
                'default_channels': '2',
                'default_chunk_size': '4000',
                'queue_timeout': '0.1',
                'audio_level_threshold': '0.01',
                'format': 'float32',
                'device_name': 'BlackHole 2ch',
                'show_device_scan': 'true',
                'show_status': 'true',
                'show_detection': 'true'
            },
            'pipeline_workers': {
                'sampling_rate': self.config.getint('audio', 'processing_sample_rate'),
                'max_length': '448',
                'num_beams': '5',
                'no_repeat_ngram_size': '3',
                'default_confidence': '0.95',
                'normalize_audio': 'true',
                'queue_timeout': '0.1',
                'buffer_time': '2.0',
                'min_buffer_size': '5',
                'clean_text': 'true',
                'log_transcriptions': 'true'
            },
            'realtime_handler': {
                'spacy_model': 'en_core_web_sm',
                'disable_components': 'ner',
                'enable_sentencizer': 'true',
                'enable_word_validation': 'true',
                'min_word_length': '2',
                'common_prefixes': 'un,re,in,dis,pre,post,non,anti,semi,over',
                'common_suffixes': 'ing,ed,s,ly,tion,ment,ness,able,ible,ful',
                'strip_text': 'true',
                'combine_partial_words': 'true'
            },
            'text_processing': {
                'similarity_threshold': '0.85',
                'strip_transcription_prefix': 'true',
                'uppercase_text': 'true',
                'preserve_chars': r'A-Z-\s\'',
                'min_word_length': '1',
                'max_repetitions': '1',
                'allowed_single_chars': 'A,I',
                'sentence_end_chars': '.,!,?,;',
                'ending_phrases': 'OKAY,RIGHT,YOU KNOW,YOU SEE,THANK YOU',
                'artifacts': r'UM:,UH:,HMM:,AH:,ER:,LIKE\s+(?=\bLIKE\b):',
                'timestamp_format': '%%H:%%M:%%S'
            },
            'vad': {
                'enabled': 'true',
                'aggressiveness': '1',
                'frame_duration_ms': '30',
                'padding_duration_ms': '500',
                'speech_threshold': '0.0005',
                'min_speech_duration_ms': '100',
                'max_silence_duration_ms': '1000',
                'initial_probability': '0.0',
                'probability_threshold': '0.4',
                'probability_decay': '0.7',
                'probability_update': '0.3',
                'silence_probability': '0.3',
                'speech_confidence': '1.0',
                'silence_confidence': '0.0'
            }
        })
        
        # Write default configuration with comments
        with open(self.config_path, 'w') as f:
            # For brevity, comments are not included in the default config creation.
            # Implementers can populate the config.ini manually or enhance this method to include comments.
            self.config.write(f)
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        model_config = {
            'model_path': self.config.get('model', 'model_path'),
            'device': self._resolve_device(),
            'beam_size': self.config.getint('model', 'beam_size'),
            'language': self.config.get('model', 'language'),
            'task': self.config.get('model', 'task'),
            'compute_type': self.config.get('model', 'compute_type'),
            'cache_dir': self.config.get('model', 'cache_dir'),
            'use_cached': self.config.getboolean('model', 'use_cached'),
            'force_download': self.config.getboolean('model', 'force_download'),
            'show_chunk_info': self.config.getboolean('model', 'show_chunk_info'),
            'show_hardware_info': self.config.getboolean('model', 'show_hardware_info')
        }
        return model_config
    
    def _resolve_device(self) -> str:
        """Resolve device based on configuration and availability"""
        device = self.config.get('model', 'device').lower()
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def get_model_hierarchy_config(self) -> Dict[str, Dict[str, Any]]:
        """Get model hierarchy configuration"""
        return {
            "high": {
                "name": self.config.get('model', 'high_model_name'),
                "min_memory": self.config.getint('model', 'high_min_memory'),
                "chunk_size": self.config.getint('model', 'high_chunk_size'),
                "min_chunk_size": self.config.getint('model', 'high_min_chunk_size')
            },
            "medium": {
                "name": self.config.get('model', 'medium_model_name'),
                "min_memory": self.config.getint('model', 'medium_min_memory'),
                "chunk_size": self.config.getint('model', 'medium_chunk_size'),
                "min_chunk_size": self.config.getint('model', 'medium_min_chunk_size')
            },
            "low": {
                "name": self.config.get('model', 'low_model_name'),
                "min_memory": self.config.getint('model', 'low_min_memory'),
                "chunk_size": self.config.getint('model', 'low_chunk_size'),
                "min_chunk_size": self.config.getint('model', 'low_min_chunk_size')
            },
            "tiny": {
                "name": self.config.get('model', 'tiny_model_name'),
                "min_memory": self.config.getint('model', 'tiny_min_memory'),
                "chunk_size": self.config.getint('model', 'tiny_chunk_size'),
                "min_chunk_size": self.config.getint('model', 'tiny_min_chunk_size')
            }
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return {
            'sample_rate': self.config.getint('audio', 'processing_sample_rate'),
            'channels': self.config.getint('audio', 'channels'),
            'chunk_size': self.config.getint('audio', 'chunk_size'),
            'input_device': self.config.get('audio', 'input_device'),
            'audio_scaling_factor': self.config.getint('audio', 'audio_scaling_factor'),
            'normalization_enabled': self.config.getboolean('audio', 'normalization_enabled'),
            'dc_offset_removal': self.config.getboolean('audio', 'dc_offset_removal'),
            'enabled': self.config.getboolean('audio', 'enabled'),
            'aggressiveness': self.config.getint('audio', 'aggressiveness'),
            'frame_duration_ms': self.config.getint('audio', 'frame_duration_ms'),
            'padding_duration_ms': self.config.getint('audio', 'padding_duration_ms'),
            'speech_threshold': self.config.getfloat('audio', 'speech_threshold'),
            'min_speech_duration_ms': self.config.getint('audio', 'min_speech_duration_ms'),
            'max_silence_duration_ms': self.config.getint('audio', 'max_silence_duration_ms'),
            'vad_min_speech_frames': self.config.getint('audio', 'vad_min_speech_frames'),
            'vad_initial_speech_probability': self.config.getfloat('audio', 'vad_initial_speech_probability'),
            'vad_speech_probability_threshold': self.config.getfloat('audio', 'vad_speech_probability_threshold'),
            'vad_probability_decay': self.config.getfloat('audio', 'vad_probability_decay'),
            'vad_probability_update': self.config.getfloat('audio', 'vad_probability_update'),
            'vad_silence_threshold_ratio': self.config.getfloat('audio', 'vad_silence_threshold_ratio'),
            'vad_min_speech_duration': self.config.getfloat('audio', 'vad_min_speech_duration'),
            'ms_to_sec_factor': self.config.getint('audio', 'ms_to_sec_factor'),
            'vad_frame_size_factor': self.config.getfloat('audio', 'vad_frame_size_factor'),
            'vad_padding_size_factor': self.config.getfloat('audio', 'vad_padding_size_factor'),
            'vad_silence_frames_factor': self.config.getfloat('audio', 'vad_silence_frames_factor'),
            'frame_pack_format': self.config.get('audio', 'frame_pack_format'),
            'vad_speech_confidence': self.config.getfloat('audio', 'vad_speech_confidence'),
            'vad_silence_confidence': self.config.getfloat('audio', 'vad_silence_confidence'),
            'vad_speech_frame_decrease': self.config.getfloat('audio', 'vad_speech_frame_decrease'),
            'vad_silence_frame_decrease': self.config.getfloat('audio', 'vad_silence_frame_decrease'),
            'input_sample_rate': self.config.getint('audio', 'input_sample_rate'),
            'processing_sample_rate': self.config.getint('audio', 'processing_sample_rate'),
            'resampling_enabled': self.config.getboolean('audio', 'resampling_enabled')
        }
    
    def get_buffer_config(self) -> Dict[str, Any]:
        """Get buffer configuration with alias mappings"""
        return {
            'max_buffer_size': self.config.getint('buffer', 'max_buffer_size'),
            'min_buffer_size': self.config.getint('buffer', 'min_buffer_size'),
            'chunk_size': self.config.getint('buffer', 'chunk_size'),
            'min_chunk_size': self.config.getint('buffer', 'min_chunk_size'),
            'overlap_size': self.config.getint('buffer', 'overlap_size'),
            'max_queue_size': self.config.getint('buffer', 'max_queue_size'),
            'transcription_queue_size': self.config.getint('buffer', 'transcription_queue_size'),
            'output_queue_size': self.config.getint('buffer', 'output_queue_size'),
            'silence_threshold_seconds': self.config.getfloat('buffer', 'silence_threshold_seconds'),
            'min_chunk_interval': self.config.getfloat('buffer', 'min_chunk_interval'),
            'buffer_maxlen': self.config.getint('buffer', 'buffer_maxlen'),
            'max_chunks_memory': self.config.getint('buffer', 'max_chunks_memory'),
            'maintain_overlap': self.config.getboolean('buffer', 'maintain_overlap'),
            'clear_on_reset': self.config.getboolean('buffer', 'clear_on_reset'),
            'max_size': self.config.getint('buffer', 'max_buffer_size'),       # Alias for 'max_buffer_size'
            'min_size': self.config.getint('buffer', 'min_buffer_size'),       # Alias for 'min_buffer_size'
            'overlap': self.config.getint('buffer', 'overlap_size'),           # Alias for 'overlap_size'
            'max_chunks': self.config.getint('buffer', 'max_chunks_memory')    # Alias for 'max_chunks_memory'
        }
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription configuration"""
        return {
            'confidence_threshold': self.config.getfloat('transcription', 'confidence_threshold'),
            'max_segment_duration': self.config.getfloat('transcription', 'max_segment_duration'),
            'similarity_threshold': self.config.getfloat('transcription', 'similarity_threshold'),
            'min_words_per_segment': self.config.getint('transcription', 'min_words_per_segment'),
            'max_words_per_segment': self.config.getint('transcription', 'max_words_per_segment'),
            'remove_filler_words': self.config.getboolean('transcription', 'remove_filler_words'),
            'capitalize_sentences': self.config.getboolean('transcription', 'capitalize_sentences'),
            'add_punctuation': self.config.getboolean('transcription', 'add_punctuation'),
            'merge_short_segments': self.config.getboolean('transcription', 'merge_short_segments')
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration"""
        return {
            'buffer_time': self.config.getfloat('processing', 'buffer_time'),
            'min_buffer_size': self.config.getint('processing', 'min_buffer_size'),
            'max_buffer_size': self.config.getint('processing', 'max_buffer_size'),
            'parallel_processing': self.config.getboolean('processing', 'parallel_processing'),
            'num_workers': self.config.getint('processing', 'num_workers')
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': self.config.get('logging', 'level'),
            'max_file_size': self.config.getint('logging', 'max_file_size'),
            'backup_count': self.config.getint('logging', 'backup_count'),
            'log_format': self.config.get('logging', 'log_format'),
            'include_timestamps': self.config.getboolean('logging', 'include_timestamps'),
            'log_to_console': self.config.getboolean('logging', 'log_to_console'),
            'log_to_file': self.config.getboolean('logging', 'log_to_file')
        }
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return {
            'show_confidence': self.config.getboolean('display', 'show_confidence'),
            'show_timestamps': self.config.getboolean('display', 'show_timestamps'),
            'colored_output': self.config.getboolean('display', 'colored_output'),
            'live_wpm': self.config.getboolean('display', 'live_wpm'),
            'show_speaker_changes': self.config.getboolean('display', 'show_speaker_changes'),
            'progress_bar': self.config.getboolean('display', 'progress_bar')
        }
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration settings"""
        return {
            'use_gpu_acceleration': self.config.get('advanced', 'use_gpu_acceleration'),
            'gpu_memory_fraction': self.config.getfloat('advanced', 'gpu_memory_fraction'),
            'batch_size': self.config.getint('advanced', 'batch_size'),
            'thread_priority': self.config.get('advanced', 'thread_priority'),
            'enable_optimization': self.config.getboolean('advanced', 'enable_optimization'),
            'debug_mode': self.config.getboolean('advanced', 'debug_mode')
        }
    
    def get_device_config(self) -> Dict[str, float]:
        """Get device configuration"""
        return {
            'cpu_memory_fraction': self.config.getfloat('device', 'cpu_memory_fraction'),
            'mps_memory_fraction': self.config.getfloat('device', 'mps_memory_fraction'),
            'cuda_memory_limit': self.config.getfloat('device', 'cuda_memory_limit')
        }
    
    def get_audio_stream_config(self) -> Dict[str, Any]:
        """Get audio stream configuration"""
        return {
            'default_sample_rate': self.config.getint('audio', 'input_sample_rate'),
            'default_channels': self.config.getint('audio_stream', 'default_channels'),
            'default_chunk_size': self.config.getint('audio_stream', 'default_chunk_size'),
            'queue_timeout': self.config.getfloat('audio_stream', 'queue_timeout'),
            'audio_level_threshold': self.config.getfloat('audio_stream', 'audio_level_threshold'),
            'format': self.config.get('audio_stream', 'format'),
            'device_name': self.config.get('audio_stream', 'device_name'),
            'show_device_scan': self.config.getboolean('audio_stream', 'show_device_scan'),
            'show_status': self.config.getboolean('audio_stream', 'show_status'),
            'show_detection': self.config.getboolean('audio_stream', 'show_detection')
        }
    
    def get_pipeline_worker_config(self) -> Dict[str, Any]:
        """Get pipeline worker configuration"""
        return {
            'sampling_rate': self.config.getint('pipeline_workers', 'sampling_rate'),
            'max_length': self.config.getint('pipeline_workers', 'max_length'),
            'num_beams': self.config.getint('pipeline_workers', 'num_beams'),
            'no_repeat_ngram_size': self.config.getint('pipeline_workers', 'no_repeat_ngram_size'),
            'default_confidence': self.config.getfloat('pipeline_workers', 'default_confidence'),
            'normalize_audio': self.config.getboolean('pipeline_workers', 'normalize_audio'),
            'queue_timeout': self.config.getfloat('pipeline_workers', 'queue_timeout'),
            'buffer_time': self.config.getfloat('pipeline_workers', 'buffer_time'),
            'min_buffer_size': self.config.getint('pipeline_workers', 'min_buffer_size'),
            'clean_text': self.config.getboolean('pipeline_workers', 'clean_text'),
            'log_transcriptions': self.config.getboolean('pipeline_workers', 'log_transcriptions')
        }
    
    def get_realtime_handler_config(self) -> Dict[str, Any]:
        """Get realtime handler configuration"""
        prefixes = self.config.get('realtime_handler', 'common_prefixes').split(',')
        suffixes = self.config.get('realtime_handler', 'common_suffixes').split(',')
        
        return {
            'spacy_model': self.config.get('realtime_handler', 'spacy_model'),
            'disable_components': [comp.strip() for comp in self.config.get('realtime_handler', 'disable_components').split(',')],
            'enable_sentencizer': self.config.getboolean('realtime_handler', 'enable_sentencizer'),
            'enable_word_validation': self.config.getboolean('realtime_handler', 'enable_word_validation'),
            'min_word_length': self.config.getint('realtime_handler', 'min_word_length'),
            'common_prefixes': set(prefixes),
            'common_suffixes': set(suffixes),
            'strip_text': self.config.getboolean('realtime_handler', 'strip_text'),
            'combine_partial_words': self.config.getboolean('realtime_handler', 'combine_partial_words')
        }
    
    def get_text_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration"""
        ending_phrases = self.config.get('text_processing', 'ending_phrases').split(',')
        allowed_chars = self.config.get('text_processing', 'allowed_single_chars').split(',')
        sentence_end_chars = self.config.get('text_processing', 'sentence_end_chars').split(',')
        
        # Parse artifacts into dictionary
        artifacts_str = self.config.get('text_processing', 'artifacts')
        artifacts = {}
        for pair in artifacts_str.split(','):
            if ':' in pair:
                pattern, replacement = pair.split(':', 1)
                artifacts[pattern] = replacement
        
        preserve_chars = self.config.get('text_processing', 'preserve_chars').replace('\\', '\\\\')
        
        # Get timestamp format and convert %% to % for strftime
        timestamp_format = self.config.get('text_processing', 'timestamp_format')
        if timestamp_format.startswith('%%'):
            timestamp_format = timestamp_format.replace('%%', '%')
        
        return {
            'similarity_threshold': self.config.getfloat('text_processing', 'similarity_threshold'),
            'strip_transcription_prefix': self.config.getboolean('text_processing', 'strip_transcription_prefix'),
            'uppercase_text': self.config.getboolean('text_processing', 'uppercase_text'),
            'preserve_chars': preserve_chars,
            'min_word_length': self.config.getint('text_processing', 'min_word_length'),
            'max_repetitions': self.config.getint('text_processing', 'max_repetitions'),
            'allowed_single_chars': set(allowed_chars),
            'sentence_end_chars': set(sentence_end_chars),
            'ending_phrases': set(ending_phrases),
            'artifacts': artifacts,
            'timestamp_format': timestamp_format  # Now properly formatted for strftime
        }
    
    def get_vad_config(self) -> Dict[str, Any]:
        """Get VAD configuration"""
        return {
            'enabled': self.config.getboolean('vad', 'enabled'),
            'aggressiveness': self.config.getint('vad', 'aggressiveness'),
            'frame_duration_ms': self.config.getint('vad', 'frame_duration_ms'),
            'padding_duration_ms': self.config.getint('vad', 'padding_duration_ms'),
            'speech_threshold': self.config.getfloat('vad', 'speech_threshold'),
            'min_speech_duration_ms': self.config.getint('vad', 'min_speech_duration_ms'),
            'max_silence_duration_ms': self.config.getint('vad', 'max_silence_duration_ms'),
            'initial_probability': self.config.getfloat('vad', 'initial_probability'),
            'probability_threshold': self.config.getfloat('vad', 'probability_threshold'),
            'probability_decay': self.config.getfloat('vad', 'probability_decay'),
            'probability_update': self.config.getfloat('vad', 'probability_update'),
            'silence_probability': self.config.getfloat('vad', 'silence_probability'),
            'speech_confidence': self.config.getfloat('vad', 'speech_confidence'),
            'silence_confidence': self.config.getfloat('vad', 'silence_confidence')
        }
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations"""
        return {
            'model': self.get_model_config(),
            'model_hierarchy': self.get_model_hierarchy_config(),
            'audio': self.get_audio_config(),
            'buffer': self.get_buffer_config(),
            'transcription': self.get_transcription_config(),
            'processing': self.get_processing_config(),
            'logging': self.get_logging_config(),
            'display': self.get_display_config(),
            'advanced': self.get_advanced_config(),
            'device': self.get_device_config(),
            'audio_stream': self.get_audio_stream_config(),
            'pipeline_workers': self.get_pipeline_worker_config(),
            'realtime_handler': self.get_realtime_handler_config(),
            'text_processing': self.get_text_processing_config(),
            'vad': self.get_vad_config()
        }
    
    def get_model_settings(self) -> Dict[str, Any]:
        """Proxy method to retain backward compatibility for get_model_settings"""
        return self.get_model_config()
    
    def get_buffer_management_config(self) -> Dict[str, Any]:
        """Get buffer management configuration"""
        return {
            'min_chunk_interval': self.config.getfloat('buffer', 'min_chunk_interval'),
            'buffer_maxlen': self.config.getint('buffer', 'buffer_maxlen'),
            'overlap_size': self.config.getint('buffer', 'overlap_size'),
            'overlap': self.config.getint('buffer', 'overlap_size'),           # Alias for overlap_size
            'max_chunks_memory': self.config.getint('buffer', 'max_chunks_memory'),
            'max_chunks': self.config.getint('buffer', 'max_chunks_memory'),  # Alias for max_chunks_memory
            'maintain_overlap': self.config.getboolean('buffer', 'maintain_overlap'),
            'clear_on_reset': self.config.getboolean('buffer', 'clear_on_reset'),
            'max_size': self.config.getint('buffer', 'max_buffer_size'),
            'min_size': self.config.getint('buffer', 'min_buffer_size')
        }