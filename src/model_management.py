import torch
import psutil
import logging
import os
from typing import Dict, Optional, Any
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ModelConfig:
    """Manages model configuration and selection based on available resources"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.MODEL_HIERARCHY = self.config_manager.get_model_hierarchy_config()
        self._device = None
        self._memory = None

    def get_device_info(self) -> tuple[str, float]:
        """Detect available device and memory"""
        # Only detect device info if not already done
        if self._device is None or self._memory is None:
            config_manager = ConfigManager()
            device_config = config_manager.get_device_config()
            
            if torch.cuda.is_available():
                self._device = "cuda"
                total_memory = torch.cuda.get_device_properties(0).total_memory
                self._memory = total_memory * device_config['cuda_memory_limit']
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
                total_memory = psutil.virtual_memory().total
                self._memory = total_memory * device_config['mps_memory_fraction']
            else:
                self._device = "cpu"
                total_memory = psutil.virtual_memory().total
                self._memory = total_memory * device_config['cpu_memory_fraction']
            
        return self._device, self._memory

    @classmethod
    def get_model_config(cls, model_path: Optional[str] = None) -> Dict:
        """Get model configuration based on available resources"""
        instance = cls()
        device, memory = instance.get_device_info()
        
        # Get model settings and audio config
        model_settings = instance.config_manager.get_model_settings()
        audio_config = instance.config_manager.get_audio_config()
        sample_rate = audio_config['sample_rate']  # Get sample rate from config
        
        # Create a special logger for configuration display
        config_logger = logging.getLogger('config_display')
        config_logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        config_logger.addHandler(handler)
        config_logger.propagate = False
        
        # Select model based on available memory
        if model_path:
            selected_config = {
                "model_path": model_path,
                "device": device,
                "cache_dir": model_settings['cache_dir'],
                "use_cached": model_settings['use_cached'],
                "force_download": model_settings['force_download']
            }
            model_name = model_path.split('/')[-1].title()
        else:
            # Select model based on available memory
            if memory >= instance.MODEL_HIERARCHY["high"]["min_memory"]:
                model_config = instance.MODEL_HIERARCHY["high"]
                model_name = "Whisper Medium"
            elif memory >= instance.MODEL_HIERARCHY["medium"]["min_memory"]:
                model_config = instance.MODEL_HIERARCHY["medium"]
                model_name = "Whisper Small"
            elif memory >= instance.MODEL_HIERARCHY["low"]["min_memory"]:
                model_config = instance.MODEL_HIERARCHY["low"]
                model_name = "Whisper Base"
            else:
                model_config = instance.MODEL_HIERARCHY["tiny"]
                model_name = "Whisper Tiny"
            
            selected_config = {
                "model_path": model_config["name"],
                "device": device,
                "cache_dir": model_settings['cache_dir'],
                "use_cached": model_settings['use_cached'],
                "force_download": model_settings['force_download']
            }
            
        # Print consolidated system information
        config_logger.info("=== System Configuration ===")
        
        # Hardware Information (if enabled)
        if model_settings['show_hardware_info']:
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                config_logger.info("Hardware:")
                config_logger.info(f"✓ GPU: {gpu_name}")
                config_logger.info(f"✓ CUDA: {torch.version.cuda}")
                config_logger.info(f"✓ Memory: {memory / 1e9:.1f}GB VRAM")
            elif device == "mps":
                config_logger.info("Hardware:")
                config_logger.info(f"✓ GPU: Apple Silicon")
                config_logger.info(f"✓ Backend: MPS")
                config_logger.info(f"✓ Memory: {memory / 1e9:.1f}GB Unified")
            else:
                cpu_count = psutil.cpu_count(logical=False)
                cpu_threads = psutil.cpu_count(logical=True)
                config_logger.info("Hardware:")
                config_logger.info(f"✓ CPU: {cpu_count} cores ({cpu_threads} threads)")
                config_logger.info(f"✓ Memory: {memory / 1e9:.1f}GB RAM")

        # Model Information
        config_logger.info("\nModel Configuration:")
        config_logger.info(f"✓ Type: {model_name}")
        config_logger.info(f"✓ Path: {selected_config['model_path']}")
        if not model_path and model_settings['show_chunk_info']:
            config_logger.info(f"✓ Chunk Size: {model_config['chunk_size'] / sample_rate:.1f}s")
            config_logger.info(f"✓ Min Chunk: {model_config['min_chunk_size'] / sample_rate:.1f}s")
        
        config_logger.info("===========================")
        
        config_logger.removeHandler(handler)
        
        return selected_config

class ModelManager:
    """Handles model loading and initialization"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.is_whisper = False
        
    def load_model(self):
        try:
            model_path = self.config.get('model_path')
            if not model_path:
                raise ValueError("No model path specified in configuration")
                
            # Use configured cache directory
            cache_dir = os.path.join(os.getcwd(), self.config.get('cache_dir', 'models'))
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Using cache directory: {cache_dir}")
            
            # Common kwargs for model loading
            load_kwargs = {
                'cache_dir': cache_dir,
                'local_files_only': self.config.get('use_cached', True),
                'force_download': self.config.get('force_download', False)
            }
            
            if 'whisper' in model_path.lower():
                self.is_whisper = True
                self.processor = WhisperProcessor.from_pretrained(model_path, **load_kwargs)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
            else:
                self.is_whisper = False
                self.processor = Wav2Vec2Processor.from_pretrained(model_path, **load_kwargs)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path, **load_kwargs)
                
            device = self.config.get('device', 'cpu')
            self.model = self.model.to(device)
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Config used: {self.config}")
            raise