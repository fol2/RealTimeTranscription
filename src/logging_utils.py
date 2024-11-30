import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform color support
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Simplified formatter for system messages"""
    COLORS = {
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        if record.levelname in self.COLORS and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return f"{self.COLORS[record.levelname]}{record.msg}{Style.RESET_ALL}"
        return record.msg

class TranscriptionFormatter(logging.Formatter):
    """Minimal formatter showing only transcribed text with proper line handling"""
    def __init__(self):
        super().__init__()
        self.last_length = 0
        self.current_line = ""
        
    def format(self, record):
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            text = record.msg
            padding = ' ' * max(self.last_length - len(text), 0)
            self.last_length = len(text)
            return f"\033[2K\r{Fore.BLUE}▶ {Style.RESET_ALL}{text}{padding}"
        return record.msg

class TranscriptionFileFormatter(logging.Formatter):
    """Clean formatter for log files"""
    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] {record.msg}"

def setup_logging():
    """Configure minimal logging system focused on transcriptions"""
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{timestamp}"
    
    # Configure root logger to WARNING to suppress most logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []
    
    # Console handler for system messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    
    # File handler for system logs
    system_file_handler = RotatingFileHandler(
        f"logs/system_{timestamp}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    system_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(system_file_handler)
    
    # Configure transcription logger
    transcription_logger = logging.getLogger('transcription')
    transcription_logger.setLevel(logging.INFO)
    transcription_logger.propagate = False
    
    # Console handler for transcriptions
    trans_console_handler = logging.StreamHandler()
    trans_console_handler.setFormatter(TranscriptionFormatter())
    transcription_logger.addHandler(trans_console_handler)
    
    # File handler for transcriptions
    trans_file_handler = RotatingFileHandler(
        f"logs/transcription_{timestamp}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    trans_file_handler.setFormatter(TranscriptionFileFormatter())
    transcription_logger.addHandler(trans_file_handler)
    
    # Suppress third-party logging
    for logger_name in ['transformers', 'torch', 'numpy', 'PIL', 'urllib3', 'requests']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Only show startup info
    logger = logging.getLogger(__name__)
    logger.info("Transcription system initialized")
    
    return session_id

# Get loggers
logger = logging.getLogger(__name__)
transcription_logger = logging.getLogger('transcription') 