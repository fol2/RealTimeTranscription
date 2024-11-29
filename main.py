import logging
import time
from src.audio_stream import AudioStream
from src.transcriber import Transcriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Simplified format
)

logger = logging.getLogger(__name__)

def print_startup_message():
    print("\n=== Real-Time Transcription System ===")
    print("Make sure BlackHole 2ch is set up correctly:")
    print("1. Open System Settings > Sound")
    print("2. Set input device to 'BlackHole 2ch'")
    print("3. Route your audio to BlackHole 2ch")
    print("\nPress Ctrl+C to stop the transcription")
    print("=====================================\n")

def main():
    try:
        print_startup_message()
        
        # Initialize components
        audio_stream = AudioStream(sample_rate=16000, channels=1, chunk_size=4000)
        transcriber = Transcriber()
        
        # Start audio streaming
        audio_stream.start()
        
        # Main processing loop
        while True:
            # Get audio chunk
            audio_chunk = audio_stream.get_audio_chunk()
            
            # Process chunk
            if audio_chunk is not None:
                transcription = transcriber.process_chunk(audio_chunk)
                if transcription:
                    print("\n  Transcription:", transcription)
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\n  Stopping transcription...")
    except Exception as e:
        print(f"\n Error: {str(e)}")
    finally:
        if 'audio_stream' in locals():
            audio_stream.stop()

if __name__ == "__main__":
    main()
