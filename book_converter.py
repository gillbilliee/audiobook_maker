import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import re
import shutil
from striprtf.striprtf import rtf_to_text
from pathlib import Path
import tempfile
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tts_sdkokoro import TTSSDKokoro
import argparse
import sys
import scipy.io.wavfile as wav

# Try to import optional dependencies
try:
    import mobi
    MOBI_AVAILABLE = True
except ImportError:
    MOBI_AVAILABLE = False
    print("Note: MOBI/PRC support is not available. Install with: pip install mobi")

# Check for Calibre's ebook-convert for LIT support
CALIBRE_AVAILABLE = False
if shutil.which('ebook-convert') is not None:
    CALIBRE_AVAILABLE = True
else:
    print("Note: LIT support requires Calibre's ebook-convert to be in PATH. Install from calibre-ebook.com")

class BookConverter:
    def __init__(self, tts: TTSSDKokoro):
        """
        Initialize the book converter with a TTS engine.
        
        :param tts: Instance of TTSSDKokoro for text-to-speech conversion
        """
        self.tts = tts
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks at sentence boundaries (periods),
        but only split when we're close to the maximum chunk size.
        Each chunk will contain complete sentences.

        :param text: Input text to split
        :return: List of text chunks
        """
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Split text into sentences using periods
        sentences = text.split('.')
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Maximum characters per chunk
        max_chunk_size = 5000
        
        # Add a buffer to prevent splitting too early
        split_buffer = 1000
        
        for sentence in sentences:
            # Add back the period we removed
            if sentence:
                sentence += '.'
            
            # Calculate new length if we add this sentence
            new_length = current_length + len(sentence)
            
            # Only split if we're close to max size
            if new_length > max_chunk_size - split_buffer and current_chunk:
                # Add current chunk to chunks
                chunks.append(''.join(current_chunk))
                # Start new chunk
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length = new_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"Trying to open PDF at: {os.path.abspath(pdf_path)}")
        print(f"Exists? → {os.path.exists(pdf_path)}")
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text




    def extract_text_from_epub(self, epub_path: str) -> str:
        """
        Extract text from an EPUB file.
        
        :param epub_path: Path to the EPUB file
        :return: Extracted text
        """
        book = epub.read_epub(epub_path)
        text = ""
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
        
        return text
        
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from a plain text file.
        
        :param txt_path: Path to the text file
        :return: Extracted text
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Fallback to system default encoding if UTF-8 fails
            with open(txt_path, 'r') as file:
                return file.read()
                
    def extract_text_from_rtf(self, rtf_path: str) -> str:
        """
        Extract text from an RTF file.
        
        :param rtf_path: Path to the RTF file
        :return: Extracted text
        """
        try:
            with open(rtf_path, 'r', encoding='utf-8') as file:
                rtf_text = file.read()
                return rtf_to_text(rtf_text)
        except Exception as e:
            print(f"Error extracting text from RTF: {e}")
            raise
            
    def extract_text_from_mobi(self, mobi_path: str) -> str:
        """
        Extract text from MOBI/PRC file using mobi library.
        
        :param mobi_path: Path to the MOBI/PRC file
        :return: Extracted text
        :raises ImportError: If mobi library is not installed
        """
        if not MOBI_AVAILABLE:
            raise ImportError("MOBI/PRC support requires the 'mobi' package. Install with: pip install mobi")
            
        try:
            # mobi library handles both MOBI and PRC formats
            temp_dir, filepath = mobi.extract(mobi_path)
            try:
                # Try to find the HTML file in the extracted directory
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.html', '.htm')):
                            html_file = os.path.join(root, file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                soup = BeautifulSoup(f.read(), 'html.parser')
                                return soup.get_text()
                # If no HTML file found, try to find a text file
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            txt_file = os.path.join(root, file)
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                return f.read()
                raise ValueError("No readable content found in MOBI/PRC file")
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error extracting text from MOBI/PRC: {e}")
            raise
            
    def extract_text_from_lit(self, lit_path: str) -> str:
        """
        Extract text from LIT file by converting to MOBI first using calibre's ebook-convert.
        
        :param lit_path: Path to the LIT file
        :return: Extracted text
        :raises RuntimeError: If Calibre's ebook-convert is not available
        """
        if not CALIBRE_AVAILABLE:
            raise RuntimeError("LIT support requires Calibre's ebook-convert to be in PATH. Install from calibre-ebook.com")
            
        try:
            # Create a temporary file for the MOBI output
            with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as temp_mobi:
                temp_mobi_path = temp_mobi.name
            
            # Convert LIT to MOBI using calibre's ebook-convert
            import subprocess
            try:
                subprocess.run(
                    ['ebook-convert', lit_path, temp_mobi_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Now extract text from the MOBI file
                if not os.path.exists(temp_mobi_path):
                    raise RuntimeError("Failed to convert LIT to MOBI: No output file was created")
                    
                return self.extract_text_from_mobi(temp_mobi_path)
            except subprocess.CalledProcessError as e:
                error_msg = f"Error converting LIT to MOBI: {e.stderr or e.stdout or 'Unknown error'}"
                raise RuntimeError(error_msg) from e
            finally:
                # Clean up temporary MOBI file if it exists
                if os.path.exists(temp_mobi_path):
                    try:
                        os.unlink(temp_mobi_path)
                    except Exception as e:
                        print(f"Warning: Failed to clean up temporary file {temp_mobi_path}: {e}")
        except Exception as e:
            print(f"Error processing LIT file: {e}")
            raise

    def split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Split `text` into chunks at sentence boundaries (., !, ?) without ever breaking a word.

        :param text: full text to split
        :param max_chunk_size: approximate maximum characters per chunk
        :return: list of chunks, each ending on a sentence-boundary
        """
        # 1) Normalize whitespace: collapse multiple spaces/newlines into single space
        text = re.sub(r'\s+', ' ', text).strip()

        # 2) Use regex to locate sentence boundaries:
        #    We split on “(?<=[.!?]) +” so that each piece ends in ., !, or ? (and we keep the space).
        sentences = re.split(r'(?<=[\.\!\?])\s+', text)
        #    Example: "Hello world. How are you? I'm fine!"
        #    → ["Hello world.", "How are you?", "I'm fine!"]

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # If a single sentence is already longer than max_chunk_size,
            # we still keep it whole (to avoid cutting a word). That one chunk will exceed max_chunk_size.
            if len(sentence) > max_chunk_size:
                #  a) First, push whatever is in current_chunk onto the list
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                #  b) Push the oversized sentence as its own chunk
                chunks.append(sentence.strip())
                continue

            # If adding this sentence would exceed max_chunk_size,
            # start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Otherwise, append it to the current chunk (with a space)
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add the final chunk (if any)
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_file_extension(self, filepath: str) -> str:
        """Get the lowercase file extension, handling hidden extensions"""
        # Split the filename and handle multiple dots
        base, ext = os.path.splitext(filepath)
        # If no extension found, try to get the last part after dot
        if not ext and '.' in os.path.basename(filepath):
            ext = os.path.basename(filepath).split('.')[-1]
        return ext.lower().lstrip('.')
    
    def process_book(self, book_path: str, output_path: str, format: str = "wav") -> str:
        """
        Convert a book to audio.
        
        :param book_path: Path to the input book
        :param output_path: Path where the audio file will be saved
        :param format: Output audio format (wav, mp3, etc.)
        :return: Path to the generated audio file
        """
        # Get file extension in a case-insensitive way
        ext = self.get_file_extension(book_path)
        
        try:
            if ext == 'pdf':
                text = self.extract_text_from_pdf(book_path)
            elif ext == 'epub':
                text = self.extract_text_from_epub(book_path)
            elif ext == 'txt':
                text = self.extract_text_from_txt(book_path)
            elif ext == 'rtf':
                text = self.extract_text_from_rtf(book_path)
            elif ext in ('mobi', 'prc'):
                if not MOBI_AVAILABLE:
                    raise ImportError("MOBI/PRC support requires the 'mobi' package. Install with: pip install mobi")
                text = self.extract_text_from_mobi(book_path)
            elif ext == 'lit':
                if not CALIBRE_AVAILABLE:
                    raise RuntimeError("LIT support requires Calibre's ebook-convert to be in PATH. Install from calibre-ebook.com")
                text = self.extract_text_from_lit(book_path)
            else:
                raise ValueError(f"Unsupported file format: .{ext}. Supported formats are: PDF, EPUB, TXT, RTF" + 
                               (", MOBI, PRC" if MOBI_AVAILABLE else "") + 
                               (", LIT" if CALIBRE_AVAILABLE else ""))
        except Exception as e:
            print(f"Error processing {book_path}: {str(e)}")
            raise

        # Split text into chunks at sentence boundaries
        chunks = self.split_text_into_chunks(text)
        audio_chunks = []
        temp_files = []  # Track all temporary files for cleanup
        
        # Process all chunks
        all_audio = []
        sample_rate = None
        
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i + 1}/2...")
            print(f"Chunk text: {chunk[:50]}...")  # Show first 50 chars
            try:
                # Get audio data directly
                audio_data, sample_rate = self.tts.synthesize(chunk)
                all_audio.append(audio_data)
                print(f"Chunk {i + 1} processed successfully")
            except Exception as e:
                print(f"Error processing chunk {i + 1}: {str(e)}")
                continue
        
        # Combine audio data if we have any
        if all_audio and sample_rate:
            print("\nCombining audio chunks...")
            # Combine audio arrays
            combined_audio = np.concatenate(all_audio, axis=0)
            print(f"Combined audio shape: {combined_audio.shape}")
            print(f"Combined audio min/max: {combined_audio.min()}, {combined_audio.max()}")
            
            # Prepare output path
            output_path = Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)

            # Add file extension if not present
            if not output_path.suffix:
                output_path = output_path.with_suffix('.wav')

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Writing to output file: {output_path}")
            
            # Use wave module for direct WAV writing
            import wave
            
            # Convert to 16-bit integer format
            audio_data = combined_audio.astype(np.int16)
            
            # Open WAV file
            with wave.open(str(output_path), 'wb') as wav_file:
                # Set WAV parameters
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Write audio data
                wav_file.writeframes(audio_data.tobytes())
            
            print(f"Audio saved to: {output_path}")
        else:
            print("No audio chunks were processed successfully")

        return str(output_path)

    def __del__(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob('*'):
                file.unlink()
            self.temp_dir.rmdir()


def ensure_directory(path: Path):
    """Ensure directory exists, create if it doesn't"""
    path.mkdir(parents=True, exist_ok=True)
    return path

import random

# Voice options
AMERICAN_FEMALE_VOICES = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
    'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky'
]

BRITISH_FEMALE_VOICES = [
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily'
]

AMERICAN_MALE_VOICES = [
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
    'am_liam', 'am_michael', 'am_onyx', 'am_puck'
]

BRITISH_MALE_VOICES = [
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'
]

ALL_VOICES = (
    AMERICAN_FEMALE_VOICES +
    BRITISH_FEMALE_VOICES +
    AMERICAN_MALE_VOICES +
    BRITISH_MALE_VOICES
)

def get_random_voice():
    """Randomly select a voice from all available voices"""
    return random.choice(ALL_VOICES)

def process_single_file(input_path: str, output_path: str, voice: str = None):
    """Process a single file with the given input and output paths
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output audio file
        voice: Optional voice to use. If not provided, a random voice will be selected.
    """
    # Use 'a' for American English which is a valid language code for Kokoro TTS
    if voice is None:
        voice = get_random_voice()
    print(f"Using voice: {voice}")
    tts = TTSSDKokoro(lang_code='a', voice=voice)
    converter = BookConverter(tts)
    
    try:
        output_file = converter.process_book(input_path, output_path)
        print(f"Successfully created audio file: {output_file}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_batch():
    """Process all supported files in the books/to-do directory"""
    base_dir = Path("books")
    todo_dir = base_dir / "to-do"
    audio_dir = base_dir / "audio"
    done_dir = base_dir / "done"
    
    # Ensure all directories exist
    ensure_directory(todo_dir)
    ensure_directory(audio_dir)
    ensure_directory(done_dir)
    
    # Get all files in the todo directory
    all_files = list(todo_dir.glob('*'))
    
    if not all_files:
        print("No files found in books/to-do directory.")
        return
    
    print(f"Found {len(all_files)} file(s) in the to-do directory.")
    
    # Filter and process supported files
    processed_count = 0
    skipped_count = 0
    
    for file_path in all_files:
        if not file_path.is_file():
            print(f"\nSkipping non-file: {file_path.name}")
            skipped_count += 1
            continue
            
        ext = file_path.suffix.lower()
        file_ext = ext.lstrip('.')
        
        # Check if file extension is supported
        supported = False
        if file_ext in ['pdf', 'epub', 'txt', 'rtf']:
            supported = True
        elif file_ext in ['mobi', 'prc'] and MOBI_AVAILABLE:
            supported = True
        elif file_ext == 'lit' and CALIBRE_AVAILABLE:
            supported = True
            
        if not supported:
            reason = "Unsupported format"
            if file_ext in ['mobi', 'prc'] and not MOBI_AVAILABLE:
                reason = "MOBI/PRC support not available (install with: pip install mobi)"
            elif file_ext == 'lit' and not CALIBRE_AVAILABLE:
                reason = "LIT support requires Calibre (install from calibre-ebook.com)"
                
            print(f"\nSkipping {file_path.name}: {reason}")
            skipped_count += 1
            continue
            
        print(f"\nProcessing: {file_path.name}")
        
        # Create output path with same name but .wav extension in audio directory
        output_file = audio_dir / f"{file_path.stem}.wav"
        
        # Process the file with a random voice
        try:
            success = process_single_file(str(file_path), str(output_file))
            
            # Move the file to done directory if processing was successful
            if success:
                done_file = done_dir / file_path.name
                shutil.move(str(file_path), str(done_file))
                print(f"Moved {file_path.name} to done directory.")
                processed_count += 1
            else:
                print(f"Failed to process {file_path.name}")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            skipped_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print("="*50)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert books to audio')
    parser.add_argument('input_path', nargs='?', help='Path to the input book (PDF, EPUB, or TXT)')
    parser.add_argument('output_path', nargs='?', help='Path where the audio file will be saved')
    args = parser.parse_args()

    # If no arguments provided, run in batch mode
    if not args.input_path and not args.output_path:
        process_batch()
        return
    
    # If only one argument is provided, show help
    if not args.input_path or not args.output_path:
        parser.print_help()
        return

    # Process single file
    process_single_file(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
