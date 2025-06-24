# Ebook to Audiobook Converter

A Python script that converts ebooks to audiobooks using text-to-speech. Supports multiple input formats and batch processing.

## Features

- 🎧 Convert ebooks to high-quality audio files
- 📚 Supports multiple formats: PDF, EPUB, TXT, RTF, MOBI, PRC, and LIT
- 🔄 Batch processing of multiple files
- 🎭 Multiple AI voices with random selection
- 📂 Automatic file organization
- 🚀 Fast and efficient processing

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- Calibre (optional, for LIT support)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gillbillieee/audiobook_maker.git
   cd audiobook_maker
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For MOBI/PRC support:
   ```bash
   pip install mobi
   ```

4. (Optional) For LIT support, install Calibre from [calibre-ebook.com](https://calibre-ebook.com/)

## Usage

### Directory Structure

```
book_maker/
├── books/
│   ├── to-do/      # Place your input files here
│   ├── audio/       # Output audio files will be saved here
│   └── done/        # Processed files will be moved here
├── book_converter.py # Main script
└── requirements.txt  # Python dependencies
└── tts_sdkokoro.py # Text to speech functions
```

### Basic Usage

1. Place your ebook files in the `books/to-do` directory
2. Run the script:
   ```bash
   python book_converter.py
   ```
3. Processed audio files will be saved in `books/audio`
4. Original files will be moved to `books/done`

### Command Line Arguments

Process a single file:
```bash
python book_converter.py input.epub output.wav
```

## Supported Formats

| Format | Support | Notes |
|--------|---------|-------|
| PDF    | ✅      | Full support |
| EPUB   | ✅      | Full support |
| TXT    | ✅      | Full support |
| RTF    | ✅      | Full support |
| MOBI   | ✅      | Requires `mobi` package |
| PRC    | ✅      | Requires `mobi` package |
| LIT    | ⚠️      | Requires Calibre installation |

## Voice Options

The script will randomly select from these voices for each file by default. You can also specify a voice using the `--voice` parameter.

### American Female Voices
- af_alloy
- af_aoede
- af_bella
- af_heart
- af_jessica
- af_kore
- af_nicole
- af_nova
- af_river
- af_sarah
- af_sky

### British Female Voices
- bf_alice
- bf_emma
- bf_isabella
- bf_lily

### American Male Voices
- am_adam
- am_echo
- am_eric
- am_fenrir
- am_liam
- am_michael
- am_onyx
- am_puck

### British Male Voices
- bm_daniel
- bm_fable
- bm_george
- bm_lewis

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - For MOBI/PRC: `pip install mobi`
   - For LIT: Install Calibre from [calibre-ebook.com](https://calibre-ebook.com/)
   - If you just installed Calibre, restart your computer or sign out and back in to ensure file associations are properly registered

2. **File Not Found**
   - Ensure files are in the `books/to-do` directory
   - Check file extensions are correct

3. **Audio Quality Issues**
   - Try converting to a different format
   - Check input file quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- Uses Kokoro TTS for text-to-speech
- Built with Python's awesome ecosystem
- Inspired by the need for accessible reading options
