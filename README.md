# PDF Chat Application with Hugging Face Models

A Flask-based web application that allows users to upload PDF files and chat with AI models about their content using Hugging Face transformers.

## Features

- Upload multiple PDF files
- Choose between TinyLlama and PHI-3 models
- Intelligent text chunking for large PDFs
- Context-aware responses based on PDF content
- Clean, dark-themed web interface

## Requirements

- Python 3.8+
- At least 4GB RAM (8GB recommended)
- 5GB free disk space for models

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Models will be downloaded automatically:**
   - Models are downloaded from Hugging Face on first use
   - Ensure you have internet connection and sufficient disk space
   - Models are cached locally after first download

## Available Models

The application uses these Hugging Face models:

### TinyLlama 1.1B Chat v1.0
- **Model ID:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size:** ~2.2GB
- **Features:** Fast inference, good for quick responses

### Microsoft Phi-3 Mini 4K Instruct
- **Model ID:** `microsoft/Phi-3-mini-4k-instruct`
- **Size:** ~7.6GB
- **Features:** Better quality responses, larger context window

## Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Use the application:**
   - Upload one or more PDF files
   - Select a model (TinyLlama for speed, PHI-3 for better quality)
   - Click "Start Chat"
   - Ask questions about your PDF content

## Model Configuration

The models are configured in `app.py`:

```python
MODELS = {
    'model1': {
        'name': 'TinyLlama 1.1B Chat v1.0',
        'model_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'context_window': 2048,
        'chunk_size': 1500,
        'max_tokens': 512
    },
    'model2': {
        'name': 'Microsoft Phi-3 Mini 4K Instruct',
        'model_id': 'microsoft/Phi-3-mini-4k-instruct',
        'context_window': 4096,
        'chunk_size': 2000,
        'max_tokens': 512
    }
}
```

## Troubleshooting

### Model Loading Issues
- Ensure you have internet connection for first-time downloads
- Check available disk space (models are large)
- Verify you have enough RAM (models load into memory)

### Performance Issues
- TinyLlama is faster but less capable
- PHI-3 is slower but provides better responses
- GPU acceleration will be used automatically if CUDA is available

### Memory Issues
- Close other applications to free up RAM
- Consider using CPU instead of GPU if memory is limited
- Reduce `context_window` in model configuration if needed

## File Structure

```
pdftutor/
├── app.py              # Main Flask application
├── home.html           # Web interface
├── requirements.txt    # Python dependencies
├── test_setup.py       # Setup verification script
├── README.md          # This file
└── uploads/           # Temporary PDF storage
```

## Technical Details

- **Backend:** Flask with Hugging Face Transformers
- **PDF Processing:** PyPDF2 for text extraction
- **Text Processing:** Intelligent chunking with overlap
- **Model Loading:** Lazy loading (models loaded on first use)
- **Context Retrieval:** Keyword-based chunk selection
- **Hardware:** Automatic GPU/CPU detection and usage

## License

This project is for educational purposes. Please respect the licenses of the individual models used.