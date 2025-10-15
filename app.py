from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import PyPDF2
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__, static_folder='.')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store PDF content in memory (simple session-like storage)
pdf_content_store = {}

# Store loaded models in memory
loaded_models = {}

# Model configurations
MODELS = {
    'model1': {
        'name': 'TinyLlama 1.1B Chat v1.0',
        'model_id': 'jykh01/TinyLlama-1.1B-Chat-v1.0-local',
        'context_window': 2048,
        'chunk_size': 1500,
        'max_tokens': 512
    },
    'model2': {
        'name': 'Microsoft Phi-3 Mini 4K Instruct',
        'model_id': 'jykh01/Phi-3-mini-4k-instruct-local',
        'context_window': 4096,
        'chunk_size': 2000,
        'max_tokens': 512
    }
}

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configure requests session with retry strategy for better connectivity
def setup_requests_session():
    """Setup requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Setup the session
requests_session = setup_requests_session()

def load_model(model_key):
    """Load a Hugging Face model if not already loaded"""
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    model_config = MODELS[model_key]
    model_id = model_config['model_id']
    
    print(f"Loading model: {model_config['name']} ({model_id})")
    print("This may take a few minutes on first run as the model downloads...")
    
    try:
        # Set environment variables for Hugging Face model usage
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'models')
        os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models')
        
        # Ensure models directory exists for caching
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Try multiple approaches for loading the model
        pipe = None
        
        # Approach 1: Standard pipeline with retry
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}/3: Loading model...")
                pipe = pipeline(
                    "text-generation",
                    model=model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    use_fast=False,  # Use slower but more reliable tokenizer
                    revision="main",  # Explicitly specify revision
                    cache_dir=os.path.join(os.getcwd(), 'models'),  # Cache downloaded models locally
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    raise e
                print("Retrying in 5 seconds...")
                import time
                time.sleep(5)
        
        if pipe is None:
            raise Exception("Failed to load model after all attempts")
        
        loaded_models[model_key] = pipe
        print(f"Successfully loaded model: {model_config['name']}")
        return pipe
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error loading model {model_config['name']}: {error_msg}")
        
        # Provide helpful error messages
        if "couldn't connect" in error_msg.lower():
            print("Connection issue detected. Suggestions:")
            print("1. Check your internet connection")
            print("2. Try again in a few minutes (Hugging Face might be temporarily down)")
            print("3. Consider using a VPN if you're behind a firewall")
        
        raise Exception(f"Model loading failed: {error_msg}")

def allowed_file(filename):
    """Check if file is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def clean_text(text):
    """Clean and normalize extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def find_relevant_chunks(chunks, query, top_k=3):
    """Find most relevant chunks based on keyword matching"""
    query_words = set(query.lower().split())
    chunk_scores = []
    
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Simple scoring: count matching words
        score = sum(1 for word in query_words if word in chunk_lower)
        chunk_scores.append((idx, score, chunk))
    
    # Sort by score and return top_k chunks
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for _, score, chunk in chunk_scores[:top_k] if score > 0]
    
    return relevant_chunks

def query_local_model(model_key, query, context=""):
    """Query local Hugging Face model with the given prompt"""
    try:
        # Load the model if not already loaded
        pipe = load_model(model_key)
        model_config = MODELS[model_key]
        
        # Construct the full prompt with context
        full_prompt = f"""You are a helpful assistant that answers questions based on the provided PDF content.

Context from PDF:
{context}

User Question: {query}

Instructions: Answer the question based ONLY on the context provided above. If the answer is not in the context, say "I cannot find that information in the provided PDF." Be concise and specific.

Answer:"""

        # Generate response using the pipeline
        response = pipe(
            full_prompt,
            max_new_tokens=model_config['max_tokens'],
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text'].strip()
        
        return generated_text if generated_text else "No response generated from model"
    
    except Exception as e:
        return f"Error querying model: {str(e)}"

@app.route('/')
def index():
    """Serve the home page"""
    # Serve home.html from the same directory as app.py
    return app.send_static_file('home.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle PDF upload and processing"""
    try:
        if 'pdfs' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('pdfs')
        model = request.form.get('model')
        
        if not model or model not in MODELS:
            return jsonify({'error': 'Invalid model selection'}), 400
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Process all PDFs
        all_text = ""
        processed_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Extract text from PDF
                text = extract_text_from_pdf(filepath)
                
                if text:
                    all_text += text + "\n\n"
                    processed_files.append(filename)
                
                # Clean up uploaded file (optional - comment out if you want to keep files)
                # os.remove(filepath)
        
        if not all_text:
            return jsonify({'error': 'Could not extract text from PDFs'}), 400
        
        # Clean and chunk the text
        cleaned_text = clean_text(all_text)
        chunk_size = MODELS[model]['chunk_size']
        chunks = chunk_text(cleaned_text, chunk_size=chunk_size)
        
        # Store chunks with a simple session ID (use the first filename as ID)
        session_id = secure_filename(processed_files[0])
        pdf_content_store[session_id] = {
            'chunks': chunks,
            'model': model,
            'files': processed_files
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_processed': processed_files,
            'chunks_created': len(chunks),
            'model': model
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        if session_id not in pdf_content_store:
            return jsonify({'error': 'Session not found. Please upload PDFs again.'}), 400
        
        # Get stored data
        session_data = pdf_content_store[session_id]
        chunks = session_data['chunks']
        model_key = session_data['model']
        
        # Find relevant chunks based on the query
        relevant_chunks = find_relevant_chunks(chunks, message, top_k=3)
        
        if not relevant_chunks:
            return jsonify({
                'response': "I couldn't find relevant information in the PDF to answer your question."
            })
        
        # Combine relevant chunks as context
        context = "\n\n".join(relevant_chunks)
        
        # Query the local model
        response_text = query_local_model(model_key, message, context)
        
        return jsonify({
            'response': response_text,
            'chunks_used': len(relevant_chunks)
        })
    
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500



if __name__ == '__main__':
    print("=" * 60)
    print("PDF Chat Application with Hugging Face Models")
    print("=" * 60)
    print(f"Device: {device}")
    print("\nAvailable models:")
    for model_key, config in MODELS.items():
        print(f"   - {config['name']} ({config['model_id']})")
    print("\nIMPORTANT: Before running this app, make sure you have:")
    print("1. Required Python packages installed:")
    print("   - pip install -r requirements.txt")
    print("2. Internet connection for first-time model downloads")
    print("3. Sufficient disk space (models are ~1-3GB each)")
    print("\nModels will be downloaded automatically on first use.")
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)