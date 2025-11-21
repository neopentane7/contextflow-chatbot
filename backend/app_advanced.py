from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import json
import os
import sqlite3
from datetime import datetime
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
from backend.inference import AdvancedInference
from backend.context_manager import ContextManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

app = Flask(__name__)
CORS(app)

# ===== GLOBALS =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
tokenizer = None
inference = None
context_manager = None
gpt2_model = None
gpt2_tokenizer = None

DB_PATH = 'backend/conversations.db'

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_input TEXT,
            bot_response TEXT,
            inference_method TEXT DEFAULT 'gpt2_fallback',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(session_id, user_input, bot_response, method='gpt2_fallback'):
    """Save conversation to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (session_id, user_input, bot_response, inference_method)
        VALUES (?, ?, ?, ?)
    ''', (session_id, user_input, bot_response, method))
    conn.commit()
    conn.close()

def load_model():
    """Load trained model, tokenizer, and GPT-2"""
    global model, tokenizer, inference, context_manager, gpt2_model, gpt2_tokenizer
    
    print("Loading model and tokenizer...")
    
    # Load tokenizer
    if not os.path.exists('models/tokenizer.pkl'):
        raise FileNotFoundError("Tokenizer not found.")
    
    tokenizer = Tokenizer.load('models/tokenizer.pkl')
    
    # Load model
    model = HybridChatbotModel(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    checkpoint_path = 'models/checkpoints/checkpoint_epoch_10.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # Initialize inference and context manager
    inference = AdvancedInference(model, tokenizer, device)
    context_manager = ContextManager(DB_PATH)
    
    print("Advanced inference initialized")
    print("Context manager initialized")
    
    # Load GPT-2
    print("Loading GPT-2 for fallback...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.to(device)
    gpt2_model.eval()
    print("GPT-2 ready")

def is_response_valid(response):
    """Check if response is valid (not empty, not just special tokens)"""
    if not response or len(response.strip()) == 0:
        return False
    if len(response.strip()) < 3:  # Too short
        return False
    # Check if it's mostly repetitive
    words = response.split()
    if len(words) > 0 and len(set(words)) / len(words) < 0.3:  # More than 70% repetition
        return False
    return True

# Pattern-based responses for common questions
RESPONSE_PATTERNS = {
    r'\b(hello|hey|hi)\b': [
        'Hello! How can I help you today?',
        'Hi there! What can I do for you?',
        'Hey! Nice to talk to you!'
    ],
    r'\bhow are you\b': [
        'I am doing well, thanks for asking! How can I assist you?',
        'Great! How about you? What brings you here today?'
    ],
    r'\bwhat is (deep learning|dl)\b': [
        'Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from data. It powers applications like image recognition, natural language processing, and autonomous vehicles.'
    ],
    r'\bwhat is (machine learning|ml)\b': [
        'Machine learning is a field of AI where computers learn from data without being explicitly programmed. It enables systems to improve their performance on tasks through experience.'
    ],
    r'\bwhat is (artificial intelligence|ai)\b': [
        'Artificial Intelligence (AI) enables machines to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and understanding language.'
    ],
    r'\bwhat is (neural network|nn)\b': [
        'A neural network is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) organized in layers that process information to learn patterns from data.'
    ],
    r'\bwhat is python\b': [
        'Python is a versatile, high-level programming language widely used in data science, web development, automation, and AI. It is known for its simple syntax and powerful libraries.'
    ],
    r'\bwhat is (cnn|convolutional neural network)\b': [
        'CNN (Convolutional Neural Network) is a type of neural network primarily used for image processing and computer vision tasks. It uses convolutional layers to automatically learn spatial hierarchies of features.'
    ],
    r'\bwhat is (rnn|recurrent neural network)\b': [
        'RNN (Recurrent Neural Network) is designed for sequential data like text and time series. It has connections that form cycles, allowing it to maintain memory of previous inputs.'
    ],
    r'\bwhat is (lstm|long short.term memory)\b': [
        'LSTM (Long Short-Term Memory) is a type of RNN that can learn long-term dependencies. It uses special gates to control information flow, making it effective for tasks like language modeling and speech recognition.'
    ],
    r'\b(thank you|thanks)\b': [
        'You are very welcome!',
        'Happy to help!',
        'Anytime!'
    ],
    r'\b(bye|goodbye)\b': [
        'Goodbye! Have a great day!',
        'See you later! Feel free to come back anytime.'
    ],
}

def check_patterns(user_input):
    """Check if input matches predefined patterns"""
    import random
    user_lower = user_input.lower().strip()
    # Remove punctuation for better matching
    user_clean = re.sub(r'[^\w\s]', ' ', user_lower).strip()
    
    for pattern, responses in RESPONSE_PATTERNS.items():
        if re.search(pattern, user_lower, re.IGNORECASE) or re.search(pattern, user_clean, re.IGNORECASE):
            return random.choice(responses)
    return None

def generate_with_gpt2(user_input, context_history=None, max_length=80):
    """Generate response using GPT-2 with context"""
    global gpt2_model, gpt2_tokenizer
    
    # Build prompt with context and system instruction
    context_parts = [
        "The following is a conversation with an AI assistant. The assistant is helpful, knowledgeable, and provides clear, concise answers.",
        ""
    ]
    
    if context_history:
        for turn in context_history[-3:]:  # Last 3 turns
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['bot']}")
    
    context_parts.append(f"User: {user_input}")
    context_parts.append("Assistant:")
    
    prompt = "\\n".join(context_parts)
    
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    if inputs.shape[1] > 800:
        inputs = inputs[:, -800:]
    
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=inputs.shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.75,
            top_p=0.92,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Clean up response
    if '\\n' in response:
        response = response.split('\\n')[0].strip()
    
    if 'User:' in response:
        response = response.split('User:')[0].strip()
    if 'Assistant:' in response:
        response = response.split('Assistant:')[0].strip()
    
    # Get first complete sentence
    if '.' in response:
        sentences = response.split('.')
        response = sentences[0] + '.' if sentences[0] else response
    elif '?' in response:
        sentences = response.split('?')
        response = sentences[0] + '?'
    elif '!' in response:
        sentences = response.split('!')
        response = sentences[0] + '!'
    
    return response

# ===== API ENDPOINTS =====

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model is not None,
        'gpt2_loaded': gpt2_model is not None
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Advanced chat endpoint with GPT-2 fallback
    
    Request JSON:
    {
        "message": "your message",
        "session_id": "user_123",
        "method": "auto",
        "params": {
            "temperature": 0.8,
            "max_length": 50
        },
        "memory_type": "full"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            logging.error("Error: Missing message field")
            return jsonify({'error': 'Missing "message" field'}), 400
        
        logging.info(f"Received message: {data['message']}")
        
        user_input = data['message'].strip()
        session_id = data.get('session_id', 'default')
        method = data.get('method', 'auto')  # auto, custom, gpt2
        memory_type = data.get('memory_type', 'full')
        params = data.get('params', {})
        
        if not user_input or len(user_input) > 500:
            return jsonify({'error': 'Invalid message length'}), 400
        
        # Get or create context
        context = context_manager.get_or_create_context(
            session_id,
            max_turns=10,
            memory_type=memory_type
        )
        
        # Get conversation history for GPT-2
        context_history = []
        if hasattr(context, 'conversation_history'):
            context_history = context.conversation_history
        
        response = None
        actual_method = method
        
        # First, check for pattern-based responses (highest quality)
        pattern_response = check_patterns(user_input)
        # Save to database
        save_conversation(session_id, user_input, response, actual_method)
        
        return jsonify({
            'user_message': user_input,
            'bot_response': response,
            'session_id': session_id,
            'inference_method': actual_method,
            'memory_type': memory_type,
            'context_stats': context.get_context_stats(),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logging.error(f"Error in /api/chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_context():
    """Clear conversation context for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        context = context_manager.get_or_create_context(session_id)
        context.clear_context()
        
        return jsonify({
            'status': 'success',
            'message': 'Context cleared',
            'session_id': session_id
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== MAIN =====

if __name__ == '__main__':
    try:
        print("\\nBackend initialized successfully!")
        print(f"Starting server on http://localhost:5000\\n")
        
        init_db()
        load_model()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Error initializing backend: {str(e)}")
        print(f"\\nError initializing backend: {str(e)}")
        exit(1)
