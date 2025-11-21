from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import os
import sqlite3
from datetime import datetime
import sys
import re
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
from backend.inference import AdvancedInference
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
custom_model = None
custom_tokenizer = None
custom_inference = None
gpt2_model = None
gpt2_tokenizer = None

DB_PATH = 'backend/conversations.db'

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
    r'\b(bye|goodbye)\\b': [
        'Goodbye! Have a great day!',
        'See you later! Feel free to come back anytime.'
    ],
}

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
            inference_method TEXT DEFAULT 'hybrid',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(session_id, user_input, bot_response, method='hybrid'):
    """Save conversation to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (session_id, user_input, bot_response, inference_method)
        VALUES (?, ?, ?, ?)
    ''', (session_id, user_input, bot_response, method))
    conn.commit()
    conn.close()

def load_models():
    """Load custom model and GPT-2"""
    global custom_model, custom_tokenizer, custom_inference, gpt2_model, gpt2_tokenizer
    
    print("Loading custom model and tokenizer...")
    
    # Load custom tokenizer
    custom_tokenizer = Tokenizer.load('models/tokenizer.pkl')
    
    # Load custom model
    custom_model = HybridChatbotModel(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    checkpoint_path = 'models/checkpoints/checkpoint_epoch_10.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    custom_model.load_state_dict(checkpoint['model_state_dict'])
    custom_model.to(device)
    custom_model.eval()
    print("Custom model loaded")
    
    # Initialize custom inference
    custom_inference = AdvancedInference(custom_model, custom_tokenizer, device)
    print("Custom inference initialized")
    
    # Load GPT-2
    print("Loading GPT-2 for fallback...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.to(device)
    gpt2_model.eval()
    print("GPT-2 ready")

def check_patterns(user_input):
    """Check if input matches predefined patterns"""
    user_lower = user_input.lower().strip()
    user_clean = re.sub(r'[^\w\s]', ' ', user_lower).strip()
    
    for pattern, responses in RESPONSE_PATTERNS.items():
        if re.search(pattern, user_lower, re.IGNORECASE) or re.search(pattern, user_clean, re.IGNORECASE):
            return random.choice(responses)
    return None

def check_context_recall(user_input, context_history):
    """Check if user is asking about previous conversation"""
    user_lower = user_input.lower().strip()
    
    recall_patterns = [
        r'what (did|have) i (ask|say|tell|mention)',
        r'what was (my|the) (last|previous|earlier) (question|message)',
        r'remind me what (i|we) (talked|discussed|said)',
        r'what were we (talking|discussing) about'
    ]
    
    for pattern in recall_patterns:
        if re.search(pattern, user_lower):
            if context_history and len(context_history) > 0:
                last_user_msg = context_history[-1]['user']
                return f"You asked: '{last_user_msg}'"
            else:
                return "This is the start of our conversation, so there's no previous message."
    
    return None

def is_response_valid(response):
    """Check if response is valid (not empty, not just special tokens)"""
    if not response or len(response.strip()) == 0:
        return False
    if len(response.strip()) < 3:
        return False
    
    words = response.split()
    
    # Check if it's mostly repetitive
    if len(words) > 0 and len(set(words)) / len(words) < 0.3:
        return False
    
    # Check if it's just random short words
    if len(words) <= 2 and all(len(w) <= 3 for w in words):
        return False
    
    # Check for common gibberish patterns
    gibberish_indicators = [
        'and then but',
        'the a an',
        'is are was',
        'in on at of',
        'i you he she it we they'
    ]
    
    response_lower = response.lower()
    for indicator in gibberish_indicators:
        # If response is just a sequence of common words without meaning
        if indicator in response_lower and len(words) < 8:
            return False
    
    # Check if response has too many consecutive short words (likely gibberish)
    short_word_count = sum(1 for w in words if len(w) <= 3)
    if len(words) > 0 and short_word_count / len(words) > 0.7:
        return False
    
    # Check for minimum word length average
    if len(words) > 0:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 3:
            return False
    
    return True

def generate_with_custom_model(user_input, context_history=None):
    """Generate response using custom model"""
    global custom_inference
    
    try:
        # Build context string
        context_parts = []
        if context_history:
            for turn in context_history[-2:]:  # Last 2 turns
                context_parts.append(f"User: {turn['user']}")
                context_parts.append(f"Bot: {turn['bot']}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        full_input = f"{context_str}\nUser: {user_input}".strip() if context_str else user_input
        
        # Use temperature sampling for more variety
        response = custom_inference.temperature_sampling(
            full_input,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            max_length=30
        )
        
        return response
    except Exception as e:
        logging.error(f"Custom model error: {str(e)}")
        return None

def generate_with_gpt2(user_input, context_history=None, max_length=80):
    """Generate response using GPT-2 with context"""
    global gpt2_model, gpt2_tokenizer
    
    context_parts = [
        "You are a helpful AI assistant. Follow these rules strictly:",
        "1. Answer based ONLY on the conversation history shown below",
        "2. If asked about previous messages, refer to the actual conversation history",
        "3. Stay on topic and be concise",
        "4. Do not make up information that wasn't discussed",
        "5. Keep responses under 2-3 sentences",
        ""
    ]
    
    if context_history and len(context_history) > 0:
        context_parts.append("=== Conversation History ===")
        for turn in context_history[-3:]:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['bot']}")
        context_parts.append("=== End History ===")
        context_parts.append("")
    
    context_parts.append(f"User: {user_input}")
    context_parts.append("Assistant:")
    
    prompt = "\n".join(context_parts)
    
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
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3
        )
    
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Clean up response
    if '\n' in response:
        response = response.split('\n')[0].strip()
    
    if 'User:' in response:
        response = response.split('User:')[0].strip()
    if 'Assistant:' in response:
        response = response.split('Assistant:')[0].strip()
    if '===' in response:
        response = response.split('===')[0].strip()
    
    # Get first complete sentence or two
    sentences = []
    for delimiter in ['. ', '? ', '! ']:
        if delimiter in response:
            parts = response.split(delimiter)
            if len(parts) >= 2:
                sentences = [parts[0] + delimiter.strip(), parts[1] + delimiter.strip() if len(parts) > 2 else parts[1]]
                response = ''.join(sentences[:2])
                break
    
    if not sentences:
        if '.' in response:
            response = response.split('.')[0] + '.'
        elif '?' in response:
            response = response.split('?')[0] + '?'
        elif '!' in response:
            response = response.split('!')[0] + '!'
    
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
        'custom_model_loaded': custom_model is not None,
        'gpt2_loaded': gpt2_model is not None
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """Hybrid chat endpoint using custom model + pattern matching + GPT-2"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            logging.error("Error: Missing message field")
            return jsonify({'error': 'Missing "message" field'}), 400
        
        logging.info(f"Received message: {data['message']}")
        
        user_input = data['message'].strip()
        session_id = data.get('session_id', 'default')
        params = data.get('params', {})
        
        if not user_input or len(user_input) > 500:
            return jsonify({'error': 'Invalid message length'}), 400
        
        # Get conversation history from database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT user_input, bot_response FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 3
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        
        context_history = [{'user': row[0], 'bot': row[1]} for row in reversed(rows)]
        
        response = None
        actual_method = 'hybrid'
        
        # TIER 1: Check for context recall questions
        context_recall = check_context_recall(user_input, context_history)
        if context_recall:
            response = context_recall
            actual_method = 'context_recall'
            logging.info(f"Context recall response: '{response}'")
        
        # TIER 2: Check for pattern-based responses
        elif check_patterns(user_input):
            response = check_patterns(user_input)
            actual_method = 'pattern_match'
            logging.info(f"Pattern match response: '{response}'")
        
        # TIER 3: Try custom model
        else:
            custom_response = generate_with_custom_model(user_input, context_history)
            
            if custom_response and is_response_valid(custom_response):
                response = custom_response
                actual_method = 'custom_model'
                logging.info(f"Custom model response: '{response}'")
            else:
                # TIER 4: Fallback to GPT-2
                logging.info(f"Custom model failed (response: '{custom_response}'), using GPT-2")
                response = generate_with_gpt2(user_input, context_history, params.get('max_length', 80))
                actual_method = 'gpt2_fallback'
                logging.info(f"GPT-2 response: '{response}'")
        
        # Save to database
        save_conversation(session_id, user_input, response, actual_method)
        
        return jsonify({
            'user_message': user_input,
            'bot_response': response,
            'session_id': session_id,
            'inference_method': actual_method,
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
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        
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
        print("\nContextFlow Hybrid Backend Starting...")
        print("This version uses: Custom Model + Pattern Matching + GPT-2 Fallback")
        print(f"Server will run on http://localhost:5000\n")
        
        init_db()
        load_models()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Error initializing backend: {str(e)}")
        print(f"\nError initializing backend: {str(e)}")
        exit(1)
