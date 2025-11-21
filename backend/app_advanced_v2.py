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
    r'\b(bye|goodbye)\b': [
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
            inference_method TEXT DEFAULT 'gpt2',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(session_id, user_input, bot_response, method='gpt2'):
    """Save conversation to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (session_id, user_input, bot_response, inference_method)
        VALUES (?, ?, ?, ?)
    ''', (session_id, user_input, bot_response, method))
    conn.commit()
    conn.close()

def load_gpt2():
    """Load GPT-2 model"""
    global gpt2_model, gpt2_tokenizer
    
    print("Loading GPT-2...")
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
    
    # Patterns for context recall
    recall_patterns = [
        r'what (did|have) i (ask|say|tell|mention)',
        r'what was (my|the) (last|previous|earlier) (question|message)',
        r'remind me what (i|we) (talked|discussed|said)',
        r'what were we (talking|discussing) about'
    ]
    
    for pattern in recall_patterns:
        if re.search(pattern, user_lower):
            if context_history and len(context_history) > 0:
                # Get the last user message
                last_user_msg = context_history[-1]['user']
                return f"You asked: '{last_user_msg}'"
            else:
                return "This is the start of our conversation, so there's no previous message."
    
    return None

def generate_with_gpt2(user_input, context_history=None, max_length=80):
    """Generate response using GPT-2 with context"""
    global gpt2_model, gpt2_tokenizer
    
    # Build prompt with enhanced system instruction
    context_parts = [
        "You are a helpful AI assistant. Follow these rules strictly:",
        "1. Answer based ONLY on the conversation history shown below",
        "2. If asked about previous messages, refer to the actual conversation history",
        "3. Stay on topic and be concise",
        "4. Do not make up information that wasn't discussed",
        "5. Keep responses under 2-3 sentences",
        ""
    ]
    
    # Add conversation history if available
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
            temperature=0.7,  # Lower temperature for more focused responses
            top_p=0.9,  # Slightly more conservative
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3  # Higher penalty to avoid repetition
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
            # Take first 1-2 sentences
            if len(parts) >= 2:
                sentences = [parts[0] + delimiter.strip(), parts[1] + delimiter.strip() if len(parts) > 2 else parts[1]]
                response = ''.join(sentences[:2])
                break
    
    # Fallback: just take first sentence
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
        'gpt2_loaded': gpt2_model is not None
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint using pattern matching + GPT-2"""
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
        actual_method = 'gpt2'
        
        # First, check for context recall questions
        context_recall = check_context_recall(user_input, context_history)
        if context_recall:
            response = context_recall
            actual_method = 'context_recall'
            logging.info(f"Context recall response: '{response}'")
        # Second, check for pattern-based responses
        elif check_patterns(user_input):
            response = check_patterns(user_input)
            actual_method = 'pattern_match'
            logging.info(f"Pattern match response: '{response}'")
        else:
            # Use GPT-2 for all other responses
            response = generate_with_gpt2(user_input, context_history, params.get('max_length', 80))
            actual_method = 'gpt2'
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
        print("\nContextFlow Backend Starting...")
        print(f"Server will run on http://localhost:5000\n")
        
        init_db()
        load_gpt2()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Error initializing backend: {str(e)}")
        print(f"\nError initializing backend: {str(e)}")
        exit(1)
