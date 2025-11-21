import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random

print("=" * 70)
print("CONTEXTFLOW CHATBOT - ADVANCED SYSTEM")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

print("\nLoading chatbot model...")
custom_tokenizer = Tokenizer.load('models/tokenizer.pkl')

custom_model = HybridChatbotModel(
    vocab_size=len(custom_tokenizer.word2idx),
    embedding_dim=MODEL_CONFIG['embedding_dim'],
    hidden_dim=MODEL_CONFIG['hidden_dim'],
    num_layers=MODEL_CONFIG['num_layers'],
    num_heads=MODEL_CONFIG['num_heads'],
    dropout=MODEL_CONFIG['dropout']
)

checkpoint = torch.load('models/checkpoints/checkpoint_epoch_10.pt', map_location=device)
custom_model.load_state_dict(checkpoint['model_state_dict'])
custom_model.to(device)
custom_model.eval()
print("✅ Model loaded (Epoch 10)")

print("\nLoading language model...")
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.to(device)
gpt2_model.eval()
print("✅ GPT-2 ready")

print("\n" + "=" * 70)
print("System Ready - Full Contextual Awareness Enabled")
print("=" * 70 + "\n")

conversation_history = []

RESPONSE_PATTERNS = {
    r'\b(hello|hey)\b': ['Hello! How can I help you today?', 'Hi there! What can I do for you?'],
    r'\bhi\b': ['Hi! How are you?', 'Hello! Nice to talk to you!'],
    r'\bhow are you\b': ['I am doing well, thanks for asking!', 'Great! How about you?'],
    
    r'\bwhat is machine learning\b': ['Machine learning is a field of AI where computers learn from data without explicit programming'],
    r'\bmachine learning\b': ['Machine learning enables computers to learn patterns from data'],
    
    r'\bwhat is deep learning\b': ['Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns'],
    r'\bdeep learning\b': ['Deep learning uses deep neural networks to automatically learn hierarchical representations from data'],
    r'\bwhat is dl\b': ['DL (Deep Learning) is an advanced form of machine learning using multi-layered neural networks'],
    
    r'\bwhat is neural network\b': ['A neural network is a computing system inspired by biological neural networks'],
    r'\bneural network\b': ['Neural networks process information using layers of interconnected nodes'],
    
    r'\bwhat is python\b': ['Python is a versatile programming language widely used in data science and web development'],
    r'\bpython\b': ['Python is great for beginners and powerful for experts'],
    
    r'\bwhat is programming\b': ['Programming is the process of creating instructions for computers to follow'],
    r'\bprogramming\b': ['Programming is both an art and a science of problem-solving'],
    
    r'\bwhat is ai\b': ['AI enables machines to perform tasks that typically require human intelligence'],
    r'\bartificial intelligence\b': ['AI includes learning, reasoning, and problem-solving capabilities'],
    
    r'\bwhat is cnn\b': ['CNN (Convolutional Neural Network) is primarily used for image processing and computer vision'],
    r'\bwhat is rnn\b': ['RNN (Recurrent Neural Network) is designed for sequential data like text and time series'],
    r'\bwhat is lstm\b': ['LSTM (Long Short-Term Memory) is a type of RNN that can remember long-term dependencies'],
    
    r'\bhelp me\b': ['I am here to help! What do you need assistance with?'],
    r'\bcan you help\b': ['Of course! Ask me anything!'],
    
    r'\bthank(s| you)\b': ['You are very welcome!', 'Happy to help!', 'Anytime!'],
    r'\b(bye|goodbye)\b': ['Goodbye! Have a great day!', 'See you later!'],
}

def check_patterns(user_input):
    """Check if input matches predefined patterns"""
    user_lower = user_input.lower().strip()
    for pattern, responses in RESPONSE_PATTERNS.items():
        if re.search(pattern, user_lower):
            return random.choice(responses)
    return None

def check_context_recall(user_input):
    """Check if user is asking about previous conversation"""
    user_lower = user_input.lower().strip()
    
    recall_patterns = [
        r'what (did|have) i (ask|asked|say|said|tell|told).*(before|earlier|previously|prior)',
        r'what was (my|the) (last|previous|earlier) (question|query|topic)',
        r'(what|which) (topic|question|subject).*(earlier|before|previous)',
        r'about what (did|have) i (ask|asked|say)',
        r'what (were|was) we (talking|discussing|speaking) about'
    ]
    
    for pattern in recall_patterns:
        if re.search(pattern, user_lower):
            return True
    return False

def check_follow_up(user_input):
    """Check if this is a follow-up question requiring context"""
    user_lower = user_input.lower().strip()
    
    follow_up_patterns = [
        r'^(can you |could you )?(explain|tell|give|show).*(more|example|detail|further)',
        r'^(how|why|when|where|what).*(it|that|this|them)',
        r'^(is|are|does|do|can|could|would).*(it|that|this)',
        r'^tell me more',
        r'^explain (that|this|it)',
        r'^give me (an )?example',
        r'^how (does|do|is|are) (it|that|this|they)',
        r'^what about',
        r'^and what about',
        r'^can you elaborate',
        r'^could you clarify'
    ]
    
    for pattern in follow_up_patterns:
        if re.search(pattern, user_lower):
            return True
    return False

def get_conversation_summary():
    """Analyze conversation history and return topics discussed"""
    global conversation_history
    
    if len(conversation_history) == 0:
        return "We haven't discussed anything yet."
    
    topics = []
    
    for turn in conversation_history:
        user_msg = turn['user'].lower()
        
        if 'deep learning' in user_msg or 'dl' in user_msg:
            topics.append('deep learning')
        elif 'machine learning' in user_msg or 'ml' in user_msg:
            topics.append('machine learning')
        elif 'neural network' in user_msg or 'nn' in user_msg:
            topics.append('neural networks')
        elif 'python' in user_msg:
            topics.append('Python programming')
        elif 'ai' in user_msg or 'artificial intelligence' in user_msg:
            topics.append('artificial intelligence')
        elif 'programming' in user_msg or 'coding' in user_msg:
            topics.append('programming')
        elif 'cnn' in user_msg or 'convolutional' in user_msg:
            topics.append('CNN')
        elif 'rnn' in user_msg or 'recurrent' in user_msg:
            topics.append('RNN')
        elif 'lstm' in user_msg:
            topics.append('LSTM')
    
    if not topics:
        return "We were having a general conversation."
    
    unique_topics = list(set(topics))
    
    if len(unique_topics) == 1:
        return f"You asked about {unique_topics[0]}."
    elif len(unique_topics) == 2:
        return f"You asked about {unique_topics[0]} and {unique_topics[1]}."
    else:
        topics_str = ', '.join(unique_topics[:-1])
        return f"You asked about {topics_str}, and {unique_topics[-1]}."

def get_relevant_context(user_input):
    """Get relevant context based on current query"""
    global conversation_history
    
    if len(conversation_history) == 0:
        return None
    
    last_turn = conversation_history[-1]
    return f"Previously, you asked: '{last_turn['user']}' and I said: '{last_turn['bot']}'"

def build_contextual_prompt(user_input, use_full_context=True):
    """Build prompt with full conversation context"""
    global conversation_history
    
    if len(conversation_history) == 0:
        return user_input
    
    if not use_full_context:
        return user_input
    
    context_parts = []
    
    for turn in conversation_history[-4:]:
        context_parts.append(f"User: {turn['user']}")
        context_parts.append(f"Assistant: {turn['bot']}")
    
    context_parts.append(f"User: {user_input}")
    context_parts.append("Assistant:")
    
    full_prompt = "\n".join(context_parts)
    
    return full_prompt

def generate_with_context(user_input, max_length=60):
    """Generate response with full contextual awareness"""
    global conversation_history
    
    is_follow_up = check_follow_up(user_input)
    
    prompt = build_contextual_prompt(user_input, use_full_context=True)
    
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
    
    if '\n' in response:
        response = response.split('\n')[0].strip()
    
    if 'User:' in response:
        response = response.split('User:')[0].strip()
    if 'Assistant:' in response:
        response = response.split('Assistant:')[0].strip()
    
    sentences = []
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

def get_response(user_input):
    """Get response with full contextual awareness"""
    global conversation_history
    
    if check_context_recall(user_input):
        summary = get_conversation_summary()
        conversation_history.append({
            'user': user_input,
            'bot': summary
        })
        
        if len(conversation_history) > 15:
            conversation_history = conversation_history[-15:]
        
        return summary
    
    pattern_response = check_patterns(user_input)
    if pattern_response:
        conversation_history.append({
            'user': user_input,
            'bot': pattern_response
        })
        
        if len(conversation_history) > 15:
            conversation_history = conversation_history[-15:]
        
        return pattern_response
    
    try:
        response = generate_with_context(user_input, max_length=50)
        
        if response and len(response.split()) >= 3:
            conversation_history.append({
                'user': user_input,
                'bot': response
            })
            
            if len(conversation_history) > 15:
                conversation_history = conversation_history[-15:]
            
            return response
        else:
            return "I'm processing that. Could you rephrase?"
    except Exception as e:
        return "I encountered an issue. Please try again."

def clear_context():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    print("🔄 Context cleared\n")

print("Type 'quit' to exit | 'clear' to reset context\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == 'quit':
        print("\nThank you for chatting! Goodbye!\n")
        break
    
    if user_input.lower() == 'clear':
        clear_context()
        continue
    
    if not user_input:
        continue
    
    response = get_response(user_input)
    print(f"Bot: {response}\n")
