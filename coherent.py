import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
import torch.nn.functional as F
import random
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer.load('models/tokenizer.pkl')

model = HybridChatbotModel(
    vocab_size=len(tokenizer.word2idx),
    embedding_dim=MODEL_CONFIG['embedding_dim'],
    hidden_dim=MODEL_CONFIG['hidden_dim'],
    num_layers=MODEL_CONFIG['num_layers'],
    num_heads=MODEL_CONFIG['num_heads'],
    dropout=MODEL_CONFIG['dropout']
)

checkpoint = torch.load('models/checkpoints/checkpoint_epoch_10.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Templates - using EXACT patterns
TEMPLATES = {
    # Greetings
    r'\b(hello|hey)\b': ['Hello! How can I help?', 'Hi there!', 'Hey! What can I do for you?'],
    r'\bhi\b': ['Hi! How are you?', 'Hello!', 'Hey there!'],
    r'\bhow are you\b': ['I am doing well, thanks!', 'I am fine. How about you?', 'Good! What can I help with?'],
    
    # Questions about topics
    r'\bwhat is python\b': ['Python is a popular programming language', 'Python is used for coding and data science'],
    r'\bwhat is machine learning\b': ['Machine learning teaches computers to learn from data', 'ML is a field of artificial intelligence'],
    r'\bmachine learning\b': ['Machine learning uses algorithms to learn patterns', 'ML is part of AI'],
    r'\bwhat is programming\b': ['Programming is writing instructions for computers', 'Programming creates software'],
    r'\bwhat is ai\b': ['AI is artificial intelligence', 'AI makes computers think and learn'],
    
    # Help
    r'\bhelp me\b': ['I can help you! What do you need?', 'What would you like help with?'],
    r'\bcan you help\b': ['Yes, I can help! What do you need?', 'Sure! Ask me anything'],
    r'\bhelp\b': ['How can I assist you?', 'What do you need help with?'],
    
    # Thanks
    r'\bthank(s| you)\b': ['You are welcome!', 'Happy to help!', 'No problem!', 'Anytime!'],
    
    # Goodbye
    r'\b(bye|goodbye)\b': ['Goodbye!', 'See you later!', 'Take care!'],
    
    # Learning
    r'\bhow.*(learn|study)\b': ['Start with basics and practice regularly', 'Take courses and build projects', 'Practice is key to learning'],
}

def get_template(user_input):
    """Check templates using regex for exact word matching"""
    user_lower = user_input.lower().strip()
    
    for pattern, responses in TEMPLATES.items():
        if re.search(pattern, user_lower):
            return random.choice(responses)
    return None

def generate_focused(prompt, max_len=10):
    """Generate short, focused response"""
    input_ids = torch.tensor([tokenizer.encode(prompt.lower())], dtype=torch.long).to(device)
    tokens = []
    current = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(current)
            logits = output[0, -1, :]
            
            # Block bad tokens
            for bad_id in [0, 1, 2, 3]:
                logits[bad_id] = -1e9
            
            # Lower temperature for coherence
            logits = logits / 0.65
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token in [0, 1, 2, 3]:
                break
                
            tokens.append(next_token)
            current = torch.cat([current, torch.tensor([[next_token]], device=device)], dim=1)
    
    response = tokenizer.decode(tokens)
    
    # Clean up
    words = response.split()[:10]  # Max 10 words
    if words:
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        result = ' '.join(unique_words)
        # Capitalize first letter
        return result[0].upper() + result[1:] if result else ""
    return ""

def smart_chat(user_input):
    """Smart chat: template first, then generation"""
    # Try template (exact matching)
    template_response = get_template(user_input)
    if template_response:
        return template_response
    
    # Generate with model
    generated = generate_focused(user_input, max_len=10)
    
    if generated and len(generated.split()) >= 2:
        return generated
    else:
        # Fallback
        return "I'm still learning. Can you ask something else?"

# Interactive
print("=" * 50)
print("SMART CHATBOT (Fixed Template Matching)")
print("=" * 50)
print("Type 'quit' to exit\n")

while True:
    user = input("You: ").strip()
    if user.lower() == 'quit':
        break
    if not user:
        continue
    
    response = smart_chat(user)
    print(f"Bot: {response}\n")
