import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
import torch.nn.functional as F

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

def chat(prompt):
    """Generate response blocking token 3"""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    tokens = []
    current = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(15):
            output = model(current)
            logits = output[0, -1, :]
            
            # FORCE diversity - block bad tokens
            logits[0] = -1e9  # Block PAD
            logits[1] = -1e9  # Block UNK
            logits[2] = -1e9  # Block EOS  
            logits[3] = -1e9  # Block token 3
            
            # Add temperature
            logits = logits / 1.2
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token in [0, 1, 2, 3]:
                break
                
            tokens.append(next_token)
            current = torch.cat([current, torch.tensor([[next_token]], device=device)], dim=1)
    
    return tokenizer.decode(tokens) if tokens else "[No response generated]"

# Interactive
print("Simple Chatbot Test (type 'quit' to exit)\n")
while True:
    user = input("You: ").strip()
    if user.lower() == 'quit':
        break
    response = chat(user)
    print(f"Bot: {response}\n")
