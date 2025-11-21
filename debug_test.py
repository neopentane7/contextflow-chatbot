import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer

# Load
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

# Test
test_input = "Hello, how are you?"
print(f"Input: {test_input}")

input_ids = torch.tensor([tokenizer.encode(test_input)], dtype=torch.long).to(device)
print(f"Input IDs: {input_ids.tolist()}")

generated_tokens = []
current_input = input_ids.clone()

with torch.no_grad():
    for step in range(20):  # Generate up to 20 tokens
        output = model(current_input)
        next_token = torch.argmax(output[0, -1, :]).item()
        
        print(f"Step {step+1}: Generated token {next_token} = '{tokenizer.get_word(next_token)}'")
        
        generated_tokens.append(next_token)
        
        if next_token == 0:  # Padding
            print("  → Hit padding token, stopping")
            break
        
        current_input = torch.cat([
            current_input,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)

print(f"\nGenerated token IDs: {generated_tokens}")
print(f"Decoded response: '{tokenizer.decode(generated_tokens)}'")
