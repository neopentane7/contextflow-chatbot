import torch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from utils.tokenizer import Tokenizer
from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG

def test_model():
    print("Loading tokenizer...")
    tokenizer = Tokenizer.load('models/tokenizer.pkl')
    
    print("Loading model...")
    device = torch.device('cpu')
    model = HybridChatbotModel(
        vocab_size=len(tokenizer),
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    checkpoint_path = 'models/checkpoints/checkpoint_epoch_10.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    input_text = "Hello you"
    print(f"Input: {input_text}")
    
    input_ids = tokenizer.encode(input_text)
    print(f"Input IDs: {input_ids}")
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # output shape: (batch, seq_len, vocab_size)
        
        # Get top 5 predictions
        probs = torch.softmax(output[0, -1], dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print("\nTop 5 predictions:")
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = tokenizer.decode([idx])
            if idx == 3: token = "<EOS>"
            if idx == 2: token = "<SOS>"
            print(f"Token ID {idx} ('{token}'): {prob:.4f}")

if __name__ == "__main__":
    test_model()
