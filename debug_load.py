import torch
import sys
import os
import pickle

# Add root to path
sys.path.append(os.getcwd())

from utils.tokenizer import Tokenizer
from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG

def test_load():
    print("Testing load...")
    
    # 1. Test Tokenizer
    print("Loading tokenizer...")
    if os.path.exists('models/tokenizer.pkl'):
        try:
            tokenizer = Tokenizer.load('models/tokenizer.pkl')
            print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    else:
        print("models/tokenizer.pkl does not exist")

    # 2. Test Model
    print("Loading model...")
    checkpoint_path = 'models/checkpoints/checkpoint_epoch_10.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("Checkpoint loaded")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print(f"{checkpoint_path} does not exist")

if __name__ == "__main__":
    test_load()
