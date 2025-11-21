import sys
import os
import pickle

# Add root to path
sys.path.append(os.getcwd())

from utils.tokenizer import Tokenizer

def inspect_tokenizer():
    print("Loading tokenizer...")
    if os.path.exists('models/tokenizer.pkl'):
        try:
            tokenizer = Tokenizer.load('models/tokenizer.pkl')
            print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
            
            # Check ID 3
            word_3 = tokenizer.decode([3])
            print(f"ID 3 decodes to: '{word_3}'")
            
            if 3 in tokenizer.idx2word:
                print(f"ID 3 in idx2word: '{tokenizer.idx2word[3]}'")
            
            if 2 in tokenizer.idx2word:
                print(f"ID 2 in idx2word: '{tokenizer.idx2word[2]}'")
            else:
                print("ID 2 NOT in idx2word")
                
            # Check special tokens
            print(f"PAD: {tokenizer.pad_id()}")
            print(f"UNK: {tokenizer.unk_id()}")
            print(f"EOS: {tokenizer.eos_id()}")
            
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    else:
        print("models/tokenizer.pkl does not exist")

if __name__ == "__main__":
    inspect_tokenizer()
