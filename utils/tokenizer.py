import pickle
from collections import Counter

class Tokenizer:
    """
    Simple word-level tokenizer for text processing
    Supports special tokens: <PAD>, <UNK>, <EOS>
    """
    
    def __init__(self, vocab_size=10000):
        """
        Initialize tokenizer with special tokens
        
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        
        # Initialize with special tokens
        self.word2idx = {
            '<PAD>': 0,  # Padding token
            '<UNK>': 1,  # Unknown token
            '<EOS>': 2   # End of sequence token
        }
        
        self.idx2word = {
            0: '<PAD>',
            1: '<UNK>',
            2: '<EOS>'
        }
        
        self.vocab_built = False
    
    def fit(self, texts):
        """
        Build vocabulary from a list of texts
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Tokenize all texts and count word frequencies
        all_words = []
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
                all_words.extend(words)
        
        # Get most common words
        word_counts = Counter(all_words)
        
        # Reserve space for special tokens (already added 3)
        available_slots = self.vocab_size - 3
        most_common = word_counts.most_common(available_slots)
        
        print(f"Found {len(word_counts)} unique words")
        print(f"Keeping top {len(most_common)} words (vocab_size={self.vocab_size})")
        
        # Add words to vocabulary
        for idx, (word, count) in enumerate(most_common, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"✅ Vocabulary built: {len(self.word2idx)} tokens")
    
    def encode(self, text):
        """
        Convert text to list of token IDs
        
        Args:
            text: Input text string
            
        Returns:
            List of integer token IDs
        """
        if not isinstance(text, str):
            return []
        
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            # Get token ID, use <UNK> if word not in vocabulary
            token_id = self.word2idx.get(word, self.unk_id())
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids):
        """
        Convert list of token IDs back to text
        
        Args:
            token_ids: List of integer token IDs
            
        Returns:
            Decoded text string
        """
        words = []
        
        for idx in token_ids:
            # Skip padding tokens
            if idx == self.pad_id():
                continue
            
            # Stop at end of sequence
            if idx == self.eos_id():
                break
            
            # Get word from vocabulary
            word = self.idx2word.get(idx, '<UNK>')
            
            # Don't include special tokens in output
            if word not in ['<PAD>', '<UNK>', '<EOS>']:
                words.append(word)
        
        return ' '.join(words)
    
    # Special token ID getters
    
    def pad_id(self):
        """Return padding token ID"""
        return self.word2idx.get('<PAD>', 0)
    
    def unk_id(self):
        """Return unknown token ID"""
        return self.word2idx.get('<UNK>', 1)
    
    def eos_id(self):
        """Return end-of-sequence token ID"""
        return self.word2idx.get('<EOS>', 2)
    
    # Vocabulary info methods
    
    def vocab_length(self):
        """Return current vocabulary size"""
        return len(self.word2idx)
    
    def get_vocab(self):
        """Return word to index mapping"""
        return self.word2idx
    
    def get_word(self, idx):
        """Get word from token ID"""
        return self.idx2word.get(idx, '<UNK>')
    
    def get_token_id(self, word):
        """Get token ID from word"""
        return self.word2idx.get(word.lower(), self.unk_id())
    
    # Save and load methods
    
    def save(self, filepath):
        """
        Save tokenizer to file using pickle
        
        Args:
            filepath: Path to save tokenizer
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Tokenizer saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load tokenizer from file
        
        Args:
            filepath: Path to load tokenizer from
            
        Returns:
            Loaded Tokenizer object
        """
        with open(filepath, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    
    # Utility methods
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)
    
    def __repr__(self):
        """String representation"""
        return f"Tokenizer(vocab_size={len(self.word2idx)}, built={self.vocab_built})"
    
    def get_stats(self):
        """Return tokenizer statistics"""
        return {
            'vocab_size': len(self.word2idx),
            'max_vocab_size': self.vocab_size,
            'vocab_built': self.vocab_built,
            'special_tokens': {
                'PAD': self.pad_id(),
                'UNK': self.unk_id(),
                'EOS': self.eos_id()
            }
        }
