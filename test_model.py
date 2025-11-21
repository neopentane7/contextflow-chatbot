import torch
import sys
import os

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer

class ChatbotTester:
    def __init__(self, checkpoint_path='models/checkpoints/checkpoint_epoch_10.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.load('models/tokenizer.pkl')
        print(f"✅ Tokenizer loaded (vocab size: {len(self.tokenizer.word2idx)})\n")
        
        # Load model
        print("Loading model...")
        self.model = HybridChatbotModel(
            vocab_size=len(self.tokenizer.word2idx),
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            num_heads=MODEL_CONFIG['num_heads'],
            dropout=MODEL_CONFIG['dropout']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"✅ Model loaded from epoch {epoch} (val_loss: {val_loss})\n")
        
        self.conversation_history = []
    
    def generate_response(self, user_input, max_length=50, use_context=True):
        """Generate response to user input"""
        
        # Build input with context
        if use_context and self.conversation_history:
            context = "\n".join([
                f"User: {turn['user']}\nBot: {turn['bot']}" 
                for turn in self.conversation_history[-3:]  # Last 3 turns
            ])
            full_input = f"{context}\nUser: {user_input}"
        else:
            full_input = user_input
        
        # Encode
        input_ids = torch.tensor([self.tokenizer.encode(full_input)], dtype=torch.long).to(self.device)
        
        # Generate (greedy decoding)
        generated_tokens = []
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(current_input)
                next_token = torch.argmax(output[0, -1, :]).item()
                
                generated_tokens.append(next_token)
                
                # Stop if EOS or padding
                if next_token == self.tokenizer.eos_id() or next_token == 0:
                    break
                
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
        
        # Decode
        response = self.tokenizer.decode(generated_tokens)
        
        # Add to history
        if use_context:
            self.conversation_history.append({
                'user': user_input,
                'bot': response
            })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared\n")
    
    def run_test_suite(self):
        """Run comprehensive test suite"""
        print("="*60)
        print("CHATBOT TEST SUITE")
        print("="*60)
        
        # Test 1: Simple Q&A
        print("\n📝 TEST 1: Simple Questions\n")
        test_questions = [
            "Hello, how are you?",
            "What is Python?",
            "How do I learn programming?",
            "What is machine learning?",
            "Can you help me?"
        ]
        
        for question in test_questions:
            response = self.generate_response(question, use_context=False)
            print(f"User: {question}")
            print(f"Bot:  {response}\n")
        
        self.clear_history()
        
        # Test 2: Context Awareness
        print("="*60)
        print("\n🔗 TEST 2: Multi-Turn Context\n")
        
        context_test = [
            "I am learning Python programming",
            "What should I learn next?",
            "Any good resources for that?",
            "How long will it take?"
        ]
        
        for turn in context_test:
            response = self.generate_response(turn, use_context=True)
            print(f"User: {turn}")
            print(f"Bot:  {response}\n")
        
        self.clear_history()
        
        # Test 3: Greetings & Politeness
        print("="*60)
        print("\n👋 TEST 3: Greetings & Politeness\n")
        
        greetings = [
            "Hi there!",
            "Thank you",
            "That's helpful",
            "Goodbye"
        ]
        
        for greeting in greetings:
            response = self.generate_response(greeting, use_context=False)
            print(f"User: {greeting}")
            print(f"Bot:  {response}\n")
        
        print("="*60)
        print("\n✅ Test suite complete!\n")
    
    def interactive_mode(self):
        """Interactive chat mode"""
        print("="*60)
        print("INTERACTIVE CHAT MODE")
        print("="*60)
        print("Type 'quit' to exit, 'clear' to reset conversation\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                self.clear_history()
                continue
            
            if not user_input:
                continue
            
            response = self.generate_response(user_input, use_context=True)
            print(f"Bot: {response}\n")


def main():
    print("\n" + "="*60)
    print("CHATBOT MODEL TESTER")
    print("="*60 + "\n")
    
    # Initialize tester
    try:
        tester = ChatbotTester()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Menu
    while True:
        print("\nChoose an option:")
        print("1. Run test suite (automated tests)")
        print("2. Interactive chat mode")
        print("3. Single question test")
        print("4. Exit")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == '1':
            tester.run_test_suite()
        
        elif choice == '2':
            tester.interactive_mode()
        
        elif choice == '3':
            question = input("\nEnter your question: ").strip()
            if question:
                response = tester.generate_response(question, use_context=False)
                print(f"\nBot: {response}\n")
        
        elif choice == '4':
            print("\nGoodbye! 👋\n")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == '__main__':
    main()
