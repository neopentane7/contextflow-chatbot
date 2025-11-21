import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import HybridChatbotModel, MODEL_CONFIG
from utils.tokenizer import Tokenizer
from training.dataset import ConversationDataset


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(input_ids)
            
            # Calculate loss
            loss = self.criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                output = self.model(input_ids)
                loss = self.criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, filepath):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, filepath)
        print(f"Checkpoint saved to {filepath}")


def main():
    print("=" * 60)
    print("CHATBOT TRAINING PIPELINE")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Step 1: Load and prepare tokenizer
    print("\n[1/5] Preparing tokenizer...")
    data = pd.read_csv('data/merged_training_data.csv')
    
    tokenizer = Tokenizer(vocab_size=MODEL_CONFIG['vocab_size'])
    all_texts = pd.concat([data['input'], data['response']]).tolist()
    tokenizer.fit(all_texts)
    tokenizer.save('models/tokenizer.pkl')
    
    # Update vocab size in config
    MODEL_CONFIG['vocab_size'] = len(tokenizer.word2idx)
    
    # Step 2: Create datasets
    print("\n[2/5] Creating datasets...")
    full_dataset = ConversationDataset('data/merged_training_data.csv', tokenizer)
    
    # Split into train and validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=MODEL_CONFIG['batch_size'],
        num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Step 3: Initialize model
    print("\n[3/5] Initializing model...")
    model = HybridChatbotModel(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ===== NEW: Resume from checkpoint if available =====
    checkpoint_path = 'models/checkpoints/checkpoint_epoch_10.pt'
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint at {checkpoint_path}, loading for resume...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming training from epoch {start_epoch + 1}")
    # =====================================================
    
    # Step 4: Create trainer
    print("\n[4/5] Setting up trainer...")
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Step 5: Training loop
    print("\n[5/5] Starting training...")
    os.makedirs('models/checkpoints', exist_ok=True)
    
    for epoch in range(start_epoch, MODEL_CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{MODEL_CONFIG['epochs']}")
        
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(
                epoch, val_loss, 
                'models/checkpoints/best_model.pt'
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(
                epoch, val_loss,
                f'models/checkpoints/checkpoint_epoch_{epoch+1}.pt'
            )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
