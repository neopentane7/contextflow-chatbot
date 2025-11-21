import torch
import numpy as np
from typing import List, Tuple, Dict
import logging

class AdvancedInference:
    """Advanced inference strategies for better response generation"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    # ===== BEAM SEARCH =====
    
    def beam_search(
        self,
        input_text: str,
        beam_width: int = 5,
        max_length: int = 50,
        temperature: float = 0.7
    ) -> str:
        """
        Beam Search decoding - explores multiple hypotheses in parallel
        
        Advantages:
        - Better quality responses
        - Explores diverse paths
        - More natural language
        
        Trade-off: Slower than greedy
        """
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], dtype=torch.long).to(self.device)
        batch_size = input_ids.shape[0]
        
        # Initialize beams
        initial_tokens = input_ids[0].tolist()
        beams = [{'tokens': initial_tokens, 'score': 0.0}]
        completed_beams = []
        
        logging.info(f"DEBUG: Initial tokens: {initial_tokens}")
        
        with torch.no_grad():
            for step in range(max_length):
                if step % 10 == 0:
                    logging.info(f"DEBUG: Step {step}")
                candidates = []
                
                for beam_idx, beam in enumerate(beams):
                    beam_tokens = torch.tensor([beam['tokens']], dtype=torch.long).to(self.device)
                    
                    # Get model output
                    output = self.model(beam_tokens)
                    logits = output[0, -1, :] / temperature
                    
                    # Get top-k probabilities
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, beam_width)
                    
                    for idx, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
                        new_beam = {
                            'tokens': beam['tokens'] + [token_id.item()],
                            'score': beam['score'] + torch.log(prob).item()
                        }
                        candidates.append(new_beam)
                
                # Keep top beam_width candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_width]
                
                # Check if we should stop
                if all(beam['tokens'][-1] == self.tokenizer.eos_id() for beam in beams):
                    completed_beams = beams
                    break
            
            # If not completed, use best beam
            if not completed_beams:
                completed_beams = beams
        
        # Decode best beam
        best_tokens = completed_beams[0]['tokens'][len(initial_tokens):]  # Remove input tokens
        logging.info(f"DEBUG: Best tokens: {best_tokens}")
        response = self.tokenizer.decode(best_tokens)
        logging.info(f"DEBUG: Decoded response: '{response}'")
        return response
    
    # ===== TEMPERATURE SAMPLING =====
    
    def temperature_sampling(
        self,
        input_text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_length: int = 50
    ) -> str:
        """
        Temperature Sampling - generates diverse, creative responses
        
        Parameters:
        - temperature: 0.0 = deterministic, 1.0 = normal, >1.0 = creative
        - top_k: Keep only top K tokens
        - top_p: Keep tokens with cumulative prob <= p (nucleus sampling)
        
        Use cases:
        - Creative responses: temperature=1.2
        - Focused responses: temperature=0.6
        - Neutral: temperature=0.8
        """
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(current_input)
                logits = output[0, -1, :] / temperature
                
                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(logits, top_k, largest=False).indices
                    logits[indices_to_remove] = float('-inf')
                
                # Top-P (Nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=0)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[0] = False  # Keep at least one token
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                
                # Stop if EOS token
                if next_token == self.tokenizer.eos_id():
                    break
                
                # Append to input for next iteration
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
        
        response = self.tokenizer.decode(generated_tokens)
        return response
    
    # ===== CONSTRAINED DECODING =====
    
    def constrained_decoding(
        self,
        input_text: str,
        max_length: int = 50,
        temperature: float = 0.7,
        forbidden_words: List[str] = None
    ) -> str:
        """
        Constrained Decoding - generates responses avoiding certain words
        
        Use cases:
        - Avoid offensive language
        - Prevent sensitive information
        - Domain-specific constraints
        """
        forbidden_ids = []
        if forbidden_words:
            for word in forbidden_words:
                token_ids = self.tokenizer.encode(word)
                forbidden_ids.extend(token_ids)
        
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], dtype=torch.long).to(self.device)
        generated_tokens = []
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(current_input)
                logits = output[0, -1, :].clone()
                
                # Mask forbidden tokens
                for forbidden_id in forbidden_ids:
                    logits[forbidden_id] = float('-inf')
                
                # Apply temperature
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Greedy selection (or can use sampling)
                next_token = torch.argmax(probs).item()
                
                generated_tokens.append(next_token)
                
                if next_token == self.tokenizer.eos_id():
                    break
                
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
        
        response = self.tokenizer.decode(generated_tokens)
        return response
    
    # ===== ENSEMBLE METHOD =====
    
    def ensemble_decoding(
        self,
        input_text: str,
        methods: List[str] = None,
        **kwargs
    ) -> str:
        """
        Ensemble Decoding - combine multiple methods for better results
        
        Methods: ['greedy', 'beam_search', 'temperature_sampling']
        """
        if methods is None:
            methods = ['beam_search', 'temperature_sampling']
        
        responses = []
        
        for method in methods:
            if method == 'greedy':
                response = self._greedy_decoding(input_text, kwargs.get('max_length', 50))
            elif method == 'beam_search':
                response = self.beam_search(
                    input_text,
                    beam_width=kwargs.get('beam_width', 5),
                    max_length=kwargs.get('max_length', 50),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif method == 'temperature_sampling':
                response = self.temperature_sampling(
                    input_text,
                    temperature=kwargs.get('temperature', 0.8),
                    top_k=kwargs.get('top_k', 50),
                    top_p=kwargs.get('top_p', 0.9),
                    max_length=kwargs.get('max_length', 50)
                )
            
            responses.append(response)
        
        # Return the longest (usually best) response
        return max(responses, key=len) if responses else ""
    
    def _greedy_decoding(self, input_text: str, max_length: int = 50) -> str:
        """Simple greedy decoding"""
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], dtype=torch.long).to(self.device)
        generated_tokens = []
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(current_input)
                next_token = torch.argmax(output[0, -1, :]).item()
                
                generated_tokens.append(next_token)
                
                if next_token == self.tokenizer.eos_id():
                    break
                
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
        
        response = self.tokenizer.decode(generated_tokens)
        return response
