"""
This script downloads an tests a distilled DeepSeek model locally on a MacPro.
It loads the model, tokenizes input text, and generates responses.
It includes unit tests and several examples to check the AI quality
"""

# Standard library imports (included with Python)
import unittest    # Python's built-in testing framework
import sys
import warnings
from datetime import datetime
import os

# Third-party library imports (need to be installed via pip)
import torch      # PyTorch for tensor operations
import urllib3
from transformers import (    # HuggingFace Transformers for AI models
    AutoModelForCausalLM,    # For loading language models
    AutoTokenizer           # For text tokenization
)

# Suppress urllib3 warnings about SSL
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

class CustomTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        kwargs['buffer'] = False  # Force disable output buffering
        super().__init__(*args, **kwargs)

class TestDeepSeekModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure SSL warnings don't interfere with output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("\n=== Initializing DeepSeek Model ===")
            cls.model_name = "deepseek-ai/deepseek-coder-1.3b-base"
            print(f"Loading model: {cls.model_name}")
            
            # Initialize tokenizer with explicit padding settings
            cls.tokenizer = AutoTokenizer.from_pretrained(
                cls.model_name,
                padding_side='right',
                truncation_side='right'
            )
            
            # Configure padding tokens explicitly
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
                cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
            
            # Initialize model with proper token IDs
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls.model_name,
                pad_token_id=cls.tokenizer.pad_token_id,
                eos_token_id=cls.tokenizer.eos_token_id
            )
            
            # Set device
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.model = cls.model.to(cls.device)
            
            print("Model loaded successfully!")
            print(f"Using device: {cls.device}")
            print("="*50 + "\n")
            sys.stdout.flush()
        
    def _generate_response(self, prompt):
        """Helper method for consistent text generation"""
        # Force print to terminal
        print("\n" + "="*80, flush=True)
        print(f"GENERATING RESPONSE FOR: {prompt}", flush=True)
        print("-"*80, flush=True)

        # Encode with attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True
        ).to(self.device)

        # Generate with proper configuration
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # This includes both input_ids and attention_mask
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Force print to terminal
        print("\nGENERATED RESPONSE:", flush=True)
        print("-"*80, flush=True)
        print(response, flush=True)
        print("="*80 + "\n", flush=True)
        
        return response

    def test_model_loading(self):
        # Verify that both model and tokenizer are properly initialized
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
        
    def test_tokenization(self):
        # Test the tokenizer's ability to convert text to tokens
        test_text = "Write a Python function to calculate factorial"
        # Add padding and attention mask
        encoded = self.tokenizer(
            test_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200
        )
        self.assertIsInstance(encoded.input_ids, torch.Tensor)
        self.assertIsInstance(encoded.attention_mask, torch.Tensor)
        self.assertTrue(encoded.input_ids.shape[1] > 0)
        
    def test_generation(self):
        print("\nSingle Prompt Test:")
        prompt = "Write a Python function to calculate factorial"
        response = self._generate_response(prompt)
        print(f"Response length: {len(response)}")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
    def test_multiple_prompts(self):
        print("\nMultiple Prompts Test:")
        test_prompts = [
            "Write a Python class for a Stack data structure",
            "Implement bubble sort in Python",
            "Write a function to check if a string is palindrome"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            with self.subTest(prompt=prompt):
                print(f"\nTest {i}/{len(test_prompts)}:")
                response = self._generate_response(prompt)
                self.assertTrue(len(response) > 0)

# Replace the run_tests() function and main block with this simpler version:
if __name__ == '__main__':
    # Create test instance manually
    test = TestDeepSeekModel()
    
    # Setup
    print("\n=== Setting up DeepSeek Model ===", flush=True)
    TestDeepSeekModel.setUpClass()
    
    # Run individual tests directly
    print("\n=== Running Tests ===", flush=True)
    
    print("\nTest 1: Model Loading", flush=True)
    test.test_model_loading()
    
    print("\nTest 2: Tokenization", flush=True)
    test.test_tokenization()
    
    print("\nTest 3: Single Generation", flush=True)
    test.test_generation()
    
    print("\nTest 4: Multiple Prompts", flush=True)
    test.test_multiple_prompts()
    
    print("\n=== All Tests Completed ===", flush=True)




