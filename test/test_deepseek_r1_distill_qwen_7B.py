"""
This script downloads DeepSeek-R1-Distill-Qwen-7B on a MacPro.
It loads the model, tokenizes input text, and generates responses.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    model_name = "deepseek-ai/deepseek-r1-distill-qwen-7b"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test example
    prompt = "Explain what is machine learning in simple terms:"
    response = generate_response(model, tokenizer, prompt)
    
    print("Prompt:", prompt)
    print("\nResponse:", response)