import os
from llama_cpp import Llama

def chat_with_model():
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, "..", "models", "llama-3-8b-instruct.Q4_K_M.gguf")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading local AI brain into the GPU... This will take about 10 seconds.")

    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=2048,      
            verbose=False    
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nModel loaded successfully! Type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Shutting down the local server...")
            break
            
        
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        print("AI: ", end="", flush=True)
        
        
        stream = llm.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.3, 
            stream=True
        )
        
        for chunk in stream:
            text = chunk["choices"][0]["text"]
            print(text, end="", flush=True)
        print()

if __name__ == "__main__":
    chat_with_model()