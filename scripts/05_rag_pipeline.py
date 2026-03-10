import os
import chromadb
from llama_cpp import Llama

def start_rag_chatbot():
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(SCRIPT_DIR, "..", "data", "chroma_db")
    model_path = os.path.join(SCRIPT_DIR, "..", "models", "llama-3-8b-instruct.Q4_K_M.gguf")

    print("Waking up ChromaDB...")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    
  
    collection = chroma_client.get_collection(name="ipc_data") 

    print("Loading Local AI Brain...")
    # 3. Load the Model 
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )
    
    print("\n" + "="*50)
    print(" Legal RAG System Active! Type 'exit' to quit.")
    print("="*50)

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Court adjourned. Shutting down...")
            break

       
        print("\n[System: Searching IPC Database...]")
        results = collection.query(
            query_texts=[user_query],
            n_results=5 
        )
        
        
        retrieved_text = "\n\n".join(results['documents'][0])
        source_metadata = results['metadatas'][0]
        
        
        system_instruction = (
            "You are a highly strict Indian legal assistant. "
            "You must answer the user's question USING ONLY the provided Legal Context. "
            "CRITICAL RULES:\n"
            "1. If the context describes multiple variations of a crime (e.g., basic theft vs. aggravated theft), you MUST list them separately with their specific punishments.\n"
            "2. NEVER combine different punishments into one single sentence.\n"
            "3. ONLY cite the exact law names and Section numbers that are explicitly written in the text. Do not use your prior knowledge.\n"
            "4. If the exact answer is not in the context, say 'I cannot find the answer in the provided text.'\n\n"
            f"Legal Context:\n{retrieved_text}"
        )
        
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        print("\nAI: ", end="", flush=True)
        
        
        stream = llm.create_completion(
            full_prompt,
            max_tokens=512,
            temperature=0.1, 
            stream=True
        )
        
        for chunk in stream:
            print(chunk["choices"][0]["text"], end="", flush=True)
        
        print(f"\n\n[Sources Used: {source_metadata}]")
        print("-" * 50)

if __name__ == "__main__":
    start_rag_chatbot()