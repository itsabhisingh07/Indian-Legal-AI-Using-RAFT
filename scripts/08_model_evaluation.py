import time
import os
import pandas as pd
from llama_cpp import Llama
import chromadb


USE_RAG = True  


if USE_RAG:
    
    MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf" 
    OUTPUT_CSV = "evaluation_results/finetuned_rag_results.csv"
else:
    
    MODEL_PATH = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" 
    OUTPUT_CSV = "evaluation_results/base_model_results.csv"


os.makedirs("evaluation_results", exist_ok=True)


csv_path = "test_dataset/golden_dataset.csv"
print(f"Loading dataset from: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    
    if "Test Question" in df.columns:
        test_questions = df["Test Question"].tolist()
    else:
       
        test_questions = df.iloc[:, 1].tolist() 
except Exception as e:
    print(f" Error reading CSV: {e}")
    exit()

print(f"Found {len(test_questions)} questions to evaluate.\n")

print(f"Loading Model: {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,          
    n_gpu_layers=-1,     
    verbose=False
)

if USE_RAG:
    print("Connecting to ChromaDB...")
    
   
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(SCRIPT_DIR, "..", "data", "chroma_db")
    
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="ipc_data")
   
    legal_index = {
        "article 21": "Protection of life and personal liberty",
        "section 379": "punishment for theft",
        "section 103": "punishment for murder",
        "section 105": "culpable homicide not amounting to murder",
        "murder": "103. Whoever commits murder shall be punished with death or imprisonment"
    }

results_data = []
print("\nStarting Evaluation Loop...\n")

for i, prompt in enumerate(test_questions):
    print(f"Evaluating [{i+1}/{len(test_questions)}]: {prompt}")
    
    start_time = time.time()
    
   
    if USE_RAG:
        clean_prompt = str(prompt).replace('"', '').replace("'", "")
        search_query = clean_prompt
        
       
        for keyword, meaning in legal_index.items():
            if keyword.lower() in clean_prompt.lower():
                search_query = f"{clean_prompt} {meaning}"
                break
                
       
        results = collection.query(
            query_texts=[search_query], 
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        
        valid_docs = [doc for doc, dist in zip(results['documents'][0], results['distances'][0]) if dist < 0.85]
        
        if len(valid_docs) == 0:
            final_answer = "I cannot find the relevant information in the provided legal text."
        else:
            retrieved_text = "\n\n".join(valid_docs)
            system_instruction = (
                "You are a strict Indian legal assistant. "
                "You must answer the user's question USING ONLY the provided Legal Context. "
                "CRITICAL RULES:\n"
                "1. If the context describes multiple variations of a crime, list them separately.\n"
                "2. NEVER combine different punishments into one single sentence.\n"
                "3. NO DISCLAIMERS. Absolutely do not say 'I cannot provide legal advice'.\n"
                "4. ANTI-HALLUCINATION: If the exact answer is not in the context, you must say EXACTLY: 'I cannot find the relevant information in the provided legal text.' Do NOT guess.\n\n"
                f"Legal Context:\n{retrieved_text}"
            )
            prefill = "Based directly on the provided text, "
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{prefill}"
            
            output = llm.create_completion(full_prompt, max_tokens=300, temperature=0.1)
            final_answer = prefill + output["choices"][0]["text"].strip()
            
   
    else:
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        output = llm.create_completion(full_prompt, max_tokens=300, temperature=0.7)
        final_answer = output["choices"][0]["text"].strip()

    
    generation_time = round(time.time() - start_time, 2)
    
   
    results_data.append({
        "Question": prompt,
        "AI Answer": final_answer,
        "Time Taken (s)": generation_time
    })


df_results = pd.DataFrame(results_data)

df_final = pd.concat([df.reset_index(drop=True), df_results.drop(columns=["Question"])], axis=1)

df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n Evaluation Complete! Results saved to '{OUTPUT_CSV}'")