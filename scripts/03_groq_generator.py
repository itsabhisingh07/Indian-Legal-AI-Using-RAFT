import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

# Load API key securely
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_qa_pairs_groq(input_json_path, output_jsonl_path, daily_limit=200):
    print(f"Loading chunks from {input_json_path}...")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"Total chunks available: {len(chunks)}")
    
    # 1. Check where Groq left off (counting how many it has already processed)
    processed_count = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f)
            
    print(f"Found {processed_count} existing records in the Groq dataset.")
    
    # 2. Start from the BACK of the list to avoid colliding with Gemini
    start_index = (len(chunks) - 1) - processed_count
    
    if start_index < 0:
        print("All chunks have been processed by Groq! You are done.")
        return
        
    end_index = max(-1, start_index - daily_limit)
    print(f"Today's batch: Processing chunks in REVERSE from index {start_index} down to {end_index + 1}...")

    # Open the separate Groq file in append mode
    with open(output_jsonl_path, 'a', encoding='utf-8') as outfile:
        
        # Loop backwards
        for i in range(start_index, end_index, -1):
            chunk_data = chunks[i]
            legal_text = chunk_data["text"]
            source = chunk_data["metadata"]["source"]
            
            prompt = f"""
            You are a Senior Indian Legal Professor. Read the following legal text from the {source}.
            Generate ONE complex, real-world legal question that can be answered using ONLY this text.
            Then, provide a professional, detailed answer.
            
            Format your response STRICTLY as a JSON object like this, with no markdown formatting or extra text:
            {{"question": "The question here", "answer": "The detailed answer here"}}
            
            Legal Text: {legal_text}
            """
            
            try:
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile", 
                )
                
                clean_response = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
                qa_dict = json.loads(clean_response)
                
                llama3_format = {
                    "messages": [
                        {"role": "user", "content": qa_dict["question"]},
                        {"role": "assistant", "content": qa_dict["answer"]}
                    ]
                }
                
                outfile.write(json.dumps(llama3_format) + "\n")
                print(f"Success: Generated QA using Groq for chunk index {i}")
                
               
                time.sleep(4)
                    
            except Exception as e:
                print(f"Failed on chunk {i}. Skipping to next. Error: {e}")
                time.sleep(10)

    print(f"Groq daily quota complete! Data saved to {output_jsonl_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(SCRIPT_DIR, "..", "data", "processed_chunks", "legal_chunks.json")
    
    
    output_file = os.path.join(SCRIPT_DIR, "..", "data", "final_jsonl", "llama3_training_data_groq.jsonl")
    
    generate_qa_pairs_groq(input_file, output_file, daily_limit=200)