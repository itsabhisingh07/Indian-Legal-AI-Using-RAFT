import os
import json
import time
from dotenv import load_dotenv
from google import genai

# Load API key securely from the .env file
load_dotenv()
client = genai.Client()

def generate_qa_pairs_daily(input_json_path, output_jsonl_path, daily_limit=200):
    print(f"Loading chunks from {input_json_path}...")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"Total chunks available: {len(chunks)}")
    
    start_index = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            start_index = sum(1 for line in f)
            
    print(f"Found {start_index} existing records in the dataset.")
    
    if start_index >= len(chunks):
        print("All chunks have already been processed! You are done.")
        return
        
    end_index = min(start_index + daily_limit, len(chunks))
    print(f"Today's batch: Processing chunks from index {start_index} to {end_index-1}...")
    print("Note: This will take about 20 minutes to respect free-tier rate limits.")

   
    with open(output_jsonl_path, 'a', encoding='utf-8') as outfile:
        
        for i in range(start_index, end_index):
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
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                
                clean_response = response.text.replace('```json', '').replace('```', '').strip()
                qa_dict = json.loads(clean_response)
                
                llama3_format = {
                    "messages": [
                        {"role": "user", "content": qa_dict["question"]},
                        {"role": "assistant", "content": qa_dict["answer"]}
                    ]
                }
                
                
                outfile.write(json.dumps(llama3_format) + "\n")
                print(f"Success: Generated QA for chunk {i+1} / {len(chunks)}")
                
                if i < end_index - 1:
                    time.sleep(6)
                    
            except Exception as e:
                print(f"Failed on chunk {i+1}. Skipping to next. Error: {e}")
                time.sleep(10)

    print(f"\nDaily quota complete! Added {end_index - start_index} new rows to {output_jsonl_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(SCRIPT_DIR, "..", "data", "processed_chunks", "legal_chunks.json")
    output_file = os.path.join(SCRIPT_DIR, "..", "data", "final_jsonl", "llama3_training_data.jsonl")
    
    generate_qa_pairs_daily(input_file, output_file, daily_limit=200)