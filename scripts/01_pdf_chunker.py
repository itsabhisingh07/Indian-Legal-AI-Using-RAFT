import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_legal_pdfs_with_metadata(pdf_configs, output_path):
    all_chunks_data = []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for config in pdf_configs:
        pdf_path = config["path"]
        source_name = config["source"]
        status = config["status"]
        
        if not os.path.exists(pdf_path):
            print(f"Warning: File not found at {pdf_path}. Skipping.")
            continue

        print(f"Loading and chunking {source_name}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        chunks = splitter.split_documents(docs)
        
        for chunk in chunks:
            clean_text = chunk.page_content.replace('\n', ' ')
            all_chunks_data.append({
                "text": clean_text,
                "metadata": {
                    "source": source_name,
                    "status": status
                }
            })
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_data, f, indent=4)
        
    print(f"Success! {len(all_chunks_data)} total chunks saved with metadata to {output_path}")

if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    pdf_files_to_process = [
        {
            "path": os.path.join(SCRIPT_DIR, "..", "data", "raw_pdfs", "IPC_Document.pdf"),
            "source": "IPC",
            "status": "historical_pre_july_2024"
        },
        {
            "path": os.path.join(SCRIPT_DIR, "..", "data", "raw_pdfs", "BNS_Document.pdf"),
            "source": "BNS",
            "status": "active_post_july_2024"
        },
        {
            "path": os.path.join(SCRIPT_DIR, "..", "data", "raw_pdfs", "Constitution_of_India.pdf"),
            "source": "Constitution",
            "status": "active_supreme_law"
        },
        {
            "path": os.path.join(SCRIPT_DIR, "..", "data", "raw_pdfs", "BNSS_Document.pdf"), 
            "source": "BNSS",
            "status": "active_post_july_2024"
        },
        {
            "path": os.path.join(SCRIPT_DIR, "..", "data", "raw_pdfs", "BSA_Document.pdf"), 
            "source": "BSA",
            "status": "active_post_july_2024"
        }
    ]
    
    output_file = os.path.join(SCRIPT_DIR, "..", "data", "processed_chunks", "legal_chunks.json")
    
    chunk_legal_pdfs_with_metadata(pdf_files_to_process, output_file)