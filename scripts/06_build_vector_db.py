import os
import json
import chromadb

def build_database():
    # 1. Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(SCRIPT_DIR, "..", "data", "processed_chunks", "legal_chunks.json")
    db_path = os.path.join(SCRIPT_DIR, "..", "data", "chroma_db")

    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    print("Loading legal chunks from JSON...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    
    print("Creating 'ipc_data' collection...")
   
    collection = client.get_or_create_collection(name="ipc_data")

    docs = []
    metadatas = []
    ids = []

    
    print("Preparing data vectors...")
    for i, chunk in enumerate(chunks):
        docs.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        ids.append(f"chunk_{i}")

    
    print(f"Injecting {len(docs)} legal chunks into the Vector Database.")
    print("This might take 1 to 3 minutes depending on your CPU...")
    
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✅ Database perfectly built and populated!")

if __name__ == "__main__":
    build_database()