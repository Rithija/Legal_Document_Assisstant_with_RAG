import pickle
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document 

def create_and_store_embeddings():
    """
    Loads processed chunks, creates embeddings, and stores them in ChromaDB.
    This is a one-time setup process to create the vector database.
    """
    # --- 1. Load the Processed Chunks ---
    processed_data_path = "processed_chunks.pkl"
    print(f"Loading processed data from '{processed_data_path}'...")
    with open(processed_data_path, "rb") as f:
        processed_chunks: list[Document] = pickle.load(f)

    if not processed_chunks:
        print("No chunks found to process. Please run the data processing script first.")
        return

    # --- 2. Initialize the Embedding Model ---

    print("Initializing the embedding model (this may take a moment)...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    # --- 3. Set up the ChromaDB Client ---
    db_path = "chroma_db"
    print(f"Setting up ChromaDB persistent client at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # The collection is where the vectors will be stored.
    collection_name = "supreme_court_judgments"
    collection = client.get_or_create_collection(name=collection_name)

    # --- 4. Embed and Store the Chunks in Batches ---
    print(f"Preparing to embed and store {len(processed_chunks)} chunks...")
    batch_size = 100 # Process chunks in batches for efficiency
    
    for i in tqdm(range(0, len(processed_chunks), batch_size), desc="Embedding and Storing"):
        batch = processed_chunks[i:i + batch_size]
        
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = []
        for doc in batch:
            metadata = doc.metadata.copy()
            if 'judges' in metadata and isinstance(metadata['judges'], list):
                metadata['judges'] = ', '.join(metadata['judges'])
            batch_metadatas.append(metadata)
        
        embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
        

        batch_ids = [f"{doc.metadata.get('source_file', 'unknown')}_{i+j}" for j, doc in enumerate(batch)]

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    print("\n✅ Embedding and storing complete.")
    print(f"Total documents in collection '{collection_name}': {collection.count()}")

if __name__ == "__main__":
    create_and_store_embeddings()