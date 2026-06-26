import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="page_index_rag"
)

def store_chunk(
        chunk_id,
        text,
        page,
        chunk):

    collection.add(
        ids=[str(chunk_id)],
        documents=[text],
        metadatas=[
            {
                "page": page,
                "chunk": chunk
            }
        ]
    )

def retrieve_vector(
        query,
        top_k=5):

    return collection.query(
        query_texts=[query],
        n_results=top_k
    )

def clear_vectors():

    global collection

    try:

        client.delete_collection(
            "page_index_rag"
        )

    except:
        pass

    collection = client.get_or_create_collection(
        name="page_index_rag"
    )