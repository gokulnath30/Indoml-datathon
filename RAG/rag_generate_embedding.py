import faiss
import re
from uuid import uuid4
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


def initialize_vector_store(model_name="sentence-transformers/all-mpnet-base-v2",use_cuda=False):
    """
    Initializes a FAISS vector store with HuggingFace embeddings and an in-memory docstore.
    """
    model_kwargs = {'device': 'cuda:0'} if use_cuda else {} 
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )



def clean_text(text):
    """
    Cleans input text by removing numbers, single-letter words, and extra spaces.
    """
    cleaned_text = re.sub(r'\b\w\b|\d+', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()


def load_and_prepare_data(features_path, labels_path, sample_size):
    """
    Loads, merges, and prepares the dataset for processing.
    """
    train_dataset = pd.read_json(features_path, lines=True)
    labels = pd.read_json(labels_path, lines=True)
    
    # Merge datasets and sample
    dataset = pd.merge(train_dataset, labels, on="indoml_id")[:sample_size]
    
    # Create label column and clean descriptions
    dataset["labels"] = dataset["supergroup"] + "__" + dataset["group"] + "__" + dataset["module"] + "__" + dataset["brand"]
    dataset["description"] = dataset["description"].apply(clean_text)
    
    return dataset[["indoml_id", "description", "retailer", "price", "labels"]]


def add_documents_to_store(vector_store, dataset, batch_size=100):
    """
    Batches the dataset and adds documents to the vector store.
    """
    for start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[start:start + batch_size]
        documents = [
            Document(
                page_content=row["description"],
                metadata={
                    "indoml_id": row["indoml_id"],
                    "retailer": row["retailer"],
                    "price": row["price"],
                    "labels": row["labels"]
                }
            ) for _, row in batch.iterrows()
        ]
        ids = [str(uuid4()) for _ in documents]
        vector_store.add_documents(documents=documents, ids=ids)


def main():
    vector_store = initialize_vector_store(model_name="sentence-transformers/all-mpnet-base-v2", use_cuda=False)
    dataset = load_and_prepare_data("dataset/train.features", "dataset/train.labels",100)
    add_documents_to_store(vector_store, dataset)
    vector_store.save_local("vector_store")


if __name__ == "__main__":
    main()
