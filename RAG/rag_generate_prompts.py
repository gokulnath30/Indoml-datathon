import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2", use_cuda=False):
    """
    Initializes the embedding model with optional CUDA support.
    """
    model_kwargs = {'device': 'cuda:0'} if use_cuda else {}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )


def load_vectorstore(folder_path, embedding_model):
    """
    Loads the FAISS vector store from a local folder.
    """
    return FAISS.load_local(
        folder_path=folder_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


def embed_features_in_batches(feature_df, embedding_model, batch_size=100):
    """
    Embeds features in batches and adds the embeddings to the DataFrame.
    """
    batched_vectors = []
    for i in tqdm(range(0, len(feature_df), batch_size), desc="Embedding batches"):
        batch_texts = feature_df['description'][i:i + batch_size].tolist()
        batch_vectors = embedding_model.embed_documents(batch_texts)
        batched_vectors.extend(batch_vectors)
    feature_df['vector'] = batched_vectors
    return feature_df


def generate_prompt_for_row(row, vectorstore, top_k=5):
    """
    Generates a classification prompt for a given row based on similar items from the vector store.
    """
    results = vectorstore.similarity_search_by_vector(
        row["vector"],
        filters={"retailer": row["retailer"]},
        k=top_k,
    )
    
    similar_products = [
        {
            "description": res.page_content,
            "retailer": res.metadata["retailer"],
            "labels": res.metadata["labels"]
        }
        for res in results
    ]
    prompt = f"""Your task is to classify a product based on its description. Predict the label in the format: supergroup__group__module__brand.
    Product description: {row["description"]}
    Product retailer: {row["retailer"]}
    Context: {similar_products} ({top_k} similar products)
    Based on the context, provide the best-suited label for the product. No explanation needed—just give the matching label.
    """
    return prompt


def generate_prompts(dataset_df, vectorstore, top_k=5, num_workers=4):
    """
    Generates prompts for each row in the dataset in parallel using a thread pool.
    """
    prompts = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_prompt_for_row, row, vectorstore, top_k)
            for _, row in dataset_df.iterrows()
        ]
        for future in tqdm(futures, total=len(futures), desc="Generating prompts"):
            try:
                prompts.append(future.result())
            except Exception as e:
                print(f"Error generating prompt for a row: {e}")
                prompts.append(None)

    dataset_df = dataset_df.copy()
    dataset_df["prompt"] = prompts
    dataset_df.drop(columns=["vector"], inplace=True)
    return dataset_df


def main():
    # Initialize embedding model and vector store
    embedding_model = initialize_embedding_model()
    vectorstore = load_vectorstore("./vector_store", embedding_model)
    
    # Load and process the test dataset
    test_dataset = pd.read_json("dataset/final_test_data.features", lines=True)
    test_dataset = embed_features_in_batches(test_dataset, embedding_model, batch_size=100)
    
    # Generate prompts
    prompts_df = generate_prompts(test_dataset, vectorstore, top_k=5, num_workers=64)
    
    # Save prompts to file
    prompts_df.to_json("test_data.prompts", orient="records", lines=True)


if __name__ == "__main__":
    main()
