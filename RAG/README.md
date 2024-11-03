## Overview
This repository contains scripts for generating and processing prompts using a large language model (LLM) and embeddings. The main components include:

- **RAG_GENERATE_EMBEDDING.PY**: Script to clean text, load and prepare data, and add documents to a vector store.
- **RAG_INFER_PROMPTS.PY**: Script to build conversation prompts and infer responses using an LLM.
- **RAG_INFERENCE.PY**: Script to initialize the LLM model, read test data, generate predictions, and save the results.

### Prerequisites

Install the required packages

```
pip install -r requirements.txt
```
### Usage    

##### Generate Embeddings
Run the embeddings script to clean text, load and prepare data, and add documents to a vector store.

```
python rag_generate_embedding.py
```
##### Generate Prompts
Run the prompts  script to initialize an embedding model, load a vector store, embed features, and generate classification prompts.

```
python rag_generate_prompts.py
```
##### Inference

Run the inference script to build conversation prompts and infer responses using an LLM.

```
python rag_infer_prompts.py
```