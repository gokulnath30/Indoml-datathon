import pandas as pd
import tiktoken
import json
from tqdm import tqdm

def load_data(filepath):
    """Load data from a JSON file."""
    return pd.read_json(filepath, lines=True)

def preprocess_data(df):
    """Preprocess data by filling NaN values in the 'price' column and creating a prompt column."""
    df["price"].fillna(df["price"].median(), inplace=True)
    df["prompt"] = df["description"] + " | " + df["retailer"] + " | " + df["price"].astype(str) + " ->"
    return df

def calculate_token_length(df, encoding):
    """Calculate the token length of each prompt."""
    df["length"] = df["prompt"].apply(lambda x: len(encoding.encode(x)))
    total_length = df["length"].sum() / 10 ** 6
    return df, total_length

def create_jsonl_file(df, start, end, system_prompt, output_file):
    """Create a JSONL file from the data."""
    with open(output_file, "w") as fp:
        for index, row in tqdm(df[start:end].iterrows(), total=end-start):
            indoml_id = row["indoml_id"]
            prompt = row["prompt"]
            task = {
                "custom_id": f"{indoml_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "ft:gpt-4o-mini-2024-07-18:personal:with-price-epoch2:AMVcey3L",
                    "temperature": 0.1,
                    "max_completion_tokens": 64,
                    "stop": "|",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                }
            }
            fp.write(json.dumps(task) + "\n")

def main():
    # Parameters
    filepath = "../final_test_data/final_test_data.features"
    system_prompt = "Classify:"
    batch_size = 50000
    total_rows = 184664  # Total number of rows in the dataset

    # Load and preprocess data
    df = load_data(filepath)
    df = preprocess_data(df)

    # Calculate token length
    encoding = tiktoken.get_encoding("cl100k_base")
    df, total_length = calculate_token_length(df, encoding)
    print(f"Total length in millions of tokens: {total_length}")

    # Create JSONL files in batches
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        output_file = f"indoml_{start}_to_{end}.jsonl"
        create_jsonl_file(df, start, end, system_prompt, output_file)

if __name__ == "__main__":
    main()
