import json
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Initialize model and sampling parameters
def initialize_model(hf_model_id="srinathmkce/indoml_100k_llama_updated_dataset_epoch2"):
    model = LLM(model=hf_model_id, max_model_len=1024)
    params = SamplingParams(temperature=0.1, max_tokens=128)
    return model, params

# Load and sample dataset
def load_data(file_path, sample_size=100):
    df = pd.read_json(file_path, lines=True)
    print("Full dataset shape:", df.shape)
    sampled_df = df.sample(sample_size)
    print("Sampled dataset shape:", sampled_df.shape)
    return df, sampled_df

# Build conversation prompt from dataframe row
def build_prompt(row):
    description = row.get("description", "unknown")
    retailer = row.get("retailer", "unknown")
    price = row.get("price", "unknown")
    
    try:
        return [{"role": "human", "content": json.dumps({"description": description, "retailer": retailer, "price": price})}]
    except Exception as e:
        # Handle missing or malformed data
        return [{"role": "human", "content": json.dumps({"description": "unknown", "retailer": "unknown", "price": "unknown"})}]

# Generate and return predictions for a batch of prompts
def generate_predictions(llm, prompts, sampling_params):
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)
    predictions = [json.loads(output.outputs[0].text) for output in outputs]
    return predictions

# Save predictions to CSV
def save_to_csv(df, predictions, output_csv):
    result_df = pd.DataFrame(predictions)
    result_df = pd.concat([df["indoml_id"], result_df[['supergroup', 'group', 'module', 'brand']]], axis=1)
    result_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Format predictions into JSONL and save
def save_to_jsonl(df, output_jsonl):
    category_list = [
        {
            "indoml_id": row["indoml_id"],
            "supergroup": row["supergroup"],
            "group": row["group"],
            "module": row["module"],
            "brand": row["brand"]
        }
        for _, row in df.iterrows()
    ]
    
    with open(output_jsonl, "w") as fp:
        for item in category_list:
            fp.write(json.dumps(item) + "\n")
    print(f"Predictions saved to {output_jsonl}")

# Main processing function
def main():
    llm, sampling_params = initialize_model()
    df, sampled_df = load_data("final_test_data/final_test_data.features", sample_size=100)

    # Process sample for testing
    for _, row in sampled_df.iterrows():
        conversation = build_prompt(row)
        outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
        print("Sample output:", outputs[0].outputs[0].text)
        break  # Remove break if you want to process all sample rows

    # Generate predictions for all rows in dataset
    prompts = [build_prompt(row) for _, row in df.iterrows()]
    predictions = generate_predictions(llm, prompts, sampling_params)

    # Save predictions
    save_to_csv(df, predictions, "phase_2_test_set_predictions.csv")
    result_df = pd.DataFrame(predictions)
    result_df = pd.concat([df["indoml_id"], result_df[['supergroup', 'group', 'module', 'brand']]], axis=1)

    save_to_jsonl(result_df, "test_gradientgurus7.predict")

if __name__ == "__main__":
    main()

