import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from getpass import getpass
from openai import OpenAI

def load_data(features_path, labels_path):
    """Load training features and labels from JSON files."""
    features_df = pd.read_json(features_path, lines=True)
    labels_df = pd.read_json(labels_path, lines=True)
    return features_df, labels_df

def merge_data(features_df, labels_df):
    """Merge features and labels DataFrames on 'indoml_id'."""
    merged_df = pd.merge(features_df, labels_df, on="indoml_id")
    return merged_df

def create_labels(df):
    """Create a combined label column from existing categorical columns."""
    df["label"] = df['supergroup'] + " __" + df['group'] + " __" + df['module'] + " __" + df['brand'] + "|"
    return df

def undersample_data(df):
    """Undersample the dataset to balance class distribution."""
    count_df = df["label"].value_counts().reset_index(name='count')
    single_label = count_df[count_df["count"] < 5]["label"].tolist()
    single_label_df = df[df["label"].isin(single_label)]

    df = df[~df["label"].isin(single_label)]
    df_min_samples = df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True
    )
    
    undersampled_df = pd.concat([single_label_df, df_min_samples], axis=0)
    
    remaining_df, val_df = train_test_split(df, train_size=9000, stratify=df['label'], random_state=42)
    undersampled_df = pd.concat([undersampled_df, remaining_df], axis=0)
    
    return undersampled_df, val_df

def prepare_dataset(df):
    """Prepare the dataset by creating prompts and labels."""
    df["price"].fillna(df["price"].median(), inplace=True)    
    label_list = []
    prompt_list = []
    
    for _, row in tqdm(df.iterrows()):
        desc = row["description"]
        retailer = row["retailer"]
        price = row["price"]
        human_val = f"{desc} | {retailer} | {price} ->"
        prompt_list.append(human_val)
        gpt_val = row["label"]
        label_list.append(gpt_val)

    new_df = pd.DataFrame(zip(prompt_list, label_list), columns=['prompt', 'completion'])
    print("Data Shape: ", new_df.shape)
    return new_df

def save_to_jsonl(df, filename):
    """Save the DataFrame to a JSONL file."""
    lines = []
    system_msg = "Your task is to classify the user text."
    
    for _, row in df.iterrows():
        line = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}
            ]
        }
        lines.append(line)

    with open(filename, "w") as f:
        for line in lines:
            json.dump(line, f)
            f.write("\n")

def main():
    # Load data
    features_path = r"training_data/train.features"
    labels_path = r"training_data/train.labels"
    features_df, labels_df = load_data(features_path, labels_path)

    print(features_df.shape, labels_df.shape)
    merged_df = merge_data(features_df, labels_df)

    # Process data
    merged_df = create_labels(merged_df)
    undersampled_df, val_df = undersample_data(merged_df)
    
    # Prepare datasets
    train_df = prepare_dataset(undersampled_df)
    val_sub_df, _ = train_test_split(val_df, train_size=500, random_state=42)
    val_df = prepare_dataset(val_sub_df)

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
    client = OpenAI()

    # Save datasets to JSONL
    save_to_jsonl(train_df, "openai-gpt4o-training.jsonl")
    save_to_jsonl(val_df, "openai-gpt4o-valid.jsonl")

    # Load datasets and create fine-tuning files
    with open("openai-gpt4o-training.jsonl", 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f]
    
    print("Num examples:", len(training_data))
    print("First example:", training_data[0])

    # Fine-tune the model
    train_file = client.files.create(file=open("openai-gpt4o-training.jsonl", "rb"), purpose="fine-tune")
    valid_file = client.files.create(file=open("openai-gpt4o-valid.jsonl", "rb"), purpose="fine-tune")
    
    fine_tuning_job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=valid_file.id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 2}
    )

    print(fine_tuning_job)
    undersampled_df.to_csv("openai-training-data.csv", index=False)

if __name__ == "__main__":
    main()
