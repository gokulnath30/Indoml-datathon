import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import json
import wandb
from tqdm import tqdm
from datasets import Dataset
import huggingface_hub
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

def load_and_merge_data(features_path, labels_path):
    """Load and merge training features and labels on 'indoml_id'."""
    train_features_df = pd.read_json(features_path, lines=True)
    train_labels_df = pd.read_json(labels_path, lines=True)
    print("Data shapes - Features:", train_features_df.shape, "Labels:", train_labels_df.shape)
    train_df = pd.merge(train_features_df, train_labels_df, on="indoml_id")
    return train_df

def create_labels(train_df):
    """Create combined labels and identify infrequent labels."""
    train_df["label"] = train_df['supergroup'] + " __" + train_df['group'] + " __" + train_df['module'] + " __" + train_df['brand']
    count_df = train_df["label"].value_counts().reset_index()
    print("Label count dataframe shape:", count_df.shape)
    return train_df, count_df

def filter_single_label_instances(train_df, count_df, min_count=5):
    """Remove labels with fewer than min_count instances and keep them separately."""
    single_label = count_df[count_df["count"] < min_count]["label"].tolist()
    single_label_df = train_df[train_df["label"].isin(single_label)]
    train_df = train_df[~train_df["label"].isin(single_label)]
    print("Filtered train_df shape:", train_df.shape)
    return train_df, single_label_df

def undersample_data(train_df, single_label_df):
    """Undersample the dataset, keeping one instance per remaining label."""
    df_min_samples = train_df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True)
    undersampled_df = pd.concat([single_label_df, df_min_samples], axis=0)
    print("Undersampled df shape:", undersampled_df.shape)
    train_df = train_df[~train_df["indoml_id"].isin(df_min_samples["indoml_id"])]
    return train_df, undersampled_df

def stratified_split(train_df, undersampled_df, train_size=80000):
    """Perform a stratified split on remaining data and concatenate with undersampled data."""
    remaining_df, val_df = train_test_split(
        train_df, train_size=train_size, stratify=train_df['label'], random_state=42
    )
    undersampled_df = pd.concat([undersampled_df, remaining_df], axis=0)
    print("Final undersampled df shape:", undersampled_df.shape)
    return undersampled_df, val_df

def validate_missing_labels(count_df, undersampled_df):
    """Check for any missing labels after undersampling."""
    missing_labels = set(count_df["label"]) - set(undersampled_df["label"])
    print(f"Number of missing labels: {len(missing_labels)}")
    return missing_labels

def create_conversation_data(undersampled_df):
    """Convert dataset into a format suitable for conversation-based models."""
    lines_list = []
    max_length = 0
    for _, row in tqdm(undersampled_df.iterrows()):
        desc, retailer, price = row["description"], row["retailer"], row["price"]
        human_val = json.dumps({"description": desc, "retailer": retailer, "price": price})
        gpt_val = json.dumps({"supergroup": row["supergroup"], "group": row["group"], "module": row["module"], "brand": row["brand"]})
        lines_list.append([{'from': 'human', 'value': human_val}, {'from': 'gpt', 'value': gpt_val}])
        max_length = max(max_length, len(human_val) + len(gpt_val))
    print("Max length of conversations:", max_length)
    return lines_list

def create_huggingface_dataset(lines_list):
    """Create and return a Hugging Face dataset from conversation data."""
    data_dict = {'conversations': lines_list}
    dataset = Dataset.from_dict(data_dict)
    print("Sample dataset entries:", dataset[0])
    return dataset

def login_huggingface():
    """Login to Hugging Face Hub."""
    huggingface_hub.login()
    print("Logged into Hugging Face.")

def setup_and_configure_model(hf_model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"):
    """Set up and configure FastLanguageModel with PEFT configuration."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_name,
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        attn_implementation="flash_attention_2"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    )

    return model, tokenizer

def format_prompts(dataset, tokenizer):
    """Format the dataset conversations into prompts for the model."""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    return formatted_dataset

def train_model(model, tokenizer, dataset):
    """Train the model using the SFTTrainer."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
            run_name="test-llama-training"
        )
    )

    trainer_stats = trainer.train()
    return trainer_stats

def push_model_to_hub(model, tokenizer, hf_id="srinathmkce/indoml_100k_llama_updated_dataset"):
    """Push the trained model to the Hugging Face Hub."""
    model.push_to_hub_merged(hf_id, tokenizer, save_method="merged_16bit")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLaMA Training Script")
    parser.add_argument("--features_path", type=str, default="training_data/train.features", help="Path to training features JSONL file")
    parser.add_argument("--labels_path", type=str, default="training_data/train.labels", help="Path to training labels JSONL file")
    parser.add_argument("--push_to_hub", type=bool, default=False, help="Push the trained model to the Hugging Face Hub")

    wandb.login()
    
    args = parser.parse_args()

    # Access the parsed arguments
    features_path = args.features_path
    labels_path = args.labels_path
    
    # Execution sequence
    train_df = load_and_merge_data(features_path, labels_path)
    train_df, count_df = create_labels(train_df)
    train_df, single_label_df = filter_single_label_instances(train_df, count_df)
    train_df, undersampled_df = undersample_data(train_df, single_label_df)
    undersampled_df, val_df = stratified_split(train_df, undersampled_df)
    missing_labels = validate_missing_labels(count_df, undersampled_df)
    
    lines_list = create_conversation_data(undersampled_df)
    dataset = create_huggingface_dataset(lines_list)
    # login_huggingface()
    print("Sample Conversations: ")
    print(dataset["conversations"][:5])
    
    model, tokenizer = setup_and_configure_model()
    dataset = format_prompts(dataset, tokenizer)
    print("Sample Prompts: ")

    print(dataset["text"][:5])
    trainer_stats = train_model(model, tokenizer, dataset)
    
    if args.push_to_hub:
        push_model_to_hub(model, tokenizer)

