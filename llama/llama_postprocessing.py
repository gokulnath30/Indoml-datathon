import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def load_data(predict_path, train_path):
    predict_df = pd.read_json(predict_path, lines=True)
    train_label_df = pd.read_json(train_path, lines=True)
    
    predict_df["label"] = (
        predict_df['supergroup'] + " __" + 
        predict_df['group'] + " __" + 
        predict_df['module'] + " __" + 
        predict_df['brand']
    )
    train_label_df["label"] = (
        train_label_df['supergroup'] + " __" + 
        train_label_df['group'] + " __" + 
        train_label_df['module'] + " __" + 
        train_label_df['brand']
    )
    
    return predict_df, train_label_df


def initialize_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')


def encode_unique_labels(train_label_df, model):
    embeddings = {}
    embeddings["supergroup"] = model.encode(train_label_df["supergroup"].unique(), show_progress_bar=True, convert_to_tensor=True)
    embeddings["group"] = model.encode(train_label_df["group"].unique(), show_progress_bar=True, convert_to_tensor=True)
    embeddings["module"] = model.encode(train_label_df["module"].unique(), show_progress_bar=True, convert_to_tensor=True)
    embeddings["label"] = model.encode(train_label_df["label"].unique(), show_progress_bar=True, convert_to_tensor=True)
    
    return embeddings


def get_closest_category(query, embeddings, strings, model):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)
    closest_match_idx = similarities.argmax().item()
    return strings[closest_match_idx]


def correct_mismatched_entries(predict_df, train_label_df, category, embeddings, model):
    unique_values = train_label_df[category].unique()
    incorrect_df = predict_df[~predict_df[category].isin(unique_values)]
    print("Incorrect Values: ", incorrect_df.shape)
    
    for index, row in tqdm(incorrect_df.iterrows(), total=incorrect_df.shape[0]):
        indoml_id = row["indoml_id"]
        incorrect_value = row[category]
        if category == "brand":
            corrected_value = get_closest_category(incorrect_value, embeddings["label"], unique_values, model)
        else:
            corrected_value = get_closest_category(incorrect_value, embeddings[category], unique_values, model)
            
        predict_df.loc[predict_df["indoml_id"] == indoml_id, category] = corrected_value


def update_labels(predict_df):
    predict_df["label"] = (
        predict_df['supergroup'] + " __" + 
        predict_df['group'] + " __" + 
        predict_df['module'] + " __" + 
        predict_df['brand']
    )


def save_predictions(predict_df, output_path):
    category_list = [
        {
            "indoml_id": row["indoml_id"],
            "supergroup": row["supergroup"],
            "group": row["group"],
            "module": row["module"],
            "brand": row["brand"]
        }
        for _, row in predict_df.iterrows()
    ]
    with open(output_path, "w") as fp:
        for row in category_list:
            fp.write(json.dumps(row) + "\n")


def main():
    predict_path = "test_gradientgurus7.predict"
    train_path = "training_data/train.labels"
    output_path = "test_gradientgurus7_postprocessed.predict"

    # Load data
    predict_df, train_label_df = load_data(predict_path, train_path)

    # Initialize model and encode unique categories
    model = initialize_model()
    embeddings = encode_unique_labels(train_label_df, model)

    # Correct mismatches for each category
    for category in ["supergroup", "group", "module", "brand"]:
        correct_mismatched_entries(predict_df, train_label_df, category, embeddings, model)
    
    # Update labels and save predictions
    update_labels(predict_df)
    save_predictions(predict_df, output_path)


if __name__ == "__main__":
    main()

