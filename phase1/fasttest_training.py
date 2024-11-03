# -*- coding: utf-8 -*-
"""IndoML_fasttext.py - Local Version"""

import os
import string
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import fasttext

# Download NLTK data for the first time running locally
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define the paths for the dataset and fastText model locally
DATA_DIR = "data/"
MODEL_DIR = "models/"
TRAIN_FILE = os.path.join(MODEL_DIR, "indoml_try1.train")
MODEL_OUTPUT = os.path.join(MODEL_DIR, "try1_model")

# Ensure necessary directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load datasets (Ensure files are placed correctly in your local directory)
x_train_df = pd.read_json(os.path.join(DATA_DIR, "attribute_train.data"), lines=True)
y_train_df = pd.read_json(os.path.join(DATA_DIR, "attribute_train.solution"), lines=True)
x_val_df = pd.read_json(os.path.join(DATA_DIR, "attribute_val.data"), lines=True)
y_val_df = pd.read_json(os.path.join(DATA_DIR, "attribute_val.solution"), lines=True)
x_test_df = pd.read_json(os.path.join(DATA_DIR, "attribute_test.data"), lines=True)

# Merge train and validation datasets
x_train_df = pd.concat([x_train_df, x_val_df], axis=0)
y_train_df = pd.concat([y_train_df, y_val_df], axis=0)

# Print shapes of datasets for verification
print(f"Train data shape: {x_train_df.shape}, Train labels shape: {y_train_df.shape}")
print(f"Test data shape: {x_test_df.shape}")

# Preprocess text features
def preprocess_features(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Data preprocessing function
def preprocess(df):
    df.fillna("None", inplace=True)
    
    # Create features by combining relevant columns
    df["features"] = ("title: " + df["title"] + " store: " + df["store"] + " manufacturer: " + df["details_Manufacturer"]
    ).str.lower()
    
    # Remove punctuation, numbers, and apply feature tokenization
    df['features'] = df['features'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    df['features'] = df['features'].str.replace('\d+', '', regex=True)
    df['features'] = df['features'].apply(preprocess_features)
    
    # Prepare labels for fastText
    df["details_Brand"] = df["details_Brand"].str.replace(' ', '%%')
    categories = ["L0_category", "L1_category", "L2_category", "L3_category", "L4_category"]
    for category in categories:
        df[category] = df[category].str.replace(' ', '%%')
    
    df["label"] = '__label__' + df["details_Brand"] + '__' + '__'.join(df[cat] for cat in categories)
    df["training_row"] = df["label"] + ' ' + df["features"]
    
    return df

# Sort data and merge train features and labels
x_train_df.sort_values(by="indoml_id", ascending=True, inplace=True)
y_train_df.sort_values(by="indoml_id", ascending=True, inplace=True)
train_df = pd.concat([x_train_df, y_train_df.drop("indoml_id", axis=1)], axis=1)

# Apply preprocessing
train_df = preprocess(train_df)

# Save training data to file
training_rows = train_df["training_row"].tolist()
with open(TRAIN_FILE, "w") as fp:
    for row in tqdm(training_rows):
        fp.write(row + "\n")

# Train the fastText model
model = fasttext.train_supervised(
    input=TRAIN_FILE, 
    epoch=15, 
    lr=1.0, 
    wordNgrams=2
)

# Save the trained model
model.save_model(MODEL_OUTPUT + ".bin")

# Optional: Test the model with some predictions
print(model.predict("Sample input text for prediction"))
