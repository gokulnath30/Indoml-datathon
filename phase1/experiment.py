#%%

import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#%%
train_data = pd.read_json("input_data/attribute_train.data", lines=True)
train_solution = pd.read_json("input_data/attribute_train.solution", lines=True)

#%%
train_data = train_data[:1000]
train_solution = train_solution[:1000]


merged_data = pd.concat([train_data, train_solution.drop("indoml_id", axis=1)], axis=1)
drop_na = merged_data.dropna(subset=['details_Manufacturer'])


x_train = drop_na[["indoml_id",'title', 'store', 'details_Manufacturer']]
y_train = drop_na[["indoml_id","details_Brand","L0_category", "L1_category", "L2_category", "L3_category", "L4_category"]]

y_train.head()

#%%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

x_columns = ['title', 'store', 'details_Manufacturer']
for col in x_columns:
    x_train.loc[:, col] = x_train[col].str.lower().str.replace(f'[{string.punctuation}]', '', regex=True)
    x_train.loc[:, col] = x_train[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

x_train.head()

#%%

x_train.loc[:,'combined_text'] = x_train['title'] + ' ' + x_train['store'] + ' ' + x_train['details_Manufacturer']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the max_features as needed

# # Fit and transform the data
tfidf_vectors = tfidf_vectorizer.fit_transform(x_train['combined_text'])


# Convert to a DataFrame for easier handling (optional)
tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# # View the TF-IDF DataFrame
print(tfidf_vectors.shape)
# print(y_train['L0_category'].shape)
#%%
# X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, y_train['L0_category'], test_size=0.2, random_state=42, stratify=y_train['L0_category'])

model = LogisticRegression(max_iter=1000)
model.fit(tfidf_vectors, y_train)


