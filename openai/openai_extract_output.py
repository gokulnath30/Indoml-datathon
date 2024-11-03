import pandas as pd
import os
import json
from tqdm import tqdm

# Set the base directory for the outputs
base_dir = r"openai/outputs"

# List all files in the directory
files = os.listdir(base_dir)

# Initialize a list to hold DataFrames
dfs = []
for file in files:
    full_path = os.path.join(base_dir, file)
    print(full_path)
    
    # Read each JSON file into a DataFrame
    df = pd.read_json(full_path, lines=True)
    print(df.shape)
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)
print(f"Merged DataFrame shape: {merged_df.shape}")

# Sort the merged DataFrame by 'custom_id'
merged_df.sort_values(by="custom_id", ascending=True, inplace=True)

# Initialize an empty list to store category dictionaries
category_list = []
count = 0

# Extract categories from the 'response' column
for index, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
    id = int(row["custom_id"])
    resp = row["response"]["body"]["choices"][0]["message"]["content"]
    
    try:
        resp_split = resp.split("__")
        length = len(resp_split)
        
        # Check the length of the split response and extract values
        if length == 4:
            supergroup, group, module, brand = [part.strip() for part in resp_split]
        elif length > 4:
            supergroup, group, module, brand = resp_split[0].strip(), resp_split[1].strip(), resp_split[2].strip(), resp_split[3].strip()
        else:
            print(id, resp)
            count += 1
            continue
           
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue

    # Create a dictionary for the category and append it to the list
    category_dict = {
        "indoml_id": id,
        "supergroup": supergroup,
        "group": group,
        "module": module,
        "brand": brand
    }
    category_list.append(category_dict)

# Create a DataFrame for the submission
submit_df = pd.DataFrame(category_list, columns=["indoml_id", "supergroup", "group", "module", "brand"])
print(f"Submission DataFrame shape: {submit_df.shape}")

# Sort the submission DataFrame by 'indoml_id'
submit_df.sort_values(by=["indoml_id"], ascending=False, inplace=True)

# Save predictions to JSON file
output_file = "test_openai_with_price_2epoch_attempt2.predict"
with open(output_file, "w") as fp:
    for row in category_list:
        fp.write(json.dumps(row) + "\n")

print(f"Predictions saved to {output_file}")
