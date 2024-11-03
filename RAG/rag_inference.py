import pandas as pd
import json
from tqdm import tqdm
import re
# from vllm import LLM, SamplingParams
# from sentence_transformers import SentenceTransformer, util
# import huggingface_hub

def login_to_huggingface():
    """Logs in to Hugging Face Hub."""
    huggingface_hub.login()

def initialize_llm_model(model_name="meta-llama/Llama-3.1-8B-Instruct", max_model_len=2048):
    """
    Initializes the LLM model.

    Args:
        model_name (str): The model name on Hugging Face Hub.
        max_model_len (int): Maximum token length for the model.

    Returns:
        LLM: The initialized LLM model.
    """
    return LLM(model=model_name, max_model_len=max_model_len)

def read_test_data(file_path):
    """
    Reads the test data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: Loaded test data.
    """
    return pd.read_json(file_path, lines=True)

def define_sampling_params(temperature=0.1, max_tokens=128):
    """
    Defines the sampling parameters for the LLM.

    Args:
        temperature (float): Sampling temperature for LLM.
        max_tokens (int): Maximum tokens in each generated response.

    Returns:
        SamplingParams: The sampling parameters.
    """
    return SamplingParams(temperature=temperature, max_tokens=max_tokens)

def build_conversation_prompt(row):
    """
    Builds a conversation prompt from a row of data.

    Args:
        row (str): The input text for the conversation.

    Returns:
        list: A conversation dictionary list.
    """
    try:
        return [{"role": "human", "content": row}]
    except Exception as e:
        print(f"Error building prompt: {e}")
        return [{"role": "human", "content": "unknown unknown"}]

def extract_labels_from_context(prompt):
    match = re.search(r'Context:\s*(\[.*?\])', prompt)
    context_str = match.group(1)    
    context_list = json.loads(context_str.replace("'", "\""))
    labels = []
    for data in context_list:
        labels.append(data["labels"])
    return labels

def process_llm_output(index, output, test_data_prompt):
    """
    Processes the output from the LLM model.

    Args:
        index (int): Index of the current output.
        output (str): The output text from the LLM model.
        test_data_prompt (pd.DataFrame): The test data prompts.

    Returns:
        dict: A dictionary containing the processed category data.
    """
    json_str = output.outputs[0].text
    top_rank_labels = extract_labels_from_context(test_data_prompt["prompt"][index])
    if json_str not in top_rank_labels:
        json_str = top_rank_labels[0]

    category_split = json_str.split("__")
    return {
        "indoml_id": index,
        "supergroup": category_split[0],
        "group": category_split[1],
        "module": category_split[2],
        "brand": category_split[3]
    }

def generate_predictions(test_data_prompt, llm_model, sampling_params):
    """
    Generates predictions using the LLM model and processes each output.

    Args:
        test_data_prompt (pd.DataFrame): The test data prompts.
        llm_model (LLM): The initialized LLM model.
        sampling_params (SamplingParams): The sampling parameters.

    Returns:
        list: List of processed output dictionaries.
    """
    conversation_prompts = [build_conversation_prompt(row["prompt"]) for _, row in test_data_prompt.iterrows()]
    outputs = llm_model.chat(conversation_prompts, sampling_params=sampling_params, use_tqdm=True)

    return [process_llm_output(index, output, test_data_prompt) for index, output in enumerate(outputs)]

def save_predictions(predictions, output_file):
    """
    Saves the predictions to a file in JSON format.

    Args:
        predictions (list): List of prediction dictionaries.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as fp:
        for prediction in predictions:
            fp.write(json.dumps(prediction) + "\n")

def main():
    # login_to_huggingface()
    
    # Initialize model and sampling parameters
    llm_model = initialize_llm_model()
    sampling_params = define_sampling_params()

    # Load and process test data
    test_data_prompt = read_test_data(file_path="test_data.prompts")
    
    predictions = generate_predictions(test_data_prompt, llm_model, sampling_params)

    # Save the generated predictions
    save_predictions(predictions,"test_data.predict")

if __name__ == "__main__":
    main()
