### Unzip the data

Run the following command

```
apt update && apt install -y unzip zip
unzip phase2_input_data.zip
```
### Prerequisites

Install the required packages

```
pip install -r requirements.txt
```

### Training

To start the training, run the following command

```
python llama_training.py
```
The training files and pushing the model to huggingface hub can be controlled via parser arguments. 

### Inference

Update VLLm to the latest version
```
pip install -qU vllm
```

Run the inference using the following command

```
python llama_inference.py
```

### Postprocessing

To improve the results, run the following postprocessing command

Ensure the *.predict file exists in the folder and update the name of the predict file in llama_postprocessing.py file. 

```
python llama_postprocessing.py
```