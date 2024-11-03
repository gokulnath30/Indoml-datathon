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
python openai-training.py
```
Provide the OPENAI_KEY when prompted

Wait for the finetuning job to finish

### Inference

To run the inference using batch api , generate the files

```
python openai-inference.py
```
Move to https://platform.openai.com/batches/ and upload batches one by one

### Extract OpenAI outputs

to extract the output, download all the result files from openai and place it in a folder.
Ensure to change the base directory

Now run the following command

```
python openai_extract_output.py
```


### Postprocessing

To improve the results, run the following postprocessing command

Ensure the *.predict file exists in the folder and update the name of the predict file in openai_postprocessing.py file. 

```
python openai_postprocessing.py
```

