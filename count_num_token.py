import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the Flax T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
#tokenizer.model_max_length=5000



def get_token_length(paragraph):
    tokens = tokenizer(paragraph, return_tensors="pt")
    # Get the number of tokens
    num_tokens = tokens.input_ids.shape[1]
    return num_tokens

def conut_mts(path):
    
    print(path)
    df = pd.read_csv(path)
    c_ref = []
    c_dia = []
    for index, row in df.iterrows():
        reference = row['section_text']
        dialog = row['dialogue']
        c_ref.append(get_token_length(reference))
        c_dia.append(get_token_length(dialog))
        if get_token_length(reference)>1000:
            print(row)
    print('reference:', round(np.mean(c_ref),2),'/', np.max(c_ref),'/', round(np.std(c_ref),2))
    print('dialog:', round(np.mean(c_dia),2),'/', np.max(c_dia), '/', round(np.std(c_dia),2))

def conut_aci(path):
    
    print(path)
    df = pd.read_csv(path)
    c_ref = []
    c_dia = []
    for index, row in df.iterrows():
        reference = row['note']
        dialog = row['dialogue']
        c_ref.append(get_token_length(reference))
        c_dia.append(get_token_length(dialog))
    
    print('reference:', round(np.mean(c_ref),2),'/', np.max(c_ref), '/', round(np.std(c_ref),2))
    print('dialog:', round(np.mean(c_dia),2),'/', np.max(c_dia), '/', round(np.std(c_dia),2))

conut_mts('./data/mts-dialog/MTS-Dialog-TrainingSet.csv')
conut_mts('./data/mts-dialog/MTS-Dialog-ValidationSet.csv')
conut_mts('./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

conut_aci('./data/aci-bench/aci-train.csv')
conut_aci('./data/aci-bench/aci-valid.csv')
conut_aci('./data/aci-bench/clinicalnlp_taskB_test1.csv')



