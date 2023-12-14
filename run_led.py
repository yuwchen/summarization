import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModel, Trainer, TrainingArguments
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM


model_id = "wanglab/task-b-led-large-16384-pubmed-run-3"
device = 'cuda'
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
tok_id=model_id
tok = AutoTokenizer.from_pretrained(tok_id)
tok.model_max_length=4096

df = pd.read_csv("./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv")

outdir = './Results/'+model_id+'-4096-1024_mts-testset1-chat/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

for index, row in df.iterrows():
    id = row['ID']
    dialog = row['dialogue']
    sequence = (dialog)
    text="Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. You should first predict the most relevant clinical note section header and then summarize the dialogue. Dialogue:\n"+dialog
    input_ids = tok(text, return_tensors="pt", padding=True).to(0)
    out = model.generate(**input_ids, max_new_tokens=1024, do_sample=False)
    summary = tok.batch_decode(out, skip_special_tokens=True)[0]
    print('sum:',summary)
    f = open(outdir+str(id)+'.txt', "w")
    f.write(summary)
    f.close()
