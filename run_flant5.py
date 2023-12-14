import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "wanglab/task-a-flan-t5-large-run-2"

device = 'cuda'

model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
tok_id = model_id
tok = AutoTokenizer.from_pretrained(tok_id)
tok.model_max_length=4096

df = pd.read_csv("./data/aci-bench/clinicalnlp_taskB_test1.csv")

outdir = './Results/'+os.path.basename(model_id)+'_4096-512_aci-taskB-test1/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for index, row in df.iterrows():
    encounter_id = row['encounter_id']
    corpus_dataset = row['dataset']
    dialog = row['dialogue']
    sequence = (dialog)
    text = "Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Dialogue:\n"+dialog    
    input_ids = tok(text, return_tensors="pt", padding=True).to(0)
    out = model.generate(**input_ids, max_new_tokens=512, do_sample=False)
    summary = tok.batch_decode(out, skip_special_tokens=True)[0]
    print('sum:',summary)
    f = open(outdir+str(corpus_dataset)+'-'+str(encounter_id)+'.txt', "w")
    f.write(summary)
    f.close()