import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def dialog_segmentation(input_dialogue):
    dialog_seg_list = []
    dialog_seg_list_interval = []
    merged_dialog_list = []
    sen_list = input_dialogue.split('\n')
    word_count = 0
    dia_chunk = ''

    for sen in sen_list:
        num_of_words = len(sen.split(' '))
        new_word_count = word_count+num_of_words

        if new_word_count > 512:
            dia_chunk = dia_chunk + '\n' + sen
            dialog_seg_list.append(dia_chunk)
            dia_chunk = ''
            word_count = 0
        else:
            dia_chunk = dia_chunk + '\n' + sen
            word_count = new_word_count
    if dia_chunk!='':
        dialog_seg_list.append(dia_chunk)

    ### get interval dialogue
    word_count = 0
    dia_chunk = ''   
    start_word_count = 0
    for sen in sen_list:
        num_of_words = len(sen.split(' '))
        start_word_count = start_word_count+num_of_words
        if start_word_count>256:
            new_word_count = word_count+num_of_words
            if new_word_count > 512:
                dia_chunk = dia_chunk + '\n' + sen
                dialog_seg_list_interval.append(dia_chunk)
                dia_chunk = ''
                word_count = 0
            else:
                dia_chunk = dia_chunk + '\n' + sen
                word_count = new_word_count
        
    for i in range (len(dialog_seg_list)):
        merged_dialog_list.append(dialog_seg_list[i])
        try:
            merged_dialog_list.append(dialog_seg_list_interval[i])
        except Exception as e:
            pass

    return merged_dialog_list

def remove_duplicated_sentence(paragraph):
    sentences = paragraph.split('.')
    unique_sentences = []

    for sentence in sentences:
        if sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())

    unique_paragraph = '.'.join(unique_sentences)
    return unique_paragraph

model_id = "wanglab/task-a-flan-t5-large-run-2"

device = 'cuda'

model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
tok_id = model_id
tok = AutoTokenizer.from_pretrained(tok_id)
tok.model_max_length=1024

ori_model_id = "Falconsai/medical_summarization"
ori_model = AutoModelForSeq2SeqLM.from_pretrained(ori_model_id).to(device)
ori_tok = AutoTokenizer.from_pretrained(ori_model_id)
ori_tok.model_max_length=1024

df = pd.read_csv("./data/clinicalnlp_taskB_test1.csv")

outdir = './Results/'+os.path.basename(model_id)+'-'+os.path.basename(ori_model_id)+'_aci-taskB-test1/'
outdir_seg = './Results/'+os.path.basename(model_id)+'-'+os.path.basename(ori_model_id)+'_aci-taskB-test1_parts/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outdir_seg):
    os.makedirs(outdir_seg)

for index, row in df.iterrows():
    encounter_id = row['encounter_id']
    corpus_dataset = row['dataset']
    dialog = row['dialogue']
    dialog_seg_list = dialog_segmentation(dialog)
    collected_sum = ''
    for d_idx in range(len(dialog_seg_list)):
        the_dialog = dialog_seg_list[d_idx]
        text = "Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Dialogue:\n"+the_dialog    
        input_ids = tok(text, return_tensors="pt", padding=True).to(0)
        out = model.generate(**input_ids, max_new_tokens=512, do_sample=False)
        summary = tok.batch_decode(out, skip_special_tokens=True)[0]
        #print('sum:',summary)
        f = open(outdir_seg+str(corpus_dataset)+'-'+str(encounter_id)+'-'+str(d_idx)+'.txt', "w")
        f.write(summary)
        f.close()
        summary = summary.split('Section text:')[1]
        collected_sum = collected_sum + remove_duplicated_sentence(summary)
        
    ### final summary
    text = "Summarize:\n"+collected_sum    
    print('collection:',text)
    input_ids = ori_tok(text, return_tensors="pt", padding=True).to(0)
    out = ori_model.generate(**input_ids, max_new_tokens=512, do_sample=False)
    summary = ori_tok.batch_decode(out, skip_special_tokens=True)[0]
    print('ALL:', summary)
    f = open(outdir+str(corpus_dataset)+'-'+str(encounter_id)+'.txt', "w")
    f.write(summary)
    f.close()
