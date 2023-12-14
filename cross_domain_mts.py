import os
import json
import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm

med_word_list = open('./data/wordlist-medicalterms-en-master/wordlist.txt','r').read().splitlines()
med_word_list = [word.lower() for word in med_word_list]


def remove_pun_except_apostrophe(input_string):
    """
    remove punctuations (except for ' ) of the inupt string.
    """    
    translator = str.maketrans('', '', string.punctuation.replace("'", ""))
    output_string = input_string.translate(translator).replace('  ',' ')
    return output_string

def get_med_word(text):
    text = remove_pun_except_apostrophe(text)
    text_word = text.lower().split()
    matching_words = [word for word in text_word if word in med_word_list]
    return set(matching_words)

def count_mismatch_med_word(input_text, dia_med):
    input_med_word = get_med_word(input_text)
    return len(input_med_word - dia_med)

MTS_MAPPING = {
        'GENHX':'subjective',
        'FAM/SOCHX':'subjective',
        'PASTMEDICALHX':'subjective',
        'CC':'subjective',
        'PASTSURGICAL':'subjective',
        'ALLERGY':'subjective',
        'ROS':'subjective',
        'MEDICATIONS':'subjective',
        'ASSESSMENT':'assessment_and_plan',
        'EXAM':'objective',
        'DIAGNOSIS':'assessment_and_plan',
        'DISPOSITION':'assessment_and_plan',
        'PLAN':'assessment_and_plan',
        'EDCOURSE':'assessment_and_plan',
        'IMMUNIZATIONS':'subjective',
        'IMAGING':'objective',
        'GYNHX':'subjective',
        'PROCEDURES':'subjective',
        'OTHER_HISTORY':'subjective',
        'LABS':'objective'
}

def load_json(path):
    with open(path, 'r') as json_file:
        loaded_results = json.load(json_file)
    return loaded_results

def count_word(input_string):

    input_string = input_string.replace('.','').replace('-','').replace(':','').replace('(','').replace(')','').split(' ')
    words = [word for word in input_string if word!='']
    return len(words)

def process_t5_note(text):
    return text.split('Section text:')[0].split('Section header:')[-1], text.split('Section text:')[1]


rootdir_led = './Evaluation_scores/mts_Led_4096-1024'
rootdir_chatgpt = './Evaluation_scores/mts_openai'
rootdir_chatgpt_soap = './Evaluation_scores/mts_openai_soap'
rootdir_flant5 = './Results/mts_wanglab_flant5'
in_domain = load_json('./Evaluation_scores/mts-testset1-chat_wanglab_flant5')
df = pd.read_csv('./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')
dialogue_list = df['dialogue'].to_list()

rouge1 = {"Led-Subjective":[],"Led-Objective":[],"Led-Assessment_Plan":[],"Led-Full Note":[],'Flan-T5':[],'chatgpt':[], "Chat-Subjective":[],"Chat-Objective":[],"Chat-Assessment_Plan":[],"Chat-Full Note":[]}

bertscore_f1 = {"Led-Subjective":[],"Led-Objective":[],"Led-Assessment_Plan":[],"Led-Full Note":[],'Flan-T5':[],'chatgpt':[], "Chat-Subjective":[],"Chat-Objective":[],"Chat-Assessment_Plan":[],"Chat-Full Note":[]}

bleurt = {"Led-Subjective":[],"Led-Objective":[],"Led-Assessment_Plan":[],"Led-Full Note":[],'Flan-T5':[],'chatgpt':[], "Chat-Subjective":[],"Chat-Objective":[],"Chat-Assessment_Plan":[],"Chat-Full Note":[]}

length = {"Led-Subjective":[],"Led-Objective":[],"Led-Assessment_Plan":[],"Led-Full Note":[],'reference':[],'chatgpt':[],'Flan-T5':[], "Chat-Subjective":[],"Chat-Objective":[],"Chat-Assessment_Plan":[],"Chat-Full Note":[]}

med_words = {"Led-Subjective":[],"Led-Objective":[],"Led-Assessment_Plan":[],"Led-Full Note":[],'reference':[],'chatgpt':[],'Flan-T5':[], "Chat-Subjective":[],"Chat-Objective":[],"Chat-Assessment_Plan":[],"Chat-Full Note":[]}

soap_idx = {"subjective":[], "objective":[], "assessment_and_plan":[]}

for idx in tqdm(range(200)):
    
    data = load_json(os.path.join(rootdir_led, str(idx)+'.json'))
    section_header = data['section_header']
    mapped_header = MTS_MAPPING[section_header]
    soap_idx[mapped_header].append(idx)

    gpt_result = load_json(os.path.join(rootdir_chatgpt, str(idx)+'.json'))
    gpt_soap_result = load_json(os.path.join(rootdir_chatgpt_soap, str(idx)+'.json'))

    rouge1["Led-Subjective"].append(data['rouge1'][1])
    rouge1["Led-Objective"].append(data['rouge1'][2])
    rouge1["Led-Assessment_Plan"].append(data['rouge1'][3])
    rouge1["Led-Full Note"].append(data['rouge1'][0])
    rouge1["Flan-T5"].append(in_domain['rouge1'][idx])
    rouge1["chatgpt"].append(gpt_result['rouge1'][0])
    rouge1["Chat-Subjective"].append(gpt_soap_result['rouge1'][1])
    rouge1["Chat-Objective"].append(gpt_soap_result['rouge1'][2])
    rouge1["Chat-Assessment_Plan"].append(gpt_soap_result['rouge1'][3])
    rouge1["Chat-Full Note"].append(gpt_soap_result['rouge1'][0])


    bertscore_f1["Led-Subjective"].append(data['bertscore_f1'][1])
    bertscore_f1["Led-Objective"].append(data['bertscore_f1'][2])
    bertscore_f1["Led-Assessment_Plan"].append(data['bertscore_f1'][3])
    bertscore_f1["Led-Full Note"].append(data['bertscore_f1'][0])
    bertscore_f1["Flan-T5"].append(in_domain['bertscore_f1'][idx])
    bertscore_f1["chatgpt"].append(gpt_result['bertscore_f1'][0])
    bertscore_f1["Chat-Subjective"].append(gpt_soap_result['bertscore_f1'][1])
    bertscore_f1["Chat-Objective"].append(gpt_soap_result['bertscore_f1'][2])
    bertscore_f1["Chat-Assessment_Plan"].append(gpt_soap_result['bertscore_f1'][3])
    bertscore_f1["Chat-Full Note"].append(gpt_soap_result['bertscore_f1'][0])
   

    bleurt["Led-Subjective"].append(data['bleurt'][1])
    bleurt["Led-Objective"].append(data['bleurt'][2])
    bleurt["Led-Assessment_Plan"].append(data['bleurt'][3])
    bleurt["Led-Full Note"].append(data['bleurt'][0])
    bleurt["Flan-T5"].append(in_domain['bleurt'][idx])
    bleurt["chatgpt"].append(gpt_result['bleurt'][0])
    bleurt["Chat-Subjective"].append(gpt_soap_result['bleurt'][1])
    bleurt["Chat-Objective"].append(gpt_soap_result['bleurt'][2])
    bleurt["Chat-Assessment_Plan"].append(gpt_soap_result['bleurt'][3])
    bleurt["Chat-Full Note"].append(gpt_soap_result['bleurt'][0])


    t5_results = open(os.path.join(rootdir_flant5, str(idx)+'.txt'),'r').read()
    t5_head, t5_note = process_t5_note(t5_results)
    length["Led-Subjective"].append(count_word(data['pred_note'][1]))
    length["Led-Objective"].append(count_word(data['pred_note'][2]))
    length["Led-Assessment_Plan"].append(count_word(data['pred_note'][3]))
    length["Led-Full Note"].append(count_word(data['pred_note'][0]))
    length['reference'].append(count_word(data['gt_note'][0]))
    length['chatgpt'].append(count_word(gpt_result['pred_note'][0]))
    length['Flan-T5'].append(count_word(t5_note))
    length["Chat-Subjective"].append(count_word(gpt_soap_result['pred_note'][1]))
    length["Chat-Objective"].append(count_word(gpt_soap_result['pred_note'][2]))
    length["Chat-Assessment_Plan"].append(count_word(gpt_soap_result['pred_note'][3]))
    length["Chat-Full Note"].append(count_word(gpt_soap_result['pred_note'][0]))

    
    dia_med = get_med_word(dialogue_list[idx])
    med_words["Led-Subjective"].append(count_mismatch_med_word(data['pred_note'][1], dia_med))
    med_words["Led-Objective"].append(count_mismatch_med_word(data['pred_note'][2], dia_med))
    med_words["Led-Assessment_Plan"].append(count_mismatch_med_word(data['pred_note'][3], dia_med))
    med_words["Led-Full Note"].append(count_mismatch_med_word(data['pred_note'][0], dia_med))
    med_words['reference'].append(count_mismatch_med_word(data['gt_note'][0], dia_med))
    med_words['chatgpt'].append(count_mismatch_med_word(gpt_result['pred_note'][0], dia_med))
    med_words['Flan-T5'].append(count_mismatch_med_word(t5_note, dia_med))
    med_words["Chat-Subjective"].append(count_mismatch_med_word(gpt_soap_result['pred_note'][1], dia_med))
    med_words["Chat-Objective"].append(count_mismatch_med_word(gpt_soap_result['pred_note'][2], dia_med))
    med_words["Chat-Assessment_Plan"].append(count_mismatch_med_word(gpt_soap_result['pred_note'][3], dia_med))
    med_words["Chat-Full Note"].append(count_mismatch_med_word(gpt_soap_result['pred_note'][0], dia_med))
    
for key, value in rouge1.items():
    rouge1[key] = np.asarray(rouge1[key])
    bertscore_f1[key] = np.asarray(bertscore_f1[key])
    bleurt[key] = np.asarray(bleurt[key])

print('rouge1')
for key, value in rouge1.items():
    print(key)
    print(round(np.mean(value),4))
    for soap_key, the_idx in soap_idx.items():
        print(soap_key)
        print(round(np.mean(value[the_idx]),4))
    print("----")

# print('bertscore_f1')
# for key, value in bertscore_f1.items():
#     print(key)
#     print(round(np.mean(value),4))
#     for soap_key, the_idx in soap_idx.items():
#         print(soap_key)
#         print(round(np.mean(value[the_idx]),4))
#     print("----")

# print('bleurt')
# for key, value in bleurt.items():
#     print(key)
#     print(round(np.mean(value),4))
#     for soap_key, the_idx in soap_idx.items():
#         print(soap_key)
#         print(round(np.mean(value[the_idx]),4))
#     print("----")

for key, value in length.items():
    length[key] = np.asarray(length[key])

print("length")
print('-----')  
for key, value in length.items():
    print(key)
    print('ALL samples:', round(np.mean(value),4))
    for soap_key, the_idx in soap_idx.items():
        print(soap_key)
        print(round(np.mean(value[the_idx]),4)) 
    print('___')

print('medical words -----')  
for key, value in med_words.items():
    med_words[key] = np.asarray(med_words[key])

for key, value in med_words.items():
    print(key)
    print('ALL samples:', round(np.mean(value),4))
    for soap_key, the_idx in soap_idx.items():
        print(soap_key)
        print(round(np.mean(value[the_idx]),4)) 
    print('___')