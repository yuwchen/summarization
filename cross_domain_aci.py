
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

def load_json(path):
    with open(path, 'r') as json_file:
        loaded_results = json.load(json_file)
    return loaded_results

def count_word(input_string):

    input_string = input_string.replace('.','').replace('-','').replace(':','').replace('(','').replace(')','').split(' ')
    words = [word for word in input_string if word!='']
    return len(words)


chatgpt_path = "./Results//test1ChatGPT_.csv"
gpt4_path = "./Results/test1GPT-4_.csv"
led = './Evaluation_scores/aci_Led'
flant5 = './Evaluation_scores/aci_t5_sep_med'
chatgpt = './Evaluation_scores/aci_chatgpt'
gpt4 = './Evaluation_scores/aci_gpt4'
gpt35 = './Evaluation_scores/aci_chatgpt_general'

rouge1 = {'led_full':[], 'led_S':[], 'led_O':[], 'led_A':[], 't5_full':[], 't5_S':[],'t5_O':[],'t5_A':[], 'chat_full':[],'chat_S':[],'chat_O':[],'chat_A':[], 'gpt4_full':[],'gpt4_S':[],'gpt4_O':[],'gpt4_A':[], 'gpt35_full':[],'gpt35_S':[],'gpt35_O':[],'gpt35_A':[] }
bertscore_f1 = {'led_full':[], 'led_S':[], 'led_O':[], 'led_A':[], 't5_full':[], 't5_S':[],'t5_O':[],'t5_A':[], 'chat_full':[],'chat_S':[],'chat_O':[],'chat_A':[], 'gpt4_full':[],'gpt4_S':[],'gpt4_O':[],'gpt4_A':[], 'gpt35_full':[],'gpt35_S':[],'gpt35_O':[],'gpt35_A':[] }
bleurt = {'led_full':[], 'led_S':[], 'led_O':[], 'led_A':[], 't5_full':[], 't5_S':[],'t5_O':[],'t5_A':[], 'chat_full':[],'chat_S':[],'chat_O':[],'chat_A':[], 'gpt4_full':[],'gpt4_S':[],'gpt4_O':[],'gpt4_A':[], 'gpt35_full':[],'gpt35_S':[],'gpt35_O':[],'gpt35_A':[] }

chat_df = pd.read_csv(chatgpt_path)
for index, row in chat_df.iterrows():
    
    idx = row['encounter_id']
    if int(idx[-3:])<=97:
        filename = 'virtassist-'+idx
    elif int(idx[-3:])<=105 and int(idx[-3:])>97:
        filename = 'virtscribe-'+idx
    else:
        filename = 'aci-'+idx
    
    led_result = load_json(os.path.join(led, filename+'.json'))

    rouge1['led_full'].append(led_result['rouge1'][0])
    rouge1['led_S'].append(led_result['rouge1'][1])
    rouge1['led_O'].append(led_result['rouge1'][2])
    rouge1['led_A'].append(led_result['rouge1'][3])

    bertscore_f1['led_full'].append(led_result['bertscore_f1'][0])
    bertscore_f1['led_S'].append(led_result['bertscore_f1'][1])
    bertscore_f1['led_O'].append(led_result['bertscore_f1'][2])
    bertscore_f1['led_A'].append(led_result['bertscore_f1'][3])

    bleurt['led_full'].append(led_result['bleurt'][0])
    bleurt['led_S'].append(led_result['bleurt'][1])
    bleurt['led_O'].append(led_result['bleurt'][2])
    bleurt['led_A'].append(led_result['bleurt'][3])

    t5_result = load_json(os.path.join(flant5, filename+'.json'))
    rouge1['t5_full'].append(t5_result['rouge1'][0])
    rouge1['t5_S'].append(t5_result['rouge1'][1])
    rouge1['t5_O'].append(t5_result['rouge1'][2])
    rouge1['t5_A'].append(t5_result['rouge1'][3])

    bertscore_f1['t5_full'].append(t5_result['bertscore_f1'][0])
    bertscore_f1['t5_S'].append(t5_result['bertscore_f1'][1])
    bertscore_f1['t5_O'].append(t5_result['bertscore_f1'][2])
    bertscore_f1['t5_A'].append(t5_result['bertscore_f1'][3])

    bleurt['t5_full'].append(t5_result['bleurt'][0])
    bleurt['t5_S'].append(t5_result['bleurt'][1])
    bleurt['t5_O'].append(t5_result['bleurt'][2])
    bleurt['t5_A'].append(t5_result['bleurt'][3])

    chat_result = load_json(os.path.join(chatgpt, filename+'.json'))
    rouge1['chat_full'].append(chat_result['rouge1'][0])
    rouge1['chat_S'].append(chat_result['rouge1'][1])
    rouge1['chat_O'].append(chat_result['rouge1'][2])
    rouge1['chat_A'].append(chat_result['rouge1'][3])

    bertscore_f1['chat_full'].append(chat_result['bertscore_f1'][0])
    bertscore_f1['chat_S'].append(chat_result['bertscore_f1'][1])
    bertscore_f1['chat_O'].append(chat_result['bertscore_f1'][2])
    bertscore_f1['chat_A'].append(chat_result['bertscore_f1'][3])

    bleurt['chat_full'].append(chat_result['bleurt'][0])
    bleurt['chat_S'].append(chat_result['bleurt'][1])
    bleurt['chat_O'].append(chat_result['bleurt'][2])
    bleurt['chat_A'].append(chat_result['bleurt'][3])

    gpt4_result = load_json(os.path.join(gpt4, filename+'.json'))
    rouge1['gpt4_full'].append(gpt4_result['rouge1'][0])
    rouge1['gpt4_S'].append(gpt4_result['rouge1'][1])
    rouge1['gpt4_O'].append(gpt4_result['rouge1'][2])
    rouge1['gpt4_A'].append(gpt4_result['rouge1'][3])

    bertscore_f1['gpt4_full'].append(gpt4_result['bertscore_f1'][0])
    bertscore_f1['gpt4_S'].append(gpt4_result['bertscore_f1'][1])
    bertscore_f1['gpt4_O'].append(gpt4_result['bertscore_f1'][2])
    bertscore_f1['gpt4_A'].append(gpt4_result['bertscore_f1'][3])

    bleurt['gpt4_full'].append(gpt4_result['bleurt'][0])
    bleurt['gpt4_S'].append(gpt4_result['bleurt'][1])
    bleurt['gpt4_O'].append(gpt4_result['bleurt'][2])
    bleurt['gpt4_A'].append(gpt4_result['bleurt'][3])

    gpt35_result = load_json(os.path.join(gpt35, filename+'.json'))
    rouge1['gpt35_full'].append(gpt35_result['rouge1'][0])
    rouge1['gpt35_S'].append(gpt35_result['rouge1'][1])
    rouge1['gpt35_O'].append(gpt35_result['rouge1'][2])
    rouge1['gpt35_A'].append(gpt35_result['rouge1'][3])

    bertscore_f1['gpt35_full'].append(gpt35_result['bertscore_f1'][0])
    bertscore_f1['gpt35_S'].append(gpt35_result['bertscore_f1'][1])
    bertscore_f1['gpt35_O'].append(gpt35_result['bertscore_f1'][2])
    bertscore_f1['gpt35_A'].append(gpt35_result['bertscore_f1'][3])

    bleurt['gpt35_full'].append(gpt35_result['bleurt'][0])
    bleurt['gpt35_S'].append(gpt35_result['bleurt'][1])
    bleurt['gpt35_O'].append(gpt35_result['bleurt'][2])
    bleurt['gpt35_A'].append(gpt35_result['bleurt'][3])

print('rouge1')
for key, value in rouge1.items():
    print(key)
    print(round(np.mean(value), 4))
    #print(len(value))

# for key, value in bertscore_f1.items():
#     print(key)
#     print(round(np.mean(value), 4))

# for key, value in bleurt.items():
#     print(key)
#     print(round(np.mean(value), 4))