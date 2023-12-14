import os
import re
import pandas as pd
import evaluate
import json
from utils import *


GPT_HEADER = {"Subjective", "Objective", "Assessment", "Plan", "Assessment and Plan"}


def process_gpt_result(text):
    text = text.replace(":","")
    header_pattern = "|".join(re.escape(header) for header in GPT_HEADER)
    segments = re.split(f"({header_pattern})", text)
    segments = [segment.strip() for segment in segments if segment.strip()]
    results = {}
    current_header = None
    for segment in segments:
        if segment in GPT_HEADER:
            current_header = segment
            results[current_header] = ""
        else:
            results[current_header] += segment + " "
    return results


result_root = './Results/openai_mts_soap/'

outputdir = './Evaluation_scores/mts_openai_soap/'
gt_df = pd.read_csv('./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

gt_note = {}


for idx, row in gt_df.iterrows():

    references = []
    predictions = []

    id = row['ID']
    dialog = row['dialogue']
    gt_note = row['section_text']
    gt_header = row['section_header']

    pred = load_json(os.path.join(result_root, str(id)+'.json'))['results']
    gpt_result = process_gpt_result(pred)

    pred_soap = {'Full':'', 'Subjective':'', 'Objective':'', 'Assessment and Plan':''}
    for soap, value in gpt_result.items():
        if soap =='Assessment' or soap=='Plan':
            pred_soap['Assessment and Plan'] = pred_soap['Assessment and Plan'] + value
        else:
            pred_soap[soap] = pred_soap[soap] + value
        pred_soap['Full'] = pred_soap['Full'] + value
    pred_soap['Full'] = pred_soap['Full'].replace("N/A","")

    for soap in ['Full', 'Subjective', 'Objective', 'Assessment and Plan']:
        predictions.append(pred_soap[soap])
        references.append(gt_note)
    
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    all_scores['pred_note'] = predictions
    all_scores['gt_note'] = references
    all_scores['section_header'] = gt_header

    outputname = os.path.join(outputdir, str(id)+'.json') 
    with open(outputname, 'w') as f:
        json.dump(all_scores, f, indent=2)
    

    
