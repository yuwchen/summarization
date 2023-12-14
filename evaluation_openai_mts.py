import os
import re
import pandas as pd
import evaluate
import json
from utils import *


def process_pred_text(text):
    if "Summary:" in text:
        parts = text.split("Summary:")
        pred_head = parts[0]
        pred_note = parts[1]
    else:
        parts = text.split('\n')
        pred_head = parts[0]
        pred_note = ' '.join(parts[1:])
    if len(pred_head)==0:
        parts = pred_note.split('\n')
        pred_head = parts[0]
        pred_note = ' '.join(parts[1:])

    return pred_head, pred_note.replace('\n','')

result_root = './Results/openai_mts_general/'

outputdir = './Evaluation_scores/mts_openai/'
gt_df = pd.read_csv('./data/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

gt_note = {}

soap_header = ['Full', 'subjective', 'objective', 'assessment_and_plan']


for idx, row in gt_df.iterrows():

    references = []
    predictions = []

    id = row['ID']
    dialog = row['dialogue']
    gt_note = row['section_text']
    gt_header = row['section_header']


    pred = load_json(os.path.join(result_root, str(id)+'.json'))['results']
    pred_head, pred_note = process_pred_text(pred)

    references.append(gt_note) 
    predictions.append(pred_note)
    
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    all_scores['pred_note'] = predictions
    all_scores['gt_note'] = references
    all_scores['section_header'] = gt_header
    all_scores['pred_section_header'] = pred_head

    outputname = os.path.join(outputdir, str(id)+'.json') 
    with open(outputname, 'w') as f:
        json.dump(all_scores, f, indent=2)

    
