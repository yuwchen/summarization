import re
import json
import evaluate

scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore', device='cpu'),
            {'model_type': 'microsoft/deberta-xlarge-mnli', 'device':'cpu'},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
        'bluert': (
            evaluate.load('bleurt', config_name='BLEURT-20'),
            {},
            ['scores'],
            ['bleurt']
        ),
}

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

TASK_B_HEADER = [
    "PHYSICAL EXAMINATION",
    "PHYSICAL EXAM",
    "EXAM",
    "ASSESSMENT AND PLAN",
    "VITALS REVIEWED",
    "VITALS",
    "FAMILY HISTORY",
    "ALLERGIES",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "REVIEW OF SYSTEMS",
    "CURRENT MEDICATIONS",
    "PROCEDURE",
    "RESULTS",
    "MEDICATIONS",
    "INSTRUCTIONS",
    "IMPRESSION",
    "SURGICAL HISTORY",
    "CHIEF COMPLAINT",
    "SOCIAL HISTORY",
    "HPI",
    "PLAN",
    "HISTORY OF PRESENT ILLNESS",
    "ASSESSMENT",
    "MEDICAL HISTORY"
]

TASK_B_SECTION_HEADER_MAP = {
    "FAMILY HISTORY": "subjective",
    "PHYSICAL EXAMINATION": "objective",
    "ALLERGIES": "subjective",
    "EXAM": "objective",
    "PAST HISTORY": "subjective",
    "PAST MEDICAL HISTORY": "subjective",
    "REVIEW OF SYSTEMS": "subjective",
    "CURRENT MEDICATIONS": "subjective",
    "ASSESSMENT AND PLAN": "assessment_and_plan",
    "PROCEDURE": "subjective",
    "RESULTS": "objective",
    "MEDICATIONS": "subjective",
    "INSTRUCTIONS": "assessment_and_plan",
    "IMPRESSION": "assessment_and_plan",
    "SURGICAL HISTORY": "subjective",
    "CHIEF COMPLAINT": "subjective",
    "SOCIAL HISTORY": "subjective",
    "HPI": "subjective",
    "PHYSICAL EXAM": "objective",
    "PLAN": "assessment_and_plan",
    "HISTORY OF PRESENT ILLNESS": "subjective",
    "ASSESSMENT": "assessment_and_plan",
    "MEDICAL HISTORY": "subjective",
    "VITALS": "objective",
    "VITALS REVIEWED": "objective",
}

def remove_duplicates_ordered(paragraph):
    sentences = paragraph.split('.')
    unique_sentences = []

    for sentence in sentences:
        if sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())

    unique_paragraph = '.'.join(unique_sentences)
    return unique_paragraph

def process_text(medical_text):

    header_pattern = "|".join(re.escape(header) for header in TASK_B_HEADER)
    segments = re.split(f"({header_pattern})", medical_text)
    segments = [segment.strip() for segment in segments if segment.strip()]

    results = {}

    current_header = None
    for segment in segments:
        if segment in TASK_B_HEADER:
            current_header = segment
            results[current_header] = ""
        else:
            results[current_header] += segment + " "

    return results

def process_gt_note(text):
    text = text.replace('CC','CHIEF COMPLAINT')
    if 'SUBJECTIVE' in text:
        text = text.replace('SUBJECTIVE', 'CHIEF COMPLAINT').replace('MEDICAL HISTORY','')
        text = text.split('.')[0] + '\n\nMEDICAL HISTORY\n\n' + '.'.join(text.split('.')[1:])
    return text

def process_pred_note(text):
    return text.split('Section text:')[0].split('Section header:')[-1], text.split('Section text:')[1]

def remove_symbol(sen):
    sen = sen.replace('<','').replace('>','').replace(':','').replace("#",'').replace("*","")
    return sen

def load_json(path):
    with open(path, 'r') as json_file:
        loaded_results = json.load(json_file)
    return loaded_results