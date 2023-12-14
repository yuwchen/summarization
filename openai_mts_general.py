from openai import OpenAI
import os
import json
import pandas as pd
import pickle


def creat_dir(dir_path):
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)

df = pd.read_csv('./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

creat_dir('./Results/openai_mts_general')


for index, row in df.iterrows():
  id = row['ID']
  section_header = row['section_header']
  dialogue = row['dialogue']
  print(id)
  print(dialogue)

  client = OpenAI()

  source_prefix = "Summarize the following patient-doctor dialogue. Include all medically relevant information. You should first predict the most relevant clinical note section header and then summarize the dialogue. Dialogue:"
  text  = source_prefix+dialogue
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": text}
    ]
  )

  print(response)
  outputname =  os.path.join('./Results/openai_mts_general',str(id)+'.json')
  results = {
      'id':id,
      'dialogue':dialogue,
      "results": response.choices[0].message.content,
      "model":response.model,
      'completion_tokens':response.usage.completion_tokens,
      'prompt_tokens':response.usage.prompt_tokens,
      'total_tokens':response.usage.total_tokens

  }

  with open(outputname, 'w') as f:
      json.dump(results, f)
