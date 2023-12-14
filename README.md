# Summarization

Code for Doctor-patient conversation summarization. 

## Dataset 
- [ACI-BENCH](https://github.com/wyim/aci-bench)  
- [MTS-Dialog](https://github.com/abachaa/MTS-Dialog)  

## Code

### Run OpanAI-model
(1) Set up API-key. 
```
sudo vim ~/.bash_profile
# Add 'export OPENAI_API_KEY='MY_OPENAPI_KEY' in the end of the ./bash_profile file
source ~/.bash_profile
# check whether the key is added successfully by 'echo $OPENAI_API_KEY'
```
(2) Run openai models and save the results
```
python openai_mts_general.py
python openai_mts_soap.py
python openai_aci_general.py
```

### Run Fine-tuned-LM-based model. 
```
python run_led.py           #run wanglab LED model on MTS-dialog data
python run_flant5.py        #run wanglab FLAN-T5 model on ACI-BENCH data
python run_flant5_seg.py    #run wanglab FLAN-T5 model on segmented ACI-BENCH data then merge the result using Falconsai/medical_summarization
```

### Run evaluation scores
evaluation_{model_name}_{dataset}
```
python evalution_gpt_aci.py          
python evalution_gpt_soap_aci.py     
python evalution_led_aci.py
python evalution_led_mts.py
python evalution_t5_aci.py
python evalution_t5seg_aci.py
python evalution_openai_mts.py
python evalution_openai_soap_mts.py 

```

### Summarize evaluation scores
```
python cross_domain_aci.py
python cross_domain_mts.py
```



