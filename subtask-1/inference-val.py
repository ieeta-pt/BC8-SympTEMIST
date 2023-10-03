import click

from utils import load_model, load_model_and_tokenizer, load_model_local
from data import load_train_test_split, BIOTagger, SelectModelInputs, EvaluationDataCollator
from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForMaskedLM,AutoModel
import transformers
from decoder import decoder
from torch.utils.data import DataLoader
import torch
from ensemble import ensemble_entity_level, ensemble_span_level
import pandas as pd
from collections import defaultdict 
import random

import os
import json


def decoder_from_samples(prediction_batch, context_size):
    
    documents = {}
    padding = context_size

    # reconsturct the document in the correct order
    for i in range(len(prediction_batch)):
        doc_id = prediction_batch[i]['doc_id'].split('/')[-1]
        if doc_id not in documents.keys():
            documents[doc_id] = {}
            
            # run 1 time is enough for this stuff

        documents[doc_id][prediction_batch[i]['sequence_id']] = {
            'output': prediction_batch[i]['output'],
            'offsets': prediction_batch[i]['offsets'],
            'text':prediction_batch[i]["text"]}

    print("DOCUMENTS:", len(documents))

    predicted_entities = {}
    # decode each set of labels and store the offsets
    for doc in documents.keys():
        text = documents[doc][0]["text"]
        current_doc = [documents[doc][seq]['output'] for seq in sorted(documents[doc].keys())]
        current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
        predicted_entities[doc] = {"decoder": decoder(current_doc, current_offsets, padding=padding, text=text), "document_text": text}
    return predicted_entities

def remove_txt(data):
    new_data = {}
    for k,v in data.items():
        new_k, _ = os.path.splitext(k)
        new_data[new_k]=v
        
    return new_data

@click.command()
@click.option("--checkpoint")
# @click.option("--testset_folder")
@click.option("--out_folder", default="val-inference")
def main(checkpoint, out_folder):
    random.seed(42)
    
    if torch.cuda.is_available():
        #single GPU bc CRF
        assert torch.cuda.device_count()==1
        
    model, tokenizer, config = load_model_local(checkpoint)
    model = model.to(f"cuda")
    tokenizer.model_max_length = 512



    
    train_ds, test_ds = load_train_test_split("../symptemist_train/subtask1-ner/",
                                          tokenizer=tokenizer,
                                            context_size=config.context_size,
                                            test_split_percentage=0.15, 
                                          train_transformations=None,
                                          train_augmentations=None,
                                         test_transformations=None)
    
    # test_ds = InferenceDataset(testset_folder, tokenizer=tokenizer, context_size=config.context_size)
    
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, 
                                              padding=True,
                                              label_pad_token_id=tokenizer.pad_token_id)
    
    dl = DataLoader(test_ds, batch_size=8, collate_fn=eval_datacollator)
    
    outputs = []
    for train_batch in dl:
        with torch.no_grad():
            train_batch["output"] = model(**train_batch["inputs"].to("cuda"))
            train_batch |= train_batch["inputs"]
            del train_batch["inputs"]
        keys = list(train_batch.keys()) + ["output"]
        for i in range(len(train_batch["output"])):
            outputs.append({k:train_batch[k][i] for k in keys})
            
    predicted_entities = decoder_from_samples(outputs, context_size=config.context_size)
    predicted_entities = remove_txt(predicted_entities)
    
    # if revision=="main":
    fOut_name = "-".join(checkpoint.split("/")[-2:])
    # else:
    #     #from hun
    #     fOut_name = f"{revision}"

    with open(os.path.join(out_folder,f"{fOut_name}.json"),"w") as fOut:
        fOut.write(json.dumps(predicted_entities))
    
    
if __name__ == '__main__':
    main()