from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np

data = pd.read_csv("dataset/medprocner_gazetteer/gazzeteer_medprocner_v1_translated.tsv",sep="\t")
data_gs = pd.read_csv("dataset/medprocner_train/tsv/medprocner_tsv_train_subtask2_translated.tsv",sep="\t")

codes = data["code"]
terms = data["term"]

gs_codes = data_gs["code"]
gs_text = data_gs["text"]

def get_norm_emb(emb):
    return emb/np.linalg.norm(emb,ord=2, axis=-1, keepdims=True)

models = [
            #'paraphrase-xlm-r-multilingual-v1', #0.19
#          'hiiamsid/sentence_similarity_spanish_es', #0.18
           "Blaxzter/LaBSE-sentence-embeddings", #0.22009
#          "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli", #0.168
#          "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"] #0.176
]
#models = ["pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", #0.1844
#          "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"] #0.15

#models = [
    #"kamalkraj/BioSimCSE-BioLinkBERT-BASE", #0.1962
    #"pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", #0.2209
    #"pritamdeka/S-BioBert-snli-multinli-stsb", #0.2176
    #"Blaxzter/LaBSE-sentence-embeddings" # 0.1978
#]

for model_ckp in models:
    model = SentenceTransformer(model_ckp, device="cuda:0", cache_folder="SENTENCE_MODEL")
    
    embeddings = model.encode(terms, show_progress_bar=True, batch_size=64) # Lx768
    
    lookup_codes_emb_norm = get_norm_emb(embeddings).T
    
    preds_embeddings = model.encode(gs_text, show_progress_bar=True, batch_size=64) # Nx768
    
    scores = get_norm_emb(preds_embeddings)  @ lookup_codes_emb_norm # NxL
    
    tp = 0
    for i,index in enumerate(np.argmax(scores, axis=-1)):
        if codes[index] == gs_codes[i]:
            tp+=1
    
    print(model_ckp, "acc:", tp/len(gs_codes))
        

