import click
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def get_norm_emb(emb):
    return emb/np.linalg.norm(emb,ord=2, axis=-1, keepdims=True)

@click.command()
@click.argument("run_file")
@click.argument("codes_file")
@click.option("--output_folder", default="runs")
def main(run_file, codes_file, output_folder):
    codes_collection = pd.read_csv(codes_file,sep="\t")
    run_data = pd.read_csv(run_file,sep="\t")
    
    codes = codes_collection["code"]
    terms = codes_collection["term"]

    #print(run_data.columns)
    text_to_predict = run_data["text"]
    
    model = SentenceTransformer("Blaxzter/LaBSE-sentence-embeddings", device="cuda", cache_folder="../HF_CACHE")
    
    embeddings = model.encode(terms, show_progress_bar=True, batch_size=64) # Lx768
    
    lookup_codes_emb_norm = get_norm_emb(embeddings).T
    
    preds_embeddings = model.encode(text_to_predict, show_progress_bar=True, batch_size=64) # Nx768
    
    scores = get_norm_emb(preds_embeddings)  @ lookup_codes_emb_norm # NxL
    
    #index_max = np.argmax(scores, axis=-1) # Nx1
    list_of_codes_per_sample = []
    for index in np.argmax(scores, axis=-1):
        
        list_of_codes_per_sample.append(codes[index])
        
    run_data["code"] = list_of_codes_per_sample
    
    basename_run_file = os.path.basename(run_file)
    
    run_data.to_csv(os.path.join(output_folder,basename_run_file), sep="\t", index=False)
    

if __name__ == '__main__':
    main()