import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
# from utils import load_model
random.seed(42)

from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

translator = "Helsinki-NLP/opus-mt-es-en"
tokenizer = AutoTokenizer.from_pretrained(translator)
model = MarianMTModel.from_pretrained(translator, cache_dir="translator").to("cuda")

data = pd.read_csv("dataset/medprocner_gazetteer/gazzeteer_medprocner_v1.tsv",sep="\t")
data_gs = pd.read_csv("dataset/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv",sep="\t")

data["term_translated"] = data["term"].progress_apply(lambda x: tokenizer.batch_decode(model.generate(**tokenizer(x, return_tensors="pt", padding=True).to("cuda")), skip_special_tokens=True)[0])


data.to_csv("dataset/medprocner_gazetteer/gazzeteer_medprocner_v1_translated.tsv",sep="\t",index=False)