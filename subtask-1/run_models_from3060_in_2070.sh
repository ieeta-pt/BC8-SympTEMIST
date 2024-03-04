CUDA_VISIBLE_DEVICES=0 python hf_trainer_full.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation random --aug_prob 0.8 --percentage_tags 0.8

CUDA_VISIBLE_DEVICES=0 python hf_trainer_full.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation random --aug_prob 0.8 --percentage_tags 1
