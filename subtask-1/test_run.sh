
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py NazaGara/NER-fine-tuned-BETO --model_type_arch bilstm --context 4
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py MMG/xlm-roberta-large-ner-spanish --model_type_arch bilstm --context 4
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4  --augmentation random --#percentage_tags 0.5
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4  --augmentation random --percentage_tags 0.6
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4  --augmentation random --percentage_tags 0.8

# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 128 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 32 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 16 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 8 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 8  --augmentation ukn
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 8 --augmentation random
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch dense --context 8 
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --model_type_arch bilstm --context 8
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --model_type_arch bilstm --context 16
# CUDA_VISIBLE_DEVICES=1 python hf_trainer.py PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --model_type_arch bilstm --context 32


#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation ukn --percentage_tags 0.6
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation ukn --percentage_tags 0.4
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation ukn --percentage_tags 0.5
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --model_type_arch bilstm --context 4 --augmentation ukn --percentage_tags 0.8
#CUDA_VISIBLE_DEVICES=1 python hf_trainer.py MMG/xlm-roberta-large-ner-spanish --model_type_arch bilstm --context 4 --batch 7
CUDA_VISIBLE_DEVICES=1 python hf_trainer.py NazaGara/NER-fine-tuned-BETO --model_type_arch bilstm --context 4 --batch 7