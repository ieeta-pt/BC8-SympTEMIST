# Subtask 1.1: Training and Running Ensemble Models

## Training the Models for the MedProcNER Task

We use `roberta_trainer_config.yaml` for configuring the training process. This YAML file contains most hyperparameters used during the training of the models.

### Train with Validation Data
To train the models with validation data, use `hf_trainer.py` script. Here's how to use it:

```bash
python hf_trainer.py <checkpoint> --model_type_arch=<architecture> --percentage_tags=<tag_percentage> --augmentation=<aug_type> --context=<context_length> --epochs=<num_epochs> --batch=<batch_size> --random_seed=<seed>
```

Where the arguments are defined as follows:
- `<checkpoint>`: The checkpoint of the pre-trained model (String).
- `<architecture>`: The architecture of the classifier, options: `bilstm` or `dense` (Default: `bilstm`).
- `<tag_percentage>`: The percentage of tags that will be replaced during augmentation (Float, Default: `0.2`).
- `<aug_type>`: The type of augmentation to apply. Options: `None`, `ink`, or `random` (Default: `None`).
- `<context_length>`: The length of the context, right and left. Must be positive (Integer, Default: `64`).
- `<num_epochs>`: The number of epochs the model will train for (Integer, Default: `30`).
- `<batch_size>`: The batch size for training (Integer, Default: `8`).
- `<seed>`: The seed for generating random numbers (Integer, Default: `42`).

### Train with Full Data Without Validation

To train the models with full data without validation, use `hf_trainer_full.py` script. The parameters are the same as those for `hf_trainer.py`.

```bash
python hf_trainer_full.py <checkpoint> --model_type_arch=<architecture> --percentage_tags=<tag_percentage> --augmentation=<aug_type> --context=<context_length> --epochs=<num_epochs> --batch=<batch_size> --random_seed=<seed>
```

## How to Run Ensemble

To run the ensemble model, use `ensemble_inference.py` script. Here's how to use it:

```bash
python ensemble_inference.py <runs> --out=<output_file>
```
By default this will perform entity level ensembling.

Where the arguments are defined as follows:
- `<runs>`: The list of run files.
- `<output_file>`: The output file path where results will be stored.
