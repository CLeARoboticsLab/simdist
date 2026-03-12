## Dynamics Fintuning

- [Processing Real-World Data](#processing-real-world-data)
- [Finetuning the World Model](#finetuning-the-world-model)

### Processing Real-World Data

First, aggreage real wold data with the following, replacing `<dataset_name>` with the name of the dataset in `datasets/real`:

```bash
python scripts/aggregate_realworld_data.py dataset_name=<dataset_name>
```

Next, process the data the same way as in [Data Generation](#data-generation), with:

```bash
python scripts/process_data.py dataset_name=<dataset_name>
```

### Finetuning the World Model

Finetune the dynamics only of the world model with the below, replacing `<dataset_name>` with the name of the dataset in `datasets/real/` and `<resume_checkpoint>` with the name of the checkpoint to finetune from `checkpoints/models/`. Checkpoints will be saved in `checkpoints/models/`; use `<run_name>` to specify the name of the checkpoints (optional; it's recommended to use a different name than the original checkpoint). Optionally, launch with `wandb.log=true` and `wandb.entity=<your_wandb_entity_or_username>` to enable Weights and Biases logging. Additional configuration is found in [`config/finetune_model.yaml`](config/finetune_model.yaml).

```bash
python scripts/finetune_model.py data.dataset_name=<real_dataset> checkpoint.resume_checkpoint=<resume_checkpoint> run_name=<run_name>
```