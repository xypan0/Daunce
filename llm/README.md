# Daunce demo for LLMs

## Overview
Daunce use a single [config file](config/qwen-math-demo.yaml) to set the parameters. A demo config file is provided here. The parameters in the config file should be set before running Daunce.

Important parameters to consider in the config file:
- `model_name`: The name of the model to be used (e.g., Qwen/Qwen2.5-0.5B-Instruct).
- `tokenizer_name`: The name of the tokenizer to be used (e.g., Qwen/Qwen2.5-0.5B-Instruct).
- `model_mode`: The mode of the model (`train`/`eval`).
- `data_name`: The name of the dataset to be used to train the perturbed model (e.g. AI-MO/NuminaMath-CoT). The code requires the dataset has `messages` field with compatible conversation format.
- `eval_data_name`: The name of the dataset to be used for calculating validation loss (e.g., AI-MO/NuminaMath-CoT). The code requires the dataset has `messages` field with compatible conversation format.
- `query_data_name`: The name of the dataset to be used as query data in training data attribution task.
- `train_data_name`: The name of the dataset to be used as training data in training data attribution task, i.e., Daunce traces the query data back to this training data. Usually this data is a subset of `data_name`.
- `conversation_template`: Please refer to [LMFlow](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#conversation-template) for more details on template and tokenization.
- `max_steps`: The maximum number of training steps. In this demo, we set it to 1 for quick testing.
- `data_bootstrap`: If `True`, Daunce sample a subset of the training data for each perturbed model. This is useful for large datasets to speed up the training process.
- `base_embedding_path`: The path to save model embeddings on dataset `data_name`.

## Steps to Run Daunce for LLMs
Please follow the below steps to run Daunce for LLMs, specifically for a Qwen model on a math dataset. We recommend to use as least 2 GPUs for this demo. Always adjust the [FSDP config](config/fsdp_config_qwen.yaml#L20) if a different number of GPUs is used as well as different model/sharding strategy.

1. **Prepare the embeddings**:
    - Run the `prepare.sh` script to generate embeddings (hiddens) for the LLMs.
    - This step is for generating embeddings for the Qwen model on `data_name` (in config file) dataset so that we need to only host one model during Daunce training.

2. **Run Daunce**:
    - Specify correct parameters in `runs/qwen-math-demo.yaml` file.
    - Run the Daunce training `./train.sh`.
    - This will sequentially train multiple perturbed models based on the config file. Finally, TDA signals are generated based on the trained models and the query data.

3. **Visualize the results**:
    - After training, you can visualize the results similar to the CIFAR-ResNet demo.