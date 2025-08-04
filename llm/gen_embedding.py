import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from tqdm import tqdm
from tokenize_data.data import get_dataset
import safetensors.torch as st
from accelerate import Accelerator
from datasets import load_dataset
import yaml

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)
    
class DatasetWithIdx(Dataset):
    def __init__(self, data, embeddings=None):
        self.data = data
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = {**self.data[idx], 'idx': idx}
        if self.embeddings is not None:
            res['embedding'] = self.embeddings.get(f"embedding_{idx}", None)
        return res

def compute_embeddings(model, inputs, attention_mask, device):
    """Compute embeddings before the LM head layer."""
    model.eval()
    inputs, attention_mask = inputs.to(device), attention_mask.to(device)
    class_to_model_map = {
        'LlamaForCausalLM': 'model',
        'Qwen2ForCausalLM': 'model',
        'MistralForCausalLM': 'model',
        'MixtralForCausalLM': 'model',
        'GemmaForCausalLM': 'model',
        'GPT2LMHeadModel': 'transformer',
        'HymbaForCausalLM': 'model',
    }
    model_class_name = model.module.__class__.__name__
    if model_class_name in class_to_model_map:
        model_attribute = class_to_model_map[model_class_name]
        base_model = getattr(model, model_attribute)
    else:
        raise ValueError(f"Model class '{model_class_name}' not found in the mapping.")
    with torch.no_grad():
        outputs = base_model(input_ids=inputs, attention_mask=attention_mask, return_dict=True).last_hidden_state

    return outputs


def main():
    parser = argparse.ArgumentParser("Prepare Base Gradients")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)
    print(args)

    accelerator = Accelerator(mixed_precision='no')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bf16:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    else:
        print("Warning: Using float32 model for gen embeddings !!!!!!")
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    if args.emb_model_mode == "eval":
        model.eval()
    elif args.emb_model_mode == "train":
        model.train()
    else:
        raise ValueError(f"Invalid model mode: {args.emb_model_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = args.padding_side
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_ds = load_dataset(args.data_name, split=args.train_split)
    dataset = get_dataset(
        raw_hf_ds=hf_ds,
        tokenizer=tokenizer,
        dataset_type=args.data_type,
        conversation_template=args.conversation_template,
        disable_group_texts=args.disable_group_texts,
        block_size=args.block_size,
        model_max_length=tokenizer.model_max_length,
        preprocessing_num_workers=4,
        overwrite_cache=False
    )
    dataset = DatasetWithIdx(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=default_data_collator, drop_last=False, num_workers=8
    )
    print(dataset, len(dataset), len(data_loader))
    model, data_loader = accelerator.prepare(model, data_loader)

    embedding_data = {}
    
    for batch in tqdm(data_loader, desc="Processing batches"):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['labels']  # Assuming the dataset provides labels
        idx = batch['idx'].tolist()
        

        embeddings = compute_embeddings(model, input_ids, attention_mask, device)
        if args.save_dtype == 'bfloat16':
            embeddings = embeddings.bfloat16().cpu()
        elif args.save_dtype == 'float32':
            embeddings = embeddings.float().cpu()
        else:
            raise ValueError(f"Invalid dtype: {args.save_dtype}")
        
        for i, index in enumerate(idx):
            embedding_data[f"embedding_{index}"] = embeddings[i]

    st.save_file(embedding_data, args.base_embedding_path)
    
    print(f"embeddings saved to {args.base_embedding_path}")

if __name__ == "__main__":
    main()
