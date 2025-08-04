import torch
import torch.nn as nn
from tokenize_data.data import get_dataset
import os
import argparse
from datasets import load_dataset
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoModelForCausalLM
from peft import PeftModel
import yaml

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def loss_per_sample(logits, labels):
    vocab_size = logits.shape[-1]
    seq_len = logits.shape[1]
    batch_size = logits.shape[0]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    count_per_exp = (shift_labels.view(batch_size, -1) != -100).sum(dim=1)
    loss = loss.view(batch_size, -1).sum(dim=1) / count_per_exp
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Training')
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--model-ckpt', type=str)
    
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side

    hf_ds=load_dataset(args.train_data_name, split=args.train_data_split)
    train_dataset = get_dataset(
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
    print(f'devices count: {torch.cuda.device_count()}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.micro_batch_size*torch.cuda.device_count()*2, shuffle=False,
        collate_fn=default_data_collator,
        num_workers=torch.cuda.device_count()*4,
        pin_memory=True
    )

    query_loader = None
    if args.query_data_name:
        query_dataset = get_dataset(
            raw_hf_ds=load_dataset(args.query_data_name, split=args.query_split),
            tokenizer=tokenizer,
            dataset_type=args.data_type,
            conversation_template=args.conversation_template,
            disable_group_texts=args.disable_group_texts,
            block_size=args.block_size,
            model_max_length=tokenizer.model_max_length,
            preprocessing_num_workers=4,
            overwrite_cache=False
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=args.micro_batch_size*torch.cuda.device_count(), shuffle=False,
            collate_fn=default_data_collator,
            num_workers=torch.cuda.device_count()*4,
            pin_memory=True
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loss_matrix = []
    query_loss_matrix = []
    
    base_net = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device)
    print(f'Loading model from {args.model_ckpt}')
    for dirpath, _, filenames in tqdm(os.walk(args.model_ckpt)):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if not file_path.endswith('.safetensors'):
                continue
            try:
                net = PeftModel.from_pretrained(base_net, dirpath, torch_device=device)
            except Exception as e:
                print(f"Error loading {dirpath}: {e}")
                continue
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)

            net.eval()
            loss_list = []
            with torch.inference_mode():
                for batch in tqdm(train_loader):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    outputs = net(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).logits

                    loss = loss_per_sample(outputs, labels)
                    loss_list.append(loss)  # Ensure losses are on CPU

            train_loss_matrix.append(torch.cat(loss_list))

            if query_loader is not None:
                loss_list = []
                with torch.inference_mode():
                    for batch in tqdm(query_loader):
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)

                        outputs = net(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        ).logits

                        loss = loss_per_sample(outputs, labels)
                        loss_list.append(loss)

                query_loss_matrix.append(torch.cat(loss_list))

    train_loss_matrix = torch.stack(train_loss_matrix).cpu()
    if query_loader is not None:
        query_loss_matrix = torch.stack(query_loss_matrix).cpu()

    train_data_name = args.train_data_name.split('/')[-1]
    with open(f'{args.model_ckpt}/losses-{args.exp_id}-on-train-{train_data_name}.pkl', 'wb') as out:
        pickle.dump(train_loss_matrix, out)

    query_data_name = ( args.query_data_name.split('/')[-1] ) if query_loader is not None else None
    if query_loader is not None:
        with open(f'{args.model_ckpt}/losses-{args.exp_id}-on-query-{query_data_name}.pkl', 'wb') as out:
            pickle.dump(query_loss_matrix, out)
