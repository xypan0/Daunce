import argparse
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from datasets import load_dataset
from accelerate.utils.tqdm import tqdm
from tokenize_data.data import get_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator
import safetensors.torch as st
from accelerate import Accelerator 
from accelerate.utils import set_seed, broadcast
import time
import math
from accelerate.logging import get_logger
import wandb
import numpy as np
import logging
import logging.config
import shutil
import yaml

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

class DatasetWithIdx(Dataset):
    def __init__(self, data, embedding_path=None):
        self.data = data
        self.embeddings = st.load_file(embedding_path) if embedding_path else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx=int(idx)
        res = {**self.data[idx], 'idx': idx}

        if self.embeddings is not None and f"embedding_{idx}" in self.embeddings:
            res['embedding'] = self.embeddings[f"embedding_{idx}"]

        return res

def save_model(accelerator, accelerate_model, tokenizer=None, save_dir=None, **kargs):

    accelerator.wait_for_everyone()
    accelerator.print(f"saving model at {save_dir} ...")
    unwrapped_model = accelerator.unwrap_model(accelerate_model)
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(accelerate_model),
        # max_shard_size="2GB"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

def logging_stat_dict(stat_dict, use_wandb=False, accelerator=None, wandb_commit=True):
    logger = get_logger('accelerator')
    stat_str_list = []
    for key, value in stat_dict.items():
        stat_str_list.append(f' {key} = {value},')

    stat_str = ''.join(stat_str_list)
    logger.info(stat_str)

    if use_wandb and accelerator.is_main_process:
        wandb.log(stat_dict, commit=wandb_commit)

def loss_per_sample(logits, labels):
    vocab_size = logits.shape[-1]
    seq_len = logits.shape[1]
    batch_size = logits.shape[0]
    # logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    count_per_exp = (shift_labels.view(batch_size, -1) != -100).sum()/batch_size
    loss=loss.reshape(batch_size, -1)
    return loss.sum(dim=-1)/count_per_exp


def eval(model, data_loader, accelerator):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask, return_dict=True).logits
            outputs = loss_per_sample(logits, labels)

        total_loss += accelerator.gather(outputs).detach().cpu().mean()

    return total_loss / len(data_loader)

def train(model, lm_head, data_loader, eval_loader, rand_ksi, optimizer, scheduler, accelerator, args):
    
    lm_head.train()
    if args.model_mode == "eval":
        model.eval()
    elif args.model_mode == "train":
        model.train()
    else:
        raise ValueError(f"Invalid model mode: {args.model_mode}")

    
    grad_accumulation_steps = args.global_batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accumulation_steps = grad_accumulation_steps // world_size
    print(f"gradient_accumulation_steps: {grad_accumulation_steps}")
    if args.max_steps is None:
        max_steps = math.ceil(len(data_loader.dataset) / args.global_batch_size * args.epochs)
    else:
        max_steps = args.max_steps

    train_iterator = iter(data_loader)

    for step in tqdm(range(max_steps), desc="Training"):
        optimizer.zero_grad()
        total_loss = 0.0
        total_base_loss = 0.0
        total_model_loss = 0.0
        total_fo = 0.0
        total_fo_grad = 0.0
        total_factor = 0.0

        for inner_step in range(grad_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(data_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"] 
            attention_mask = batch["attention_mask"] 
            labels = batch["labels"] 
            embedding = batch["embedding"] 
            indexes = batch["idx"] 
            batch_rand_ksi = rand_ksi[indexes]

            factor = 1 + args.rho * (2 * batch_rand_ksi - 1)

            with torch.no_grad():
                embedding = embedding.to(lm_head.weight.dtype)
                base_outputs = lm_head(embedding)
            base_outputs = base_outputs.to(torch.float32).requires_grad_(True)
            base_loss = loss_per_sample(base_outputs, labels)

            scaler=1e5
            grad = torch.autograd.grad(
                outputs=base_loss * scaler, 
                inputs=base_outputs,
                grad_outputs=torch.ones_like(base_loss),
                only_inputs=True,  # We're only interested in logits' gradient
                create_graph=False,  # Set to True if you need higher-order derivatives
                retain_graph=False,)
            fo_grad = torch.sum(grad[0].detach()**2) / grad[0].shape[0] / scaler**2
            base_loss = base_loss.detach()
            base_outputs = base_outputs.detach()
            mean_base_loss = torch.sum(base_loss) / base_loss.shape[0]
            all = model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=False)
            outputs = all.logits.to(torch.float32)
            model_loss = loss_per_sample(outputs, labels)

            factor = factor.to(outputs.dtype)
            factor_square = (factor**2).mean()

            scaled_outputs = (outputs - base_outputs) * scaler

            scaled_grad = grad[0].detach()

            fo = torch.sum(factor.view(outputs.shape[0], 1, 1) * (scaled_outputs * scaled_grad)) / base_loss.shape[0] / scaler**2

            loss = args.gamma * torch.mean(model_loss - mean_base_loss.detach()) - fo


            loss = loss / grad_accumulation_steps
            accelerator.backward(loss)

            total_loss += accelerator.gather(loss).detach().cpu().mean()
            total_base_loss += accelerator.gather(mean_base_loss).detach().cpu().mean() / grad_accumulation_steps
            total_model_loss += accelerator.gather(torch.mean(model_loss)).detach().cpu().mean() / grad_accumulation_steps
            total_fo += accelerator.gather(fo).detach().cpu().mean() / grad_accumulation_steps
            total_fo_grad += accelerator.gather(fo_grad).detach().cpu().mean() / grad_accumulation_steps
            total_factor += accelerator.gather(factor_square).detach().cpu().mean() / grad_accumulation_steps

        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1e9)

        if (step+1) % args.log_freq == 0:
            log_dict = {
                'lr': scheduler.get_last_lr()[0],
                'train/train_loss': total_loss,
                'train/train_base_loss': total_base_loss,
                'train/train_model_loss': total_model_loss,
                'train/train_fo': total_fo / args.rho if args.rho != 0 else total_fo,
                'train/grad_norm': grad_norm,
                'train/train_fo_grad_sqared': total_fo_grad,
                'train/factor_sqared': total_factor,
                'step': (step+1),
            }
            if (step+1) % args.eval_freq == 0:
                eval_loss = eval(model, eval_loader, accelerator)
                log_dict['eval/eval_loss'] = eval_loss
            logging_stat_dict(log_dict, use_wandb=args.use_wandb, accelerator=accelerator)

        optimizer.step()
        scheduler.step()

        

    return total_loss / len(data_loader), total_base_loss / len(data_loader)


def optimize(args, accelerator, data_loader, eval_loader, rand_ksi):
    # prepare model, tokenizer, and data loader
    if args.bf16:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    
    target_modules_map = {
        'LlamaForCausalLM': ['up_proj', 'down_proj', 'gate_proj'],
        'Qwen2ForCausalLM': ['up_proj', 'down_proj', 'gate_proj'],
        'GPT2LMHeadModel': ['mlp.c_fc', 'mlp.c_proj'],
    }
    model_class_name = model.__class__.__name__
    if model_class_name in target_modules_map:
        target_modules = target_modules_map[model_class_name]
    else:
        raise ValueError(f"Model class '{model_class_name}' not found in the mapping.")
    lora_config = LoraConfig(
        r=args.lora_rank,  # Low-rank dimension
        lora_alpha=args.lora_alpha,  # Scaling factor
        target_modules=target_modules,  # Apply LoRA to attention layers
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Bias handling in LoRA
        task_type="CAUSAL_LM",  # Task type (e.g., sequence classification)
    )
    
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    accelerator.print(model)
    model.print_trainable_parameters()

    lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
    # could be all zero weight if LLM is loaded across all shards
    lm_head.load_state_dict(model.lm_head.state_dict())
    if args.bf16:
        lm_head.to(torch.bfloat16)
    else:
        lm_head.to(torch.float32)
    
    # prepare optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), fused=True)

    if args.max_steps is None:
        max_steps = math.ceil(len(data_loader.dataset) / args.global_batch_size * args.epochs)
    else:
        max_steps = args.max_steps

    warmup_steps = math.ceil(args.warmup_ratio * max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    model, data_loader, eval_loader, optimizer = accelerator.prepare(
        model, data_loader, eval_loader, optimizer
    )

    lm_head = lm_head.to(accelerator.device)

    rand_ksi = rand_ksi.to(accelerator.device)
    broadcast(rand_ksi, 0)

    loss, base_loss = train(model, lm_head, data_loader, eval_loader, rand_ksi, optimizer, scheduler, accelerator, args)

    if accelerator.is_main_process and args.save_dir is not None and os.path.exists(args.save_dir):
        accelerator.print(f"delete model at {args.save_dir} ...")
        shutil.rmtree(args.save_dir)

    if args.save_dir is not None:
        save_model(accelerator, model, tokenizer=None, save_dir=args.save_dir)

def load_all_data(args, tokenizer):
    hf_ds=load_dataset(args.data_name, split=args.train_split)
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
    dataset = DatasetWithIdx(dataset, args.base_embedding_path)
    rand_ksi = torch.rand((len(dataset),), dtype=torch.float32)

    if args.data_bootstrap:
        if args.max_steps is None:
            print(f"Warning!!! max_steps is not provided while data_bootstrap is on. Using full dataset...")
            bootstrap_size = len(dataset)
        else:
            bootstrap_size = args.max_steps * args.global_batch_size
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), bootstrap_size, replace=False))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.micro_batch_size, shuffle=False, 
        collate_fn=default_data_collator, 
        num_workers=8
    )

    eval_dataset = get_dataset(
        raw_hf_ds=load_dataset(args.eval_data_name, split=args.eval_split),
        tokenizer=tokenizer,
        dataset_type=args.data_type,
        conversation_template=args.conversation_template,
        disable_group_texts=args.disable_group_texts,
        block_size=args.block_size,
        model_max_length=tokenizer.model_max_length,
        preprocessing_num_workers=4,
        overwrite_cache=False
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.micro_batch_size, shuffle=False, 
        collate_fn=default_data_collator, 
        num_workers=8
    )

    return data_loader, eval_loader, rand_ksi

def main():
    parser = argparse.ArgumentParser("GPT2 Influence Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    
    # Override options (only those needed in shell script)
    parser.add_argument("--rho", type=float, help="Override rho")
    parser.add_argument("--gamma", type=float, help="Override gamma")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--wandb_run_name", type=str, help="Override wandb run name")
    parser.add_argument("--save_dir", type=str, help="Override checkpoint save dir")
    parser.add_argument("--pseudo_random", type=int, help="Override random seed")

    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)
    print(args)

    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    set_seed(args.pseudo_random)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side

    if args.bf16:
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
        )

    accelerator.print(accelerator.state.fsdp_plugin)

    with accelerator.main_process_first():
        data_loader, eval_loader, rand_ksi = load_all_data(args, tokenizer)

    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args
        )
    try:
        optimize(
            args=args,
            accelerator=accelerator,
            data_loader=data_loader,
            eval_loader=eval_loader,
            rand_ksi=rand_ksi,
        )
    except Exception as e:
        if args.use_wandb and accelerator.is_main_process:
            wandb.finish()
        raise e

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()
    


if __name__ == "__main__":
    
    main()