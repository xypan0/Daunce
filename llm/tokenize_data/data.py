from tokenize_data.conversation_template import PRESET_TEMPLATES
import logging
import hashlib
from tokenize_data.hf_decoder_model import (
    tokenize_function, 
    conversation_tokenize_function
)
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from itertools import chain

logger = logging.getLogger(__name__)

def group_text(
        tokenized_datasets, 
        model_max_length,
        block_size,
        disable_group_texts,
        truncate_to_model_max_length=True,
        group_texts_batch_size=1000,
        preprocessing_num_workers=8,
        overwrite_cache=False,
        streaming=False
):
    """
    Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
    a dictionary.
    """
    # training_args = TrainingArguments(output_dir=None)

    if block_size is None:
        block_size = model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is"
                " longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size`"
                " up to `tokenizer.model_max_length` you can override this "
                " default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if block_size > model_max_length:
            if truncate_to_model_max_length:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger"
                    f" than the maximum length for the model"
                    f"({model_max_length})."
                    f" Using block_size={model_max_length}."
                    f"If you would like to use a longer 'block_size' that is"
                    f" longer than the maximum length supported by the model,"
                    f" you can override this behavior with"
                    f"default with `--truncate_to_model_max_length False`."
                )
                block_size = model_max_length
            else:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger"
                    f"than the maximum length for the model"
                    f"({model_max_length})."
                    f"Using block_size={block_size}.")
                block_size = block_size
        else:
            block_size = block_size
    # Main data processing function that will concatenate all texts from
    # our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model
        # supported it instead of this drop, you can customize this part to
        # your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts
    # together, so group_texts throws away a remainder for each of those
    # groups of 1,000 texts. You can adjust that batch_size here but a
    # higher value might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation
    # of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    # with training_args.main_process_first(desc="grouping texts together"):
    group_batch_size = group_texts_batch_size
    if disable_group_texts:
        group_batch_size = 1
    if not streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=group_batch_size,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=group_batch_size,
        )

    return lm_datasets

def get_dataset(
    raw_hf_ds,
    tokenizer,
    add_special_tokens=True,
    dataset_type=None,
    conversation_template=None,
    disable_group_texts=False,
    block_size=512,
    model_max_length=None,
    preprocessing_num_workers=4,
    overwrite_cache=False,
    train_on_prompt=False
):
    label_columns = None                # Handles 3)
    if dataset_type == "text_only":
        tokenized_column_order = ["text"]
        label_columns = ["text"]
    elif dataset_type == "text2text":
        tokenized_column_order = ["input", "output"]
        label_columns = ["output"]
        add_special_tokens = False
    elif dataset_type == "conversation":
        if conversation_template:
            if conversation_template in PRESET_TEMPLATES.keys():
                conversation_template = PRESET_TEMPLATES[conversation_template]
            else:
                raise NotImplementedError(
                    f"Conversation template {conversation_template} is not supported yet."
                )
        else:
            logger.warning("No conversation template provided. Using default template.")
            conversation_template = PRESET_TEMPLATES['empty']
                    
        logger.warning(f"Conversation template: {conversation_template}")
    else:
        raise NotImplementedError(
            f"dataset type \"{dataset_type}\" is not supported yet."
        )
    column_names = list(raw_hf_ds.features)
    # Whether to truncate long sequences to fit into max_length
    use_truncation = False
    if disable_group_texts:
        use_truncation = True
    
    tokenize_fn = conversation_tokenize_function if "conversation" in dataset_type else tokenize_function
    tokenize_fn_kwargs = {
        "tokenizer": tokenizer,
        "column_names": column_names,
        'disable_group_texts': disable_group_texts,
        "block_size": block_size,
    }
    if "conversation" in dataset_type:
        tokenize_fn_kwargs["conversation_template"] = conversation_template
        tokenize_fn_kwargs["train_on_prompt"] = train_on_prompt
    else:
        tokenize_fn_kwargs["label_columns"] = label_columns
        tokenize_fn_kwargs["tokenized_column_order"] = tokenized_column_order
        tokenize_fn_kwargs["add_special_tokens"] = add_special_tokens
        tokenize_fn_kwargs["use_truncation"] = use_truncation
                        
    tokenize_kwargs = {}

    fingerprint = hashlib.md5(
        (
            raw_hf_ds._fingerprint
            + str(tokenizer)
            + f'###padding_side={tokenizer.padding_side}'
            + ('###conversation_template=' + str(conversation_template) if "conversation" in dataset_type else "")
            + f'###disable_group_texts={disable_group_texts}'
            + f'###block_size={block_size}'
        ).encode("utf-8")
    ).hexdigest()
    tokenize_kwargs = {
        "num_proc": preprocessing_num_workers,
        "load_from_cache_file": not overwrite_cache,
        "desc": "Running tokenizer on dataset",
        "new_fingerprint": fingerprint,
    }

    if block_size < tokenizer.model_max_length:
        logger.warning(
            f"block_size {block_size} < model_max_length {tokenizer.model_max_length}, "
            "use block_size for maximum tokenized sequence length."
        )
    tokenized_datasets = raw_hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=column_names,
        fn_kwargs=tokenize_fn_kwargs,
        **tokenize_kwargs
    )
    # training_args = TrainingArguments(output_dir=None)

    # with training_args.main_process_first(desc="dataset map tokenization"):
    if disable_group_texts:
        lm_dataset = tokenized_datasets
    else:
        lm_dataset = group_text(
            tokenized_datasets,
            model_max_length=model_max_length,
            block_size=block_size,
            disable_group_texts=disable_group_texts,
            overwrite_cache=overwrite_cache,
        )

    return lm_dataset


if __name__ == '__main__':
    # hf_ds = load_dataset('pxyyy/NuminaMath-CoT-smp10k', split='test')
    hf_ds = load_dataset('pxyyy/llama3.2-3B-generated-sample', split='train')

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)

    # dataset = get_dataset(
    #     raw_hf_ds=hf_ds,
    #     tokenizer=tokenizer,
    #     dataset_type='conversation',
    #     conversation_template='qwen2_5_math',
    #     disable_group_texts=False,
    #     block_size=1024,
    #     preprocessing_num_workers=4,
    #     overwrite_cache=True
    # )

    dataset = get_dataset(
        raw_hf_ds=hf_ds,
        tokenizer=tokenizer,
        dataset_type='text_only',
        conversation_template='qwen2_5_math',
        disable_group_texts=False,
        block_size=1024,
        preprocessing_num_workers=4,
        overwrite_cache=True
    )
    for d in dataset:
        print(len(d['input_ids']))
    # print(dataset)
    # print(dataset[0])
    # print(tokenizer.decode(dataset[0]['input_ids']))