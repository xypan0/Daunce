#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.

It imports several modules, including dataclasses, field from typing, Optional from typing,
require_version from transformers.utils.versions, MODEL_FOR_CAUSAL_LM_MAPPING,
and TrainingArguments from transformers.

MODEL_CONFIG_CLASSES is assigned a list of the model config classes from
MODEL_FOR_CAUSAL_LM_MAPPING. MODEL_TYPES is assigned a tuple of the model types
extracted from the MODEL_CONFIG_CLASSES.
"""
import logging
from dataclasses import dataclass, field, fields, Field, make_dataclass
from pathlib import Path
from typing import Optional, List, Union, Dict

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = logging.getLogger(__name__)



@dataclass
class DatasetArguments:
    """
    Define a class DatasetArguments using the dataclass decorator.
    The class contains several optional parameters that can be used to configure a dataset for a language model.


    dataset_path : str
        a string representing the path of the dataset to use.

    dataset_name : str
        a string representing the name of the dataset to use. The default value is "customized".

    is_custom_dataset : bool
        a boolean indicating whether to use custom data. The default value is False.

    customized_cache_dir : str
        a string representing the path to the directory where customized dataset caches will be stored.

    dataset_config_name : str
        a string representing the configuration name of the dataset to use (via the datasets library).

    train_file : str
        a string representing the path to the input training data file (a text file).

    validation_file : str
        a string representing the path to the input evaluation data file to evaluate the perplexity on (a text file).

    max_train_samples : int
        an integer indicating the maximum number of training examples to use for debugging or quicker training.
        If set, the training dataset will be truncated to this number.

    max_eval_samples: int
        an integer indicating the maximum number of evaluation examples to use for debugging or quicker training.
        If set, the evaluation dataset will be truncated to this number.

    streaming : bool
        a boolean indicating whether to enable streaming mode.

    block_size: int
        an integer indicating the optional input sequence length after tokenization. The training dataset will be
        truncated in blocks of this size for training.

    train_on_prompt: bool
        a boolean indicating whether to train on prompt for conversation datasets such as ShareGPT.

    conversation_template: str
        a string representing the template for conversation datasets.

    dataset_cache_dir: str
        a string representing the path to the dataset cache directory. Useful when the default cache dir
        (`~/.cache/huggingface/datasets`) has limited space.

    The class also includes some additional parameters that can be used to configure the dataset further, such as `overwrite_cache`,
    `validation_split_percentage`, `preprocessing_num_workers`, `disable_group_texts`, `demo_example_in_prompt`, `explanation_in_prompt`,
    `keep_linebreaks`, and `prompt_structure`.

    The field function is used to set default values and provide help messages for each parameter. The Optional type hint is
    used to indicate that a parameter is optional. The metadata argument is used to provide additional information about
    each parameter, such as a help message.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use."}
    )
    dataset_name: Optional[str] = field(
        default="customized", metadata={"help": "Should be \"customized\""}
    )
    is_custom_dataset: Optional[bool] = field(
        default=False, metadata={"help": "whether to use custom data"}
    )
    customized_cache_dir: Optional[str] = field(
        default=".cache/llm-ft/datasets",
        metadata={"help": "Where do you want to store the customized dataset caches"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=1e10,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    group_texts_batch_size: int = field(
        default=1000,
        metadata={
            "help": (
                "Number of samples that will be grouped together to go though"
                " `group_texts` operation. See `--disable_group_texts` for"
                " detailed explanation of this operation."
            )
        }
    )
    disable_group_texts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether we disable group of original samples together to"
                " generate sample sequences of length `block_size`"
                " By Default, it is True, which means the long samples"
                " are truncated to `block_size` tokens"
                " and short samples are padded to `block_size` tokens."
                " If set to False, we group every 1000 tokenized"
                " sequences together, divide them into"
                " [{total_num_tokens} / {block_size}] sequences,"
                " each with `block_size` tokens"
                " (the remaining tokens are ommited."
                " This group text behavior is useful"
                " for continual pretrain or pretrain."
            )
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation File Path"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to train on prompt for conversation datasets such as ShareGPT."}
    )
    conversation_template: Optional[str] = field(
        default=None,
        metadata={"help": "The template for conversation datasets."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("The path to the dataset cache directory. Useful when the "
                           "default cache dir (`~/.cache/huggingface/datasets`) has limited space.")}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


