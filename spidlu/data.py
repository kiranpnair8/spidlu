"""Dataset preparation for causal language-model utility evaluation."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def load_tokenizer(model_name_or_path, revision=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_lm_datasets(cfg, tokenizer):
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    text_column = cfg.text_column

    def tokenize(batch):
        return tokenizer(batch[text_column])

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset[cfg.train_split].column_names,
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // cfg.max_seq_len) * cfg.max_seq_len
        result = {
            k: [t[i : i + cfg.max_seq_len] for i in range(0, total_length, cfg.max_seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    if cfg.smoke:
        for split in (cfg.train_split, cfg.eval_split, cfg.downstream_split):
            if split in lm_datasets:
                n = min(8, len(lm_datasets[split]))
                lm_datasets[split] = lm_datasets[split].select(range(n))
    lm_datasets.set_format("torch")
    return lm_datasets


def make_dataloader(dataset, batch_size, shuffle=False, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
