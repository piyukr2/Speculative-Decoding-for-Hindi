"""
Data loading utilities for EESD Hindi experiments.

Training data : AI4Bharat IndicCorp (Hindi subset, ~5M tokens)
Evaluation data: XL-Sum Hindi (summarisation dataset with clean Hindi text)
"""

from __future__ import annotations

from typing import List, Optional

import json as _json
from urllib.request import urlopen

import pandas as pd
import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_batch(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# AI4Bharat IndicCorp – Training dataset
# ---------------------------------------------------------------------------

class AI4BharatHindiDataset(Dataset):
    """
    Streams from the AI4Bharat IndicCorpv2 Hindi split.

    HuggingFace identifier: "ai4bharat/IndicCorpv2"
    config name            : "indiccorp_v2"
    split name             : "hin_Deva"  (Hindi in Devanagari script)

    We cap at `max_samples` rows so that training stays within budget.
    The paper targets ~5M tokens; at avg. ~30 tokens/sentence that is
    roughly 167k sentences.
    """

    HF_DATASET = "ai4bharat/IndicCorpv2"
    HF_CONFIG = "indiccorp_v2"
    HF_SPLIT = "hin_Deva"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        max_samples: int = 167_000,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading AI4Bharat IndicCorpv2 (Hindi only), up to {max_samples} samples …")
        # Hindi is split into 3 files: hi-1.txt, hi-2.txt, hi-3.txt (~27 GB each).
        # Download all 3 and read lines across them until max_samples is reached.
        from huggingface_hub import hf_hub_download
        hindi_files = ["data/hi-1.txt", "data/hi-2.txt", "data/hi-3.txt"]
        self.texts = []
        for hf_file in hindi_files:
            print(f"  Downloading {hf_file} …")
            hi_path = hf_hub_download(
                repo_id=self.HF_DATASET,
                filename=hf_file,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            print(f"  Cached at: {hi_path}")
            with open(hi_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.texts.append(line)
                    if len(self.texts) >= max_samples:
                        break
            print(f"  {len(self.texts)} samples so far")
            if len(self.texts) >= max_samples:
                break
        print(f"  Loaded {len(self.texts)} Hindi samples from {len(hindi_files)} files")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = _tokenize_batch([self.texts[idx]], self.tokenizer, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# XL-Sum Hindi – Evaluation dataset
# ---------------------------------------------------------------------------

class XLSumHindiDataset(Dataset):
    """
    XL-Sum Hindi evaluation set (BBC multilingual summarisation).

    HuggingFace identifier: "csebuetnlp/xlsum"
    config name            : "hindi"

    We use the 'test' split for all evaluation runs.
    """

    HF_DATASET = "csebuetnlp/xlsum"
    HF_CONFIG = "hindi"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        split: str = "test",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading XL-Sum Hindi ({split} split) …")
        # csebuetnlp/xlsum ships a legacy loading script (xlsum.py) which all
        # recent versions of the datasets library refuse to run.  Neither
        # trust_remote_code nor hf:// parquet URLs help because the library
        # detects xlsum.py before even looking at the data files.
        #
        # Fix: use the public HuggingFace Datasets Server REST API, which
        # auto-converts every dataset to parquet and serves them directly.
        # This works for any dataset regardless of its original format and
        # never touches the loading script.
        _api = (
            f"https://datasets-server.huggingface.co/parquet"
            f"?dataset={self.HF_DATASET}&config={self.HF_CONFIG}&split={split}"
        )
        with urlopen(_api, timeout=60) as _resp:
            _parquet_urls = [
                f["url"] for f in _json.loads(_resp.read())["parquet_files"]
            ]
        dfs = [pd.read_parquet(url) for url in _parquet_urls]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        raw = HFDataset.from_pandas(df, preserve_index=False)
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))
        self.texts = [row["text"] for row in raw if row["text"].strip()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = _tokenize_batch([self.texts[idx]], self.tokenizer, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "text": self.texts[idx],
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_train_dataloader(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    batch_size: int = 16,
    max_samples: int = 167_000,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = AI4BharatHindiDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_eval_dataloader(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    batch_size: int = 1,
    max_samples: Optional[int] = 500,
    num_workers: int = 2,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    dataset = XLSumHindiDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
