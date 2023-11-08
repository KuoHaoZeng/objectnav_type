import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in zip(("train", "val"), (99061, 2950)):
        if split == "train":
            d = "./ObjectNavType_train.jsonl.gz"
        else:
            d = "./ObjectNavType_val.jsonl.gz"
        with gzip.open(d, "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(data=houses, dataset="ObjectNavType", split=split)
    return prior.DatasetDict(**data)
