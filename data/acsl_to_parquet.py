import json
import sys

from datasets import Dataset

sys.path.append("/leonardo_work/AIFAC_F02_1251/SDPO_formai")
from src.code.utils.data_loader import load_and_prep_generated_data_functional
from data.preprocess import write_rowgrouped_large


def to_row(ex, split):
    tests = json.dumps({"test_main": ex["test_main"], "num_tests": ex["num_tests"]})
    return {"data_source": "acsl", "prompt": ex["prompt"], "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": tests},
            "extra_info": {"split": split, "index": ex["problem_id"]}}


train, test = load_and_prep_generated_data_functional()
for split, ds in (("train", train), ("test", test)):
    write_rowgrouped_large(Dataset.from_list([to_row(ex, split) for ex in ds]), f"datasets/acsl/{split}.parquet")