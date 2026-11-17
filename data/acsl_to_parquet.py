"""Writes datasets/<name>/{train,test}.parquet from make_tests.py output.
Usage: python data/acsl_to_parquet.py --data /path/to/codenet_with_tests.jsonl --prompt-mode code --name acsl_code
       prompt modes: prove (original prompt, C + ACSL for Frama-C), code (C only, no ACSL)"""
import argparse
import json
import os
import sys

from datasets import Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # acsl_data_loader.py lives next to this script
from acsl_data_loader import DATASET_PATH, load_and_prep_generated_data_functional
from data.preprocess import write_rowgrouped_large


def to_row(ex, split, data_source):
    tests = json.dumps({"test_main": ex["test_main"], "num_tests": ex["num_tests"]})
    return {"data_source": data_source, "prompt": ex["prompt"], "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": tests},
            "extra_info": {"split": split, "index": ex["problem_id"]}}


ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DATASET_PATH, help="make_tests.py output, e.g. codenet_with_tests.jsonl")
ap.add_argument("--prompt-mode", default="prove", choices=["prove", "code"])
ap.add_argument("--name", default="acsl", help="writes datasets/<name>/{train,test}.parquet")
ap.add_argument("--drop-constant-tests", action="store_true")
args = ap.parse_args()

result = load_and_prep_generated_data_functional(args.drop_constant_tests, args.data, args.prompt_mode)
if result is None:
    sys.exit(f"could not load {args.data}")
train, test = result
os.makedirs(f"datasets/{args.name}", exist_ok=True)
for split, ds in (("train", train), ("test", test)):
    write_rowgrouped_large(Dataset.from_list([to_row(ex, split, "acsl") for ex in ds]),
                           f"datasets/{args.name}/{split}.parquet")
print(f"datasets/{args.name}: train {len(train)}, test {len(test)} ({args.prompt_mode} prompt, from {args.data})")