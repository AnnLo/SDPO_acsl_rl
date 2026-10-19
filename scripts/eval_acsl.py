"""Evaluates a model on the ACSL test set: compiles, tests, ACSL, proof, solved.

python scripts/eval_acsl.py run --model PATH --out results/base.json
python scripts/eval_acsl.py compare results/base.json results/trained.json --plot results/compare.png
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.append("/leonardo_work/AIFAC_F02_1251/SDPO_formai/src/code/utils")
from reward_functions import (COMPILE_FLOOR, PROOF_WEIGHT, TEST_WEIGHT, extract_code_from_response,
                              frama_c_proof_fraction, response_contains_acsl_annotation,
                              run_frama_c_on_code, run_function_tests)

METRICS = ["compiles", "all_tests", "acsl", "proof", "solved", "reward"]
WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "16")) // 2)


def evaluate_response(response: str, test_main: str, num_tests: int, language: str = "c") -> dict:
    """Each stage of the training reward, scored separately."""
    code, acsl = extract_code_from_response(response)
    tests = run_function_tests(code, test_main, num_tests, language)
    compiles, pass_rate = tests["compiled"], tests["pass_rate"]
    has_acsl = response_contains_acsl_annotation(acsl)
    proof, status = 0.0, None
    if pass_rate == 1.0 and has_acsl:
        output, status = run_frama_c_on_code(code, language)
        proof = frama_c_proof_fraction(output, status)
    return {
        "compiles": float(compiles),
        "all_tests": float(pass_rate == 1.0),
        "acsl": float(has_acsl),
        "proof": proof,
        "solved": float(status == "ok" and proof >= 0.999),
        "reward": compiles * (COMPILE_FLOOR + TEST_WEIGHT * pass_rate) + PROOF_WEIGHT * proof,
    }


def _summary(samples: list[dict], n: int) -> dict:
    df = pd.DataFrame(samples)
    summary = {f"{m}/mean@{n}": df[m].mean() for m in METRICS}
    summary[f"solved/pass@{n}"] = df.groupby("problem")["solved"].max().mean()
    summary["problems"] = df["problem"].nunique()
    return summary


def run(args) -> None:
    from vllm import LLM, SamplingParams
    data = pd.read_parquet(args.data)
    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, seed=args.seed,
              tensor_parallel_size=args.tp, max_model_len=args.max_model_len, gpu_memory_utilization=0.85)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outputs = llm.chat([[dict(m) for m in p] for p in data["prompt"]], params)

    jobs = []
    for i, (row, out) in enumerate(zip(data.itertuples(), outputs)):
        tests = json.loads(row.reward_model["ground_truth"])
        jobs += [(i, c.text, tests["test_main"], int(tests["num_tests"])) for c in out.outputs]
    with ThreadPoolExecutor(WORKERS) as pool:
        scores = list(pool.map(lambda j: evaluate_response(*j[1:], args.language), jobs))
    samples = [{"problem": i, "response": text, **s} for (i, text, *_), s in zip(jobs, scores)]

    result = {"model": args.model, "n": args.n, "temperature": args.temperature, "top_p": args.top_p,
              "summary": _summary(samples, args.n), "samples": samples}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["summary"], indent=1))


def compare(args) -> None:
    results = [json.load(open(p)) for p in args.results]
    table = pd.DataFrame({os.path.basename(p).removesuffix(".json"): r["summary"]
                          for p, r in zip(args.results, results)})
    print(table.round(3).to_string())
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        ax = table.drop("problems").plot.bar(figsize=(10, 4), rot=30, ylim=(0, 1))
        ax.set_ylabel("fraction of samples")
        ax.figure.tight_layout()
        ax.figure.savefig(args.plot, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tokenizer")
    r.add_argument("--data", default="datasets/acsl/test.parquet")
    r.add_argument("--out", required=True)
    r.add_argument("--language", default="c")
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--max-model-len", type=int, default=4608)
    r.add_argument("--tp", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("results", nargs="+")
    c.add_argument("--plot")
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()