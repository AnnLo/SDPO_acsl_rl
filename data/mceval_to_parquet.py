"""Converts McEval's C generation set into an evaluation parquet for scripts/eval_acsl.py.

McEval (https://mceval.github.io, ICLR 2025) gives per problem a function signature, a docstring with examples, and a
test: a main() with assert(...) calls. This script turns each problem into the same format as the training data:
  - the prompt: the system prompt and few-shot example from the data loader (--prompt-mode code or prove), with a
    problem statement built from the docstring and signature;
  - test_main: McEval's test program, rewritten so that `./prog i` runs it and prints "PASS i" or "FAIL i"
    for the i-th assert (same convention as make_tests.py), compiled together with the model's function.
Problems that do not fit the training setup are left out and listed in <out>.skipped.jsonl: floating-point
signatures or tests (the training data has none), struct/typedef/global definitions the function would need, and
tests without assert calls.

Usage (on the login node, after downloading the data):
  huggingface-cli download Multilingual-Multimodal-NLP/McEval --repo-type dataset --include "generation/C.jsonl" \
      --local-dir /leonardo_work/AIFAC_F02_1251/anna/data/mceval
  python data/mceval_to_parquet.py /leonardo_work/AIFAC_F02_1251/anna/data/mceval/generation/C.jsonl \
      --out datasets/mceval_c/test.parquet --prompt-mode code
Then: python scripts/eval_acsl.py run --model ... --data datasets/mceval_c/test.parquet --out ..."""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # acsl_data_loader.py lives next to this script

FLOAT = re.compile(r"\b(float|double)\b")
TYPE_DEFS = re.compile(r"^\s*(typedef\b|struct\s+\w+\s*\{|union\s+\w+\s*\{|enum\s+\w+\s*\{)", re.M)
ASSERT = re.compile(r"\bassert\s*\(")
STR_CMP = re.compile(r'([A-Za-z_]\w*\s*\((?:[^()]|\([^()]*\))*\))\s*(==|!=)\s*("(?:[^"\\]|\\.)*")')
INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"][^>\"]+[>\"]\s*$", re.M)

HARNESS_HEAD = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
{includes}
/* McEval test, run one assert per invocation: ./prog i prints "PASS i" or "FAIL i" for the i-th assert. */
static int mc_target = -1, mc_count = 0;
#undef assert
#define assert(cond) do {{ int mc_k = mc_count++; int mc_ok = (cond) ? 1 : 0; \\
    if (mc_k == mc_target) {{ printf(mc_ok ? "PASS %d\\n" : "FAIL %d: assert failed\\n", mc_k); exit(0); }} }} while (0)

{signature};

"""
HARNESS_MAIN = """
int main(int argc, char **argv) {{
  mc_target = argc > 1 ? atoi(argv[1]) : -1;
  mc_test_main();
  printf("FAIL %d: assert not reached\\n", mc_target);
  return 0;
}}
"""


def get(row, *keys):
    for k in keys:
        if row.get(k):
            return row[k]
    return ""


def build_test_main(row):
    """McEval's test as a harness in the make_tests.py convention, or (None, reason)."""
    test, prompt, sig = get(row, "test"), get(row, "prompt"), get(row, "signature").strip().rstrip(";{").strip()
    if not test or not sig:
        return None, "no test or signature"
    n = len(ASSERT.findall(test))
    if not n:
        return None, "test has no assert calls"
    includes = "\n".join(sorted({i.strip() for i in INCLUDE.findall(prompt + "\n" + test)} - {"#include <assert.h>"}))
    body = INCLUDE.sub("", test)
    m = re.search(r"\bint\s+main\s*\([^)]*\)", body)
    if not m:
        return None, "test has no main()"
    body = body[:m.start()] + "int mc_test_main(void)" + body[m.end():]  # returns and helpers stay untouched
    # `f(...) == "Dan"` compares pointers; it only works when the solution and the test share one file. Compiled
    # separately (as in training), the addresses differ, so compare the strings instead.
    body = STR_CMP.sub(lambda q: f"(strcmp({q.group(1)}, {q.group(3)}) {q.group(2)} 0)", body)
    return (HARNESS_HEAD.format(includes=includes, signature=sig) + body + HARNESS_MAIN.format(), n), None


def build_statement(row):
    name = get(row, "entry_point") or "function"
    doc = get(row, "docstring") or get(row, "instruction")
    doc = re.sub(r"^\s*/\*+|\*+/\s*$", "", doc).strip()
    doc = "\n".join(l.strip().lstrip("*").strip() for l in doc.splitlines())
    sig = get(row, "signature").strip().rstrip(";{").strip()
    return f"**{name}.** {doc}\n\n`{sig};`"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mceval", help="McEval generation/C.jsonl")
    ap.add_argument("--out", default="datasets/mceval_c/test.parquet")
    ap.add_argument("--prompt-mode", default="code", choices=["prove", "code"])
    ap.add_argument("--keep-float", action="store_true", help="keep floating-point problems")
    args = ap.parse_args()

    from acsl_data_loader import EXAMPLE_PROBLEM, EXAMPLE_SOLUTIONS, SYSTEM_PROMPTS
    rows = [json.loads(l) for l in open(args.mceval) if l.strip()]
    out, skipped, stats = [], [], Counter(problems_in=len(rows))
    for row in rows:
        pid = get(row, "task_id") or f"mceval_{len(out) + len(skipped)}"
        sig, test, prompt = get(row, "signature"), get(row, "test"), get(row, "prompt")
        reason = None
        if not args.keep_float and (FLOAT.search(sig) or FLOAT.search(test)):
            reason = "floating point"
        elif TYPE_DEFS.search(prompt):
            reason = "needs struct/typedef definitions"
        if not reason:
            built, reason = build_test_main(row)
        if reason:
            stats[f"skipped: {reason}"] += 1
            skipped.append({"task_id": pid, "reason": reason})
            continue
        test_main, n = built
        statement = build_statement(row)
        prompt_msgs = [{"role": "system", "content": SYSTEM_PROMPTS[args.prompt_mode]},
                       {"role": "user", "content": EXAMPLE_PROBLEM},
                       {"role": "assistant", "content": EXAMPLE_SOLUTIONS[args.prompt_mode]},
                       {"role": "user", "content": statement}]
        out.append({"data_source": "mceval_c", "prompt": prompt_msgs, "ability": "code",
                    "reward_model": {"style": "rule",
                                     "ground_truth": json.dumps({"test_main": test_main, "num_tests": n})},
                    "extra_info": {"split": "test", "index": pid}})
        stats["problems_out"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import pandas as pd
    pd.DataFrame(out).to_parquet(args.out)
    with open(args.out + ".skipped.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in skipped)
    print(json.dumps(dict(stats), indent=2))
    print(f"wrote {args.out} ({len(out)} problems), skipped problems in {args.out}.skipped.jsonl")


if __name__ == "__main__":
    main()