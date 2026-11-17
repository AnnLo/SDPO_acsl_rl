import json
import logging
import re

from datasets import Dataset

# ---------------------------------------------------------------------------------------------------------------
# Prompts. PROMPT_MODE selects what the model is asked for:
#   "prove"    C function + ACSL contract that Frama-C/WP must prove (the original prompt, unchanged)
#   "code"     C function only, no ACSL (for training without formal verification)
# ---------------------------------------------------------------------------------------------------------------

_INTRO = ("You will receive a problem statement that specifies a single C function: its signature, parameters, "
          "result, constraints and examples. ")

_CODE_REQUIREMENTS = (
    "Code requirements:\n"
    "- Implement exactly the given signature: same name, parameter types and order, and return type.\n"
    "- Do not write main(), read input or print output. Do not use global variables, dynamic memory allocation "
    "or recursion.\n"
    "- <stdbool.h>, <stdint.h>, <limits.h> and <stddef.h> are included automatically; do not call any library "
    "functions.\n")

_COMMON_MISTAKES = (
    "Common mistakes that make Frama-C reject the whole contract:\n"
    "- \\result may only appear in ensures clauses.\n"
    "- Annotations are logic, not C: compare with ==, never assign with = or +=.\n"
    "- In a contract, write requires first, then assigns, then ensures.\n"
    "- There are no built-in logic functions such as max or min; define them with a logic function first.\n"
    "- Write annotations only in /*@ ... */ or //@ comments, never as C attributes or plain code.\n\n")

SYSTEM_PROMPTS = {
    "prove": (
        "You are an expert C programmer specializing in formal verification with Frama-C.\n\n"
        + _INTRO + "Write that function in C99 together with an ACSL contract that Frama-C's WP plugin can prove, "
        "including the absence of runtime errors (-wp-rte: no overflow, no out-of-bounds access, no division by zero).\n\n"
        + _CODE_REQUIREMENTS
        + "- You may write helper functions, but each needs its own complete contract.\n\n"
        "Contract requirements (in /*@ ... */ comments directly before each function):\n"
        "- requires: every constraint from the problem that the function relies on, including array validity with "
        "the exact index range (\\valid_read for inputs, \\valid for outputs) and the bounds needed to rule out "
        "arithmetic overflow.\n"
        "- assigns: exactly the memory the function writes (\\nothing if none).\n"
        "- ensures: a complete specification of the result as defined in the problem statement, strong enough that "
        "only a correct result satisfies it. Do not weaken it to something trivially true.\n"
        "- Every loop needs loop invariants strong enough to prove the postcondition and the absence of runtime "
        "errors, a loop assigns clause and a loop variant.\n"
        "- Do not use axioms or admit clauses; define any logic functions or predicates you need directly.\n\n"
        + _COMMON_MISTAKES
        + "Output format: reply with a single ```c code block containing the complete code (#include lines, ACSL "
        "annotations and function definitions). Write nothing after the code block."),
    "code": (
        "You are an expert C programmer.\n\n"
        + _INTRO + "Write that function in C99, correct for every input allowed by the constraints. Your code "
        "is compiled together with a test program that provides main() and calls your function, so write only the "
        "function (and any helpers it needs).\n\n"
        + _CODE_REQUIREMENTS
        + "- You may write helper functions.\n\n"
        "Output format: reply with a single ```c code block containing the complete code (#include lines and "
        "function definitions). Write nothing after the code block."),
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["prove"]  # kept for code that imports the old name

EXAMPLE_PROBLEM = '**Maximum of an array.** Given a non-empty array of integers, return its largest element.\n\n`int array_max(const int *a, int n);`\n\nParameters:\n- `a`: the array.\n- `n`: the number of elements of `a`.\n\nResult: the largest value among `a[0]`, ..., `a[n-1]`. `a` is not modified.\n\nConstraints:\n- 1 ≤ `n` ≤ 10^5\n- -10^9 ≤ `a[i]` ≤ 10^9 for all 0 ≤ i < n\n\nExamples:\n- `array_max({3, 9, 2}, 3)` → `9`\n- `array_max({-5}, 1)` → `-5`'
EXAMPLE_SOLUTION = '```c\n/*@ requires 1 <= n <= 100000;\n    requires \\valid_read(a + (0 .. n - 1));\n    requires \\forall integer k; 0 <= k < n ==> -1000000000 <= a[k] <= 1000000000;\n    assigns \\nothing;\n    ensures \\forall integer k; 0 <= k < n ==> \\result >= a[k];\n    ensures \\exists integer k; 0 <= k < n && \\result == a[k];\n*/\nint array_max(const int *a, int n) {\n  int m = a[0];\n  /*@ loop invariant 1 <= i <= n;\n      loop invariant \\forall integer k; 0 <= k < i ==> m >= a[k];\n      loop invariant \\exists integer k; 0 <= k < i && m == a[k];\n      loop assigns i, m;\n      loop variant n - i;\n  */\n  for (int i = 1; i < n; i++) {\n    if (a[i] > m) {\n      m = a[i];\n    }\n  }\n  return m;\n}\n```'
EXAMPLE_SOLUTION_CODE = re.sub(r"[ \t]*/\*@.*?\*/\n", "", EXAMPLE_SOLUTION, flags=re.S)  # same, without ACSL
EXAMPLE_SOLUTIONS = {"prove": EXAMPLE_SOLUTION, "code": EXAMPLE_SOLUTION_CODE}

DATASET_PATH = '/leonardo_work/AIFAC_F02_1251/anna/data/preprocessed_data.jsonl'
seed = 24

logger = logging.getLogger(__name__)

KEEP_COLUMNS = ["problem_id", "statement", "signature", "test_main", "num_tests"]


def _read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_distinct_expected(row: dict) -> bool:
    """False if a constant answer would pass every test."""
    return len({json.dumps(t["expected"], sort_keys=True) for t in row.get("fn_tests", [])}) > 1


def _unique_statements(rows: list[dict]) -> list[dict]:
    first = {}
    for row in rows:
        first.setdefault(row["statement"], row)
    return list(first.values())


def _to_record(row: dict, prompt_mode: str) -> dict:
    return {**{k: row[k] for k in KEEP_COLUMNS},
            "prompt": [{"role": "system", "content": SYSTEM_PROMPTS[prompt_mode]},
                       {"role": "user", "content": EXAMPLE_PROBLEM},
                       {"role": "assistant", "content": EXAMPLE_SOLUTIONS[prompt_mode]},
                       {"role": "user", "content": row["statement"]}]}


def _assert_disjoint(train: Dataset, test: Dataset) -> None:
    assert set(train["problem_id"]).isdisjoint(test["problem_id"]), "Train and test share problems"
    assert {p[-1]["content"] for p in train["prompt"]}.isdisjoint(
        p[-1]["content"] for p in test["prompt"]), "Train and test share prompts"


def load_and_prep_generated_data_functional(drop_constant_tests: bool = False, dataset_path: str = DATASET_PATH,
                                            prompt_mode: str = "prove"):
    """(train, test) from dataset_path (make_tests.py output), using rows with tests_ok.
    prompt_mode: "prove" (default, the original prompt) or "code" (no ACSL); see SYSTEM_PROMPTS."""
    if prompt_mode not in SYSTEM_PROMPTS:
        raise ValueError(f"prompt_mode must be one of {list(SYSTEM_PROMPTS)}, not {prompt_mode!r}")
    try:
        rows = _read_rows(dataset_path)
    except Exception as e:
        logger.error(f"[data] Error loading {dataset_path}: {e}")
        return None
    rows = [r for r in rows if r.get("tests_ok")]
    if drop_constant_tests:
        rows = [r for r in rows if _has_distinct_expected(r)]
    rows = _unique_statements(rows)

    split = Dataset.from_list([_to_record(r, prompt_mode) for r in rows]).train_test_split(
        test_size=0.1, seed=seed, shuffle=True)
    _assert_disjoint(split["train"], split["test"])
    logger.info(f"[data] {dataset_path} ({prompt_mode}): train {len(split['train'])}, test {len(split['test'])}")
    return split["train"], split["test"]