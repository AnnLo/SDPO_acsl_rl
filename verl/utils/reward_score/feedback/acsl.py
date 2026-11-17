import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append("/leonardo_work/AIFAC_F02_1251/anna/SDPO_formai/src/code/utils")
from reward_functions import (COMPILE_FLOOR, TEST_WEIGHT, extract_code_from_response,
                              response_contains_acsl_annotation, run_function_tests, score_with_feedback)

WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "16")) // 2)
MAX_DETAIL_CHARS = 1500  # per detail field in the feedback text


def _score_full(solution_str, ground_truth):
    """Compilation, tests and Frama-C proof (the original reward)."""
    tests = json.loads(ground_truth)
    score, feedback = score_with_feedback(solution_str, tests["test_main"], tests["num_tests"], "c")
    return {"score": score, "feedback": feedback, "solved": float(not feedback)}


def _score_no_verification(solution_str, ground_truth, acsl_weight):
    """Compilation and tests only; Frama-C is not run. With acsl_weight > 0, a response that also contains
    ACSL annotations gets that much extra (a format check, not a proof)."""
    gt = json.loads(ground_truth)
    code, acsl = extract_code_from_response(solution_str)
    tests = run_function_tests(code, gt["test_main"], gt["num_tests"], "c")
    compiles, pass_rate = float(tests["compiled"]), float(tests["pass_rate"])
    has_acsl = bool(response_contains_acsl_annotation(acsl))
    score = compiles * (COMPILE_FLOOR + TEST_WEIGHT * pass_rate) + acsl_weight * has_acsl
    solved = compiles == 1.0 and pass_rate == 1.0 and (has_acsl or acsl_weight == 0)

    feedback = ""
    if not solved:
        lines = []
        if not compiles:
            lines.append("The function does not compile together with the test program.")
        elif pass_rate < 1.0:
            lines.append(f"The function compiles but passes only {pass_rate:.0%} of the tests.")
        if acsl_weight and not has_acsl:
            lines.append("The response contains no ACSL contract (/*@ ... */) for the function.")
        # whatever run_function_tests reports as text (compiler errors, failing tests, ...)
        for key, value in tests.items():
            if isinstance(value, str) and value.strip():
                text = value.strip()
                if len(text) > MAX_DETAIL_CHARS:
                    text = text[:MAX_DETAIL_CHARS] + " ..."
                lines.append(f"{key}:\n{text}")
        feedback = "\n".join(lines)
    return {"score": score, "feedback": feedback, "solved": float(solved),
            "compiles": compiles, "pass_rate": pass_rate, "has_acsl": float(has_acsl)}


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos,
                        verify=True, acsl_weight=0.0, **kwargs):
    """verify=False: optimise compilation and test pass rate only (no Frama-C).
    Set from the config: custom_reward_function.reward_kwargs.verify / .acsl_weight."""
    if verify:
        fn = _score_full
    else:
        fn = lambda s, g: _score_no_verification(s, g, float(acsl_weight))
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, solution_strs, ground_truths))