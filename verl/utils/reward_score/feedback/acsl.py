import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append("/leonardo_work/AIFAC_F02_352/anna/SDPO_formai/src/code/utils")
from reward_functions import score_with_feedback

WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "16")) // 2)


def _score(solution_str, ground_truth):
    tests = json.loads(ground_truth)
    score, feedback = score_with_feedback(solution_str, tests["test_main"], tests["num_tests"], "c")
    return {"score": score, "feedback": feedback, "solved": float(not feedback)}


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(_score, solution_strs, ground_truths))