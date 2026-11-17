import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FRAMA_C_TIMEOUT = float(os.environ.get("FRAMA_C_TIMEOUT", "30"))
FRAMA_C_TIMEOUT_CREDIT = float(os.environ.get("FRAMA_C_TIMEOUT_CREDIT", "0.25"))

CODE_FENCE_PATTERN = re.compile(r'```[A-Za-z0-9+_-]*[ \t]*\n?(.*?)\n?```', re.DOTALL)
OPEN_FENCE_PATTERN = re.compile(r'```[A-Za-z0-9+_-]*[ \t]*\n(.*)', re.DOTALL)
# Included in every solution, so missing standard includes are not counted as errors.
_HEADERS = ("stdbool.h", "stdint.h", "limits.h", "stddef.h")
_INCLUDE_FLAGS = [flag for h in _HEADERS for flag in ("-include", h)]
STANDARD_HEADERS = "".join(f"#include <{h}>\n" for h in _HEADERS) + "#line 1\n"
ACSL_BLOCK_PATTERN = re.compile(r'/\*@(.*?)\*/', re.DOTALL)
ACSL_LINE_PATTERN = re.compile(r'//@(.*)')
_VLA_PARAM = re.compile(r"\[[^\]]*[A-Za-z_][^\]]*\]")
_LANGUAGE_ALIASES = {"c": "c", "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp"}

# Linked into every test program: a crash exits at once with 128 + signal,
# instead of waiting for the system's core-dump handler.
_CRASH_GUARD_C = r"""
#include <signal.h>
#include <string.h>
#include <unistd.h>
static char crash_guard_stack[1 << 16];
static void crash_guard_handler(int sig) { _exit(128 + sig); }
__attribute__((constructor)) static void crash_guard_install(void) {
    stack_t ss = {.ss_sp = crash_guard_stack, .ss_size = sizeof crash_guard_stack};
    sigaltstack(&ss, 0);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = crash_guard_handler;
    sa.sa_flags = SA_ONSTACK;
    int sigs[] = {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL};
    for (unsigned i = 0; i < sizeof sigs / sizeof sigs[0]; i++) sigaction(sigs[i], &sa, 0);
}
"""



def normalize_language(language: str) -> str:
    """Maps a language label to "c" or "cpp"; unknown labels default to "cpp"."""
    key = (language or "").strip().lower()
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    logger.warning(f"Unrecognized language {language!r}; defaulting to 'cpp'.")
    return "cpp"


def _toolchain(language: str) -> Tuple[str, str, str]:
    """(compiler, standard flag, file suffix) for "c" or "cpp"."""
    return ("gcc", "-std=c11", ".c") if language == "c" else ("g++", "-std=c++17", ".cpp")


def supports_language(signature: str, language: str) -> bool:
    """C++ has no variable-length array parameters such as `int a[n]`."""
    return normalize_language(language) == "c" or not _VLA_PARAM.search(signature)


def extract_acsl_annotations(code: str) -> List[str]:
    if not code:
        return []
    return ([m.group(1).strip() for m in ACSL_BLOCK_PATTERN.finditer(code)]
            + [m.group(1).strip() for m in ACSL_LINE_PATTERN.finditer(code)])


def extract_code_from_response(response: str) -> Tuple[str, List[str]]:
    """First fenced block (an unclosed one if cut off, else the whole response) and its ACSL annotations."""
    match = CODE_FENCE_PATTERN.search(response) or OPEN_FENCE_PATTERN.search(response)
    code = match.group(1).strip() if match else response.strip()
    return code, extract_acsl_annotations(code)


def response_contains_acsl_annotation(acsl_annotations: List[str]) -> bool:
    return any(a.strip() for a in acsl_annotations)



def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"... [{len(text) - n} more characters]"


def _limit_resources(mem_bytes: int):
    """preexec_fn: caps memory and disables core dumps, within the hard limits."""
    def apply():
        for res, value in ((resource.RLIMIT_AS, mem_bytes), (resource.RLIMIT_CORE, 0)):
            try:
                _, hard = resource.getrlimit(res)
                resource.setrlimit(res, (value if hard == resource.RLIM_INFINITY else min(value, hard), hard))
            except (ValueError, OSError):
                pass
    return apply


def compile_with_test_main(code: str, test_main: str, language: str = "c",
                           timeout: float = 60.0) -> Tuple[bool, Optional[str], str]:
    """Builds the function with the test program. Returns (success, exec_path, error);
    the caller removes os.path.dirname(exec_path). A main() in `code` is ignored."""
    if not code:
        return False, None, "empty"
    compiler, std, suffix = _toolchain(normalize_language(language))
    files = {"solution" + suffix: code, "tests" + suffix: test_main, "crash_guard.c": _CRASH_GUARD_C}
    steps = [
        ["gcc", "-std=gnu11", "-O2", "-c", "crash_guard.c", "-o", "crash_guard.o"],
        [compiler, std, "-O2", *_INCLUDE_FLAGS, "-c", "solution" + suffix, "-o", "solution.o", "-Dmain=__model_main_ignored"],
        [compiler, std, "-O0", "-o", "prog", "solution.o", "tests" + suffix, "crash_guard.o", "-lm"],
    ]
    tmp = tempfile.mkdtemp(prefix="sol_")
    try:
        for name, text in files.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(text)
        for cmd in steps:
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                shutil.rmtree(tmp, ignore_errors=True)
                return False, None, result.stderr.strip()
        return True, os.path.join(tmp, "prog"), ""
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, None, "compilation timed out" if isinstance(e, subprocess.TimeoutExpired) else f"system error: {e}"


def _describe_exit(i: int, code: int) -> str:
    if code > 128:
        try:
            name = signal.Signals(code - 128).name
        except ValueError:
            name = f"signal {code - 128}"
        return f"Test {i}: runtime error ({name}, e.g. invalid memory access or abort)."
    if code < 0:
        return f"Test {i}: runtime error (killed by signal {-code})."
    return f"Test {i}: runtime error (exit code {code})."


def run_test_index(exec_path: str, i: int, timeout: float = 5.0, mem_gb: float = 2.0) -> Tuple[bool, str]:
    """Runs test i. Returns (passed, failure description)."""
    try:
        result = subprocess.run([exec_path, str(i)], capture_output=True, text=True, timeout=timeout,
                                preexec_fn=_limit_resources(int(mem_gb * 1024 ** 3)))
    except subprocess.TimeoutExpired:
        return False, f"Test {i}: time limit exceeded ({timeout}s)."
    except Exception as e:
        return False, f"Test {i}: could not run ({e})."
    if result.returncode != 0:
        return False, _describe_exit(i, result.returncode)
    lines = result.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if line == f"PASS {i}":
        return True, ""
    if line.startswith(f"FAIL {i}"):
        return False, f"Test {i}: wrong result: " + _clip(line[len(f"FAIL {i}"):].lstrip(": "), 300)
    return False, f"Test {i}: no result (the program printed something unexpected)."


def run_function_tests(code: str, test_main: str, num_tests: int, language: str = "c") -> Dict:
    """Returns compiled, passed, total, pass_rate and feedback (compile error or first failure)."""
    result = {"compiled": False, "passed": 0, "total": num_tests, "pass_rate": 0.0, "feedback": ""}
    if not code:
        result["feedback"] = "No code block found in the answer."
        return result
    success, exec_path, error = compile_with_test_main(code, test_main, language)
    if not success:
        result["feedback"] = "Compilation failed:\n" + _clip(error, 1500)
        return result
    try:
        failures = [msg for passed, msg in (run_test_index(exec_path, i) for i in range(num_tests)) if not passed]
    finally:
        shutil.rmtree(os.path.dirname(exec_path), ignore_errors=True)
    passed = num_tests - len(failures)
    result.update(compiled=True, passed=passed, pass_rate=passed / num_tests if num_tests else 0.0,
                  feedback=failures[0] if failures else f"All {num_tests} tests passed.")
    return result



def run_frama_c_wp(file_path: str, timeout: float = None) -> Tuple[str, str]:
    """Runs `frama-c -wp -wp-rte`. Returns (stdout, status), status "ok", "timeout" or "error"."""
    timeout = FRAMA_C_TIMEOUT if timeout is None else timeout
    try:
        result = subprocess.run(["frama-c", "-wp", "-wp-rte", file_path],
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.info(f"Frama-C timed out after {timeout}s on {file_path}")
        return "", "timeout"
    except Exception as e:
        logger.warning(f"Frama-C could not be run ({type(e).__name__}: {e})")
        return "", "error"
    if "Proved goals" in result.stdout:
        return result.stdout, "ok"
    logger.info(f"Frama-C produced no WP summary (exit code {result.returncode}): {result.stderr.strip()[-500:]}")
    return result.stdout + result.stderr, "error"


def run_frama_c_on_code(code: str, language: str = "c") -> Tuple[str, str]:
    suffix = _toolchain(normalize_language(language))[2]
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(STANDARD_HEADERS + code)
    try:
        return run_frama_c_wp(tmp.name)
    finally:
        os.remove(tmp.name)


def parse_frama_c_wp_to_reward(output: str) -> float:
    """proved / total from "[wp] Proved goals: X / Y"; 0.0 if missing or Y == 0."""
    match = re.search(r"\[wp\]\s+Proved goals:\s+(\d+)\s*/\s*(\d+)", output)
    if not match or int(match.group(2)) == 0:
        return 0.0
    return int(match.group(1)) / int(match.group(2))


def frama_c_proof_fraction(output: str, status: str) -> float:
    return FRAMA_C_TIMEOUT_CREDIT if status == "timeout" else parse_frama_c_wp_to_reward(output)


# Reward: compiling, + test pass rate, + proved fraction once all tests pass.
COMPILE_FLOOR = 0.2
TEST_WEIGHT = 0.3
PROOF_WEIGHT = 0.5


def _unproved_goals(wp_output: str) -> str:
    lines = [line for line in wp_output.splitlines()
             if "Goal" in line and any(s in line.lower() for s in ("not tried", "unknown", "timeout"))]
    return "\n".join(lines) or wp_output.strip()[-500:] or "(Frama-C produced no output.)"


def _frama_c_errors(output: str) -> str:
    lines = [re.sub(r"\S*/tmp\w*\.c", "solution.c", line) for line in output.splitlines()
             if line.strip() and not line.startswith(("[kernel] Parsing", "[kernel] User Error: warning annot-error"))]
    return "\n".join(lines)[-1000:] or output.strip()[-500:] or "(Frama-C gave no error message.)"


MAX_FEEDBACK_CHARS = 2000


def score_with_feedback(response: str, test_main: str, num_tests: int, language: str = "c") -> Tuple[float, str]:
    """The training reward in [0, 1] and the feedback shown to the model (empty when fully proved)."""
    score, feedback = _score_with_feedback(response, test_main, num_tests, normalize_language(language))
    return score, _clip(feedback, MAX_FEEDBACK_CHARS)


def _score_with_feedback(response: str, test_main: str, num_tests: int, language: str) -> Tuple[float, str]:
    code, acsl = extract_code_from_response(response)
    if not code:
        return 0.0, "No code found in your response. Put the function in one fenced code block."

    tests = run_function_tests(code, test_main, num_tests, language)
    if not tests["compiled"]:
        return 0.0, ("Your function does not compile together with the test program. "
                     f"Implement exactly the given signature and do not write main().\n{tests['feedback']}")
    score = COMPILE_FLOOR + TEST_WEIGHT * tests["pass_rate"]
    if tests["pass_rate"] < 1.0:
        return score, (f"Passed {tests['passed']}/{tests['total']} test cases. First failure: "
                       f"{tests['feedback']}\nFix the implementation first; the contract is only "
                       "verified once all tests pass.")
    if not response_contains_acsl_annotation(acsl):
        return score, ("All tests pass, but the code has no ACSL annotations. Add a full contract "
                       "(requires/assigns/ensures) and loop annotations.")

    wp_output, wp_status = run_frama_c_on_code(code, language)
    proof = frama_c_proof_fraction(wp_output, wp_status)
    score += PROOF_WEIGHT * proof
    if wp_status == "timeout":
        return score, (f"All tests pass, but verification timed out after {FRAMA_C_TIMEOUT:.0f} seconds, "
                       "so the contract was neither proved nor refuted. Make the proof easier: state "
                       "exactly what is needed in requires/ensures, give each loop precise invariants, "
                       "and avoid non-linear arithmetic in annotations.")
    if wp_status == "error":
        return score, ("All tests pass, but Frama-C rejected the code or its annotations, so nothing "
                       f"could be proved. Fix these errors:\n{_frama_c_errors(wp_output)}")
    if proof >= 0.999:
        return score, ""
    return score, (f"All tests pass, but Frama-C could only prove {proof:.0%} of the proof "
                   f"obligations. Unproved goals:\n{_unproved_goals(wp_output)}")