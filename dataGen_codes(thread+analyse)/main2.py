import json
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =========================
# Paths
# =========================

PROMPT_FILE = Path("prompt.txt")
ENHANCEMENT_DIR = Path("enhancements")
RESULTS_DIR = Path("results")
USED_PROMPTS_DIR = Path("used_prompts")

PLACEHOLDER = "[VARIANCE ENHANCEMENT BLOCK HERE]"

RESULTS_DIR.mkdir(exist_ok=True)
USED_PROMPTS_DIR.mkdir(exist_ok=True)

_filename_lock = threading.Lock()


# =========================
# Usage tracking
# =========================

class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

    def add(self, usage: dict, cost_usd: float = 0.0):
        with self._lock:
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)
            self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
            self.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
            self.total_cost_usd += cost_usd
            self.call_count += 1

    def report(self, prefix: str = "  ") -> str:
        total_tokens = self.total_input_tokens + self.total_output_tokens
        lines = [
            "Usage Summary:",
            f"{prefix}Claude calls    : {self.call_count}",
            f"{prefix}Input tokens    : {self.total_input_tokens:,}",
            f"{prefix}Output tokens   : {self.total_output_tokens:,}",
            f"{prefix}Total tokens    : {total_tokens:,}",
        ]
        if self.total_cache_read_tokens or self.total_cache_creation_tokens:
            lines.append(f"{prefix}Cache read      : {self.total_cache_read_tokens:,}")
            lines.append(f"{prefix}Cache creation  : {self.total_cache_creation_tokens:,}")
        if self.total_cost_usd > 0:
            lines.append(f"{prefix}Cost (USD)      : ${self.total_cost_usd:.6f}")
        return "\n".join(lines)


# =========================
# Helper functions
# =========================

def read_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    if PLACEHOLDER not in prompt_template:
        raise RuntimeError(
            f"Placeholder not found in {PROMPT_FILE}.\n"
            f"Expected placeholder: {PLACEHOLDER}"
        )

    return prompt_template


def read_enhancement_blocks(enhancement_names: list[str]) -> str:
    blocks = []

    for name in enhancement_names:
        enhancement_file = ENHANCEMENT_DIR / f"{name}.txt"

        if not enhancement_file.exists():
            raise FileNotFoundError(
                f"Enhancement file not found: {enhancement_file}"
            )

        text = enhancement_file.read_text(encoding="utf-8").strip()

        block = f"""
### ENHANCEMENT: {name}

{text}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(
    prompt_template: str,
    enhancement_names: list[str]
) -> str:
    enhancement_block = read_enhancement_blocks(enhancement_names)

    final_prompt = prompt_template.replace(
        PLACEHOLDER,
        enhancement_block
    )

    return final_prompt.strip()


def sanitize_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def get_next_result_path(task_id: int, task_name: str) -> Path:
    safe_name = sanitize_filename(task_name)

    pattern = re.compile(
        rf"task_{task_id}_{re.escape(safe_name)}_(\d+)\.json$"
    )

    existing_numbers = []

    for file in RESULTS_DIR.glob(f"task_{task_id}_{safe_name}_*.json"):
        match = pattern.match(file.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1

    return RESULTS_DIR / f"task_{task_id}_{safe_name}_{next_number}.json"


def get_used_prompt_path(
    task_id: int,
    task_name: str,
    run_number: int
) -> Path:
    safe_name = sanitize_filename(task_name)
    return USED_PROMPTS_DIR / f"task_{task_id}_{safe_name}_{run_number}.txt"


def extract_run_number(path: Path) -> int:
    match = re.search(r"_(\d+)\.json$", path.name)

    if not match:
        return 1

    return int(match.group(1))


def run_claude(
    final_prompt: str,
    check_usage: bool = False
) -> tuple[int, str, str, dict | None]:
    """Returns (returncode, stdout_text, stderr_text, usage_info_or_None)."""
    claude = shutil.which("claude-saka")

    if not claude:
        raise RuntimeError(
            "Claude CLI not found. Please check whether 'claude' is in PATH."
        )

    cmd = [claude, "-p"]
    if check_usage:
        cmd += ["--output-format", "json"]

    result = subprocess.run(
        cmd,
        input=final_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    stdout = result.stdout
    usage_info = None

    if check_usage and result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            raw_usage = data.get("usage", {})
            usage_info = {
                "input_tokens": raw_usage.get("input_tokens", 0),
                "output_tokens": raw_usage.get("output_tokens", 0),
                "cache_read_input_tokens": raw_usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": raw_usage.get("cache_creation_input_tokens", 0),
                "cost_usd": data.get("cost_usd", 0.0),
            }
            stdout = data.get("result", result.stdout)
        except (json.JSONDecodeError, AttributeError):
            pass

    return result.returncode, stdout, result.stderr, usage_info


def run_task(
    task: dict,
    prompt_template: str,
    runs: int = 1,
    check_usage: bool = False,
    tracker: UsageTracker | None = None,
):
    task_id = task["task_id"]
    task_name = task.get("task_name", f"task_{task_id}")
    enhancement_names = task["enhancements"]
    prefix = f"[task {task_id}]"

    print(f"{prefix} Starting: {task_name}")
    print(f"{prefix} Enhancements: {enhancement_names}")

    final_prompt = build_prompt(
        prompt_template=prompt_template,
        enhancement_names=enhancement_names
    )

    for run_idx in range(runs):
        run_label = f"run {run_idx + 1}/{runs} " if runs > 1 else ""

        with _filename_lock:
            result_path = get_next_result_path(task_id, task_name)
            run_number = extract_run_number(result_path)
            # Touch the file so concurrent tasks don't claim the same number
            result_path.write_text("", encoding="utf-8")

        used_prompt_path = get_used_prompt_path(
            task_id=task_id,
            task_name=task_name,
            run_number=run_number
        )

        used_prompt_path.write_text(final_prompt, encoding="utf-8")

        print(f"{prefix} {run_label}Calling Claude...")
        returncode, stdout, stderr, usage_info = run_claude(
            final_prompt, check_usage=check_usage
        )

        result_path.write_text(stdout, encoding="utf-8")

        if stderr.strip():
            stderr_path = result_path.with_suffix(".stderr.txt")
            stderr_path.write_text(stderr, encoding="utf-8")
            print(f"{prefix} STDERR saved: {stderr_path}")

        print(f"{prefix} {run_label}Result saved: {result_path}")
        print(f"{prefix} {run_label}Used prompt saved: {used_prompt_path}")
        print(f"{prefix} {run_label}Return code: {returncode}")

        if check_usage and usage_info:
            in_tok = usage_info["input_tokens"]
            out_tok = usage_info["output_tokens"]
            cost = usage_info["cost_usd"]
            total_tok = in_tok + out_tok
            print(
                f"{prefix} {run_label}Tokens: {in_tok:,} in + {out_tok:,} out"
                f" = {total_tok:,} total | Cost: ${cost:.6f}"
            )
            if tracker:
                tracker.add(usage_info, cost_usd=cost)


# =========================
# Main
# =========================

def main():
    args = sys.argv[1:]

    if not args or args[0].startswith("-"):
        print("Usage:")
        print("  python main2.py tasks_semih.json")
        print("  python main2.py tasks_semih.json --workers 2")
        print("  python main2.py tasks_semih.json --runs 3")
        print("  python main2.py tasks_semih.json --check-usage")
        sys.exit(1)

    task_file = Path(args[0])
    max_workers = None
    runs = 1
    check_usage = False

    if "--workers" in args:
        idx = args.index("--workers")
        try:
            max_workers = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --workers requires an integer value.")
            sys.exit(1)

    if "--runs" in args:
        idx = args.index("--runs")
        try:
            runs = int(args[idx + 1])
            if runs < 1:
                raise ValueError("must be >= 1")
        except (IndexError, ValueError):
            print("Error: --runs requires a positive integer value.")
            sys.exit(1)

    if "--check-usage" in args:
        check_usage = True

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    prompt_template = read_prompt_template()

    tracker = UsageTracker() if check_usage else None

    workers = max_workers or len(tasks)
    total_calls = len(tasks) * runs
    print(
        f"Loaded {len(tasks)} tasks from {task_file}"
        f" — {runs} run(s) each — {total_calls} total Claude call(s)"
        f" — {workers} worker(s)"
    )

    failures = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_task, task, prompt_template, runs, check_usage, tracker
            ): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            task_id = task["task_id"]
            try:
                future.result()
            except Exception as exc:
                failures.append((task_id, exc))
                print(f"[task {task_id}] FAILED: {exc}")

    print("\n" + "=" * 60)

    if check_usage and tracker:
        print(tracker.report())
        print("=" * 60)

    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for task_id, exc in failures:
            print(f"  task {task_id}: {exc}")
    else:
        print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
