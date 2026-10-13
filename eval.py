"""Evaluate a model with a skill file or folder against an OpenAI-compatible server.

Example:
    python eval.py --benchmark hotpotqa --data data/hotpotqa.json \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-27B \\
        --skill path/to/skill.md --output results/hotpotqa

Omit --skill to use the benchmark's basic prompt. API credentials are read from
OPENAI_API_KEY; an optional answer judge uses JUDGE_API_KEY.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from evaluation.benchmarks import evaluate_item
from evaluation.data import load_items
from evaluation.model import ModelConfig, configure, is_gemma4_model
from evaluation.skills import load_skill, render_skill

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=["hotpotqa", "dapo_math", "bigcodebench", "alfworld"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["direct", "react"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit", type=int, help="Number of questions for a smoke test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--test-timeout", type=int, default=30)
    parser.add_argument("--alfworld-data", type=Path)
    parser.add_argument("--judge-url")
    parser.add_argument("--judge-model")
    args = parser.parse_args()
    root = Path(__file__).parent
    defaults = json.loads((root / "configs/evaluation.json").read_text())[args.benchmark]
    for name in ("mode", "samples", "max_tokens", "max_steps"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    if args.samples < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("samples, workers and limit must be positive")
    if bool(args.judge_url) != bool(args.judge_model):
        parser.error("--judge-url and --judge-model must be provided together")
    if args.benchmark == "alfworld" and (args.mode != "react" or args.alfworld_data is None):
        parser.error("ALFWorld requires --mode react and --alfworld-data")
    gemma = is_gemma4_model(args.model)
    config = ModelConfig(
        base_url=args.base_url, deployment=args.model, api_key=os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=args.max_tokens, timeout_seconds=args.timeout,
        temperature=args.temperature if args.temperature is not None else (1.0 if gemma else 0.6),
        top_p=args.top_p, top_k=args.top_k if args.top_k is not None else (64 if gemma else 20),
        enable_thinking=args.thinking, seed=args.seed)
    configure(config)
    skill = load_skill(args.skill or root / "prompts" / defaults["prompt"])
    items = load_items(args.data)
    if args.limit is not None:
        items = items[:args.limit]
    if not items or len({str(item["id"]) for item in items}) != len(items):
        raise ValueError("Evaluation requires nonempty inputs with unique IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output / "results.jsonl").open("x") as handle:
        for sample in range(args.samples):
            seed = args.seed + sample
            config.seed = seed
            configure(config)
            output = args.output / f"sample_{sample + 1}"
            if args.benchmark == "alfworld":
                from evaluation.alfworld import build_alfworld_env, run_alfworld_batch
                os.environ["ALFWORLD_DATA"] = str(args.alfworld_data.resolve())
                def run_batches():
                    for start in range(0, len(items), args.workers):
                        batch = items[start:start + args.workers]
                        env = build_alfworld_env(len(batch), seed=seed,
                            specific_gamefiles=[item["gamefile"] for item in batch])
                        try:
                            yield from run_alfworld_batch(env, render_skill(skill),
                                max_steps=args.max_steps, out_root=str(output), max_api_workers=args.workers,
                                max_completion_tokens=args.max_tokens, result_ids=[item["id"] for item in batch],
                                react_history=True, llm_timeout=args.timeout)
                        finally:
                            env.close()
                rows = run_batches()
                for row in rows:
                    row.update(sample=sample + 1, seed=seed)
                    results.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[{len(results)}/{len(items) * args.samples}] {row['id']}: {row['hard']}", flush=True)
            else:
                def run(item):
                    return evaluate_item(item, skill, benchmark=args.benchmark, mode=args.mode,
                        output=output, max_tokens=args.max_tokens, max_steps=args.max_steps, seed=seed,
                        timeout=args.test_timeout, judge_url=args.judge_url, judge_model=args.judge_model)
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for row in pool.map(run, items):
                        row.update(sample=sample + 1, seed=seed)
                        results.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        print(f"[{len(results)}/{len(items) * args.samples}] {row['id']}: {row['hard']}", flush=True)
    summary = {"benchmark": args.benchmark, "model": args.model, "questions": len(items),
               "samples": args.samples, "accuracy": sum(row["hard"] for row in results) / len(results),
               "soft": sum(row["soft"] for row in results) / len(results)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
