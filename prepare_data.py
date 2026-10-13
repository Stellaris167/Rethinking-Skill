"""Prepare fixed evaluation inputs from the original benchmark files.

Example:
    python prepare_data.py --benchmark hotpotqa \\
        --source path/to/validation.parquet --output data/hotpotqa.json

For BigCodeBench, install requirements.txt and prepare its NLTK corpora:
    python -m nltk.downloader stopwords words
"""
import argparse
import json
from pathlib import Path

from evaluation.data import load_items, prepare_items

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=["hotpotqa", "dapo_math", "bigcodebench", "alfworld"], required=True)
    parser.add_argument("--source", type=Path, required=True, help="Raw parquet/JSON file; ALFWorld data directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = Path(__file__).parent / "data"
    if args.benchmark == "alfworld":
        items = json.loads((data / "alfworld_test.json").read_text())
        for item in items:
            if not (args.source / item["gamefile"]).is_file():
                raise FileNotFoundError(args.source / item["gamefile"])
    else:
        items = prepare_items(args.benchmark, load_items(args.source))
        by_id = {str(item["id"]): item for item in items}
        ids = (data / f"{args.benchmark}_test_ids.txt").read_text().splitlines()
        items = [by_id[item_id] for item_id in ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.benchmark}: {len(items)} items -> {args.output}")

if __name__ == "__main__":
    main()
