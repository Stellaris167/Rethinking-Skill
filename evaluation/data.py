import ast
import copy
import json
from pathlib import Path

def split_bigcodebench_tests(test_source: str, public_limit: int = 2) -> tuple[str, str]:
    source = str(test_source).strip()
    if not source:
        return "", ""
    tree = ast.parse(source)
    test_names = _test_method_names(tree)
    if len(test_names) < 2:
        return "", source
    n_public = min(max(1, int(public_limit)), len(test_names) - 1)
    public_names = set(test_names[:n_public])
    private_names = set(test_names[n_public:])
    public_source = _select_unittest_methods(tree, public_names)
    private_source = _select_unittest_methods(tree, private_names)
    if not public_source or not private_source:
        return "", source
    return public_source, private_source

def _test_method_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                names.append(child.name)
    return names

def _select_unittest_methods(tree: ast.Module, selected_names: set[str]) -> str:
    module = copy.deepcopy(tree)
    body: list[ast.stmt] = []
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            body.append(node)
            continue
        class_test_names = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test")
        ]
        if not class_test_names:
            body.append(node)
            continue
        selected_in_class = set(class_test_names) & selected_names
        if not selected_in_class:
            continue
        node.body = [
            child
            for child in node.body
            if not (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test")
                and child.name not in selected_names
            )
        ]
        body.append(node)
    module.body = body
    ast.fix_missing_locations(module)
    return ast.unparse(module).strip()

def format_hotpot_context(context: dict) -> str:
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    chunks = []
    for title, sent_group in zip(titles, sentences):
        sent_text = " ".join(str(sentence).strip() for sentence in sent_group if str(sentence).strip())
        chunks.append(f"[DOC] [TLE] {str(title)} [PAR] {sent_text}")
    return " ".join(chunks)

def load_items(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())

def prepare_items(benchmark: str, rows: list[dict]) -> list[dict]:
    items = []
    for index, row in enumerate(rows):
        if benchmark == "hotpotqa":
            items.append({"id": row["id"], "question": row["question"],
                          "context": format_hotpot_context(row["context"]),
                          "answer": row["answer"], "answers_json": json.dumps([row["answer"]])})
        elif benchmark == "dapo_math":
            answer = str(row["reward_model"]["ground_truth"])
            items.append({"id": str(row["extra_info"].get("index") or f"dapo_math_{index}"),
                          "question": row["prompt"], "answer": answer, "answers_json": json.dumps([answer])})
        elif benchmark == "bigcodebench":
            public, private = split_bigcodebench_tests(row["test"])
            items.append({"id": row["task_id"], "question": row["instruct_prompt"],
                          "instruct_prompt": row["instruct_prompt"], "code_prompt": row["code_prompt"],
                          "entry_point": row["entry_point"],
                          "tests_json": json.dumps({"test": private, "entry_point": row["entry_point"]}),
                          "public_tests_json": json.dumps({"test": public})})
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    return items
