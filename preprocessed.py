"""
PrimeVul (JSONL) preprocessing for vulnerability detection.

Input JSONL fields expected (per line):
  - func:   string (C/C++ function code)
  - target: int (0/1)
  - project:string (project name / grouping key)

Outputs:
  out_dir/
    train.jsonl
    val.jsonl
    test.jsonl
Each output line includes:
  - id
  - project
  - target
  - func_clean
  - prompt
  - answer_text   ("safe" or "vulnerable")

Default preprocessing:
  - project-level split (no project overlap between splits)
  - unicode cleanup + newline normalization
  - strip C/C++ comments
  - mask function name (optional flag)
  - length control via head+tail truncation (character-based by default)
  - prompt formatting for CodeLLaMA

Usage:
  python preprocess_primevul.py \
    --input /mnt/data/primevul_train.jsonl \
    --out_dir /mnt/data/primevul_preprocessed \
    --test_size 0.2 --val_size 0.2 \
    --max_chars 8000 \
    --mask_func_name

If you want token-based truncation (more accurate), also pass:
  --tokenizer meta-llama/CodeLlama-7b-hf --max_tokens 2048
(Requires: transformers, sentencepiece)
"""

import argparse
import hashlib
import json
import os
import random
import re
from typing import Dict, List, Tuple, Optional

# ---------- Cleaning helpers ----------

NULL_BYTE_RE = re.compile(r"\x00")
# crude but effective C/C++ comment stripper (keeps strings reasonably intact for most cases)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
LINE_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)

# mask function name at definition:
# handles common cases like:
#   static inline int foo_bar(...) {
#   void* baz(...) {
FUNC_DEF_RE = re.compile(
    r"^(\s*(?:[A-Za-z_]\w*\s+)*"       # return type-ish tokens
    r"(?:\*+\s*)?)"                    # pointer stars
    r"([A-Za-z_]\w*)"                  # function name
    r"(\s*\([^;]*\)\s*\{)",            # params then opening brace
    flags=re.MULTILINE
)

def clean_text(s: str) -> str:
    s = s if isinstance(s, str) else ""
    s = NULL_BYTE_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # best-effort: remove broken unicode surrogates
    s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return s.strip()

def strip_comments(code: str) -> str:
    code = BLOCK_COMMENT_RE.sub("", code)
    code = LINE_COMMENT_RE.sub("", code)
    return code

def mask_function_name(code: str) -> str:
    # replace only the first definition-like occurrence
    return FUNC_DEF_RE.sub(r"\1FUNC\3", code, count=1)

def head_tail_truncate_chars(code: str, max_chars: int, head_ratio: float = 0.6) -> str:
    if max_chars <= 0 or len(code) <= max_chars:
        return code
    head_len = int(max_chars * head_ratio)
    tail_len = max_chars - head_len
    return code[:head_len] + "\n/* ...TRUNCATED... */\n" + code[-tail_len:]

def make_prompt(code: str) -> str:
    # Simple, stable prompt (works for SFT / instruction-tuning style classification)
    return (
        "You are a security code reviewer.\n"
        "Decide whether the following C/C++ function is vulnerable.\n\n"
        "<CODE>\n"
        f"{code}\n"
        "</CODE>\n\n"
        "Answer with exactly one word: safe or vulnerable.\n"
        "Answer:"
    )

def label_to_text(y: int) -> str:
    return "vulnerable" if int(y) == 1 else "safe"

# ---------- Splitting helpers (project-level) ----------

def project_split(
    rows: List[Dict],
    test_size: float,
    val_size: float,
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Project-level split:
      - test projects are held out
      - val projects are held out from remaining train projects
    """
    rng = random.Random(seed)

    # gather projects
    proj_to_rows: Dict[str, List[Dict]] = {}
    for r in rows:
        p = r.get("project", "")
        proj_to_rows.setdefault(p, []).append(r)

    projects = list(proj_to_rows.keys())
    rng.shuffle(projects)

    n_projects = len(projects)
    n_test = max(1, int(round(n_projects * test_size)))
    test_projects = set(projects[:n_test])

    remaining = projects[n_test:]
    n_val = max(1, int(round(len(remaining) * val_size)))
    val_projects = set(remaining[:n_val])

    train_projects = set(remaining[n_val:])

    train, val, test = [], [], []
    for p, rs in proj_to_rows.items():
        if p in test_projects:
            test.extend(rs)
        elif p in val_projects:
            val.extend(rs)
        else:
            train.extend(rs)

    # sanity: no overlap
    assert train_projects.isdisjoint(val_projects)
    assert train_projects.isdisjoint(test_projects)
    assert val_projects.isdisjoint(test_projects)

    return train, val, test

# ---------- Optional tokenizer-based truncation ----------

def maybe_load_tokenizer(tokenizer_name: Optional[str]):
    if not tokenizer_name:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        return tok
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_name}'. "
            f"Install transformers+sentencepiece and ensure the model name/path is correct.\n"
            f"Error: {e}"
        )

def head_tail_truncate_tokens(tok, code: str, max_tokens: int, head_ratio: float = 0.6) -> str:
    if max_tokens <= 0:
        return code
    ids = tok.encode(code, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return code
    head_len = int(max_tokens * head_ratio)
    tail_len = max_tokens - head_len
    kept = ids[:head_len] + ids[-tail_len:]
    return tok.decode(kept, skip_special_tokens=True)

# ---------- IO ----------

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    print(f"Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip broken line
                continue
    return rows

def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stable_id(project: str, func: str) -> str:
    h = hashlib.sha256((project + "\n" + func).encode("utf-8", errors="ignore")).hexdigest()
    return h[:16]

def summarize_split(name: str, rows: List[Dict]) -> None:
    y = [int(r["target"]) for r in rows]
    pos = sum(y)
    neg = len(y) - pos
    projs = len(set(r["project"] for r in rows))
    print(f"{name:5s}: n={len(rows):6d}  pos={pos:6d}  neg={neg:6d}  projects={projs:4d}")

# ---------- Main preprocessing ----------

def preprocess_rows(
    rows: List[Dict],
    *,
    strip_c_comments: bool,
    do_mask_func_name: bool,
    max_chars: int,
    tokenizer=None,
    max_tokens: int = 0,
) -> List[Dict]:
    out = []
    for r in rows:
        func = clean_text(r.get("func", ""))
        project = str(r.get("project", "")).strip()
        target = r.get("target", None)

        # validate essentials
        if target is None:
            continue
        try:
            target = int(target)
        except Exception:
            continue
        if target not in (0, 1):
            continue
        if not func or len(func) < 30:
            continue
        if not project:
            project = "UNKNOWN"

        if strip_c_comments:
            func = strip_comments(func)
            func = clean_text(func)

        if do_mask_func_name:
            func = mask_function_name(func)

        # length control
        if tokenizer is not None and max_tokens > 0:
            func = head_tail_truncate_tokens(tokenizer, func, max_tokens=max_tokens)
            func = clean_text(func)
        else:
            func = head_tail_truncate_chars(func, max_chars=max_chars)

        pid = stable_id(project, func)
        prompt = make_prompt(func)
        answer = label_to_text(target)

        out.append({
            "id": pid,
            "project": project,
            "target": target,
            "func_clean": func,
            "prompt": prompt,
            "answer_text": answer
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to PrimeVul JSONL (func/target/project).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)  # fraction of remaining projects
    ap.add_argument("--strip_comments", action="store_true", help="Strip // and /* */ comments.")
    ap.add_argument("--mask_func_name", action="store_true", help="Mask function name at definition to FUNC.")
    ap.add_argument("--max_chars", type=int, default=8000, help="Max characters if not using tokenizer.")
    ap.add_argument("--tokenizer", type=str, default="microsoft/unixcoder-base", help="Tokenizer name/path for token-based truncation.")
    ap.add_argument("--max_tokens", type=int, default=2048, help="Max tokens if tokenizer is used (e.g., 1024/2048).")
    args = ap.parse_args()
    folder_path = os.path.dirname(args.out_dir)
    os.makedirs(folder_path, exist_ok=True)
    input_file = args.input
    rows = load_jsonl(input_file)
    if not rows:
        raise SystemExit(f"No valid rows loaded from {args.input}")

    # split first (avoid leakage through preprocessing/dedup ordering)
    train_raw, val_raw, test_raw = project_split(
        rows, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    tok = maybe_load_tokenizer(args.tokenizer)

    train = preprocess_rows(
        train_raw,
        strip_c_comments=args.strip_comments,
        do_mask_func_name=args.mask_func_name,
        max_chars=args.max_chars,
        tokenizer=tok,
        max_tokens=args.max_tokens,
    )
    val = preprocess_rows(
        val_raw,
        strip_c_comments=args.strip_comments,
        do_mask_func_name=args.mask_func_name,
        max_chars=args.max_chars,
        tokenizer=tok,
        max_tokens=args.max_tokens,
    )
    test = preprocess_rows(
        test_raw,
        strip_c_comments=args.strip_comments,
        do_mask_func_name=args.mask_func_name,
        max_chars=args.max_chars,
        tokenizer=tok,
        max_tokens=args.max_tokens,
    )

    # write outputs
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    print("\n=== Split summary (after preprocessing) ===")
    summarize_split("train", train)
    summarize_split("val", val)
    summarize_split("test", test)

    # verify project disjointness
    trp = set(r["project"] for r in train)
    vap = set(r["project"] for r in val)
    tep = set(r["project"] for r in test)
    assert trp.isdisjoint(vap) and trp.isdisjoint(tep) and vap.isdisjoint(tep)
    print("\nOK: project-level splits are disjoint.")
    print(f"Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()


# python preprocessed.py \
#   --input ./dataset/primevul_train.jsonl \
#   --out_dir ./data \                      
#   --test_size 0.2 --val_size 0.2 \
#   --strip_comments \
#   --mask_func_name \
#   --tokenizer codellama/CodeLlama-7b-hf \
#   --max_tokens 2048

# === Split summary (after preprocessing) ===
# train: n=110878  pos=  3214  neg=107664  projects= 482
# val  : n= 87122  pos=  2059  neg= 85063  projects= 121
# test : n= 26300  pos=   731  neg= 25569  projects= 151