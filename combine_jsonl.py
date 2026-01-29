#!/usr/bin/env python3
"""Concatenate multiple JSONL files into a single output file."""

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine JSONL files from a directory into one file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the source JSONL files (default: data).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=["primevul_train.jsonl", "primevul_valid.jsonl", "primevul_test.jsonl"],
        help="Specific JSONL filenames to merge from the input directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined.jsonl"),
        help="Path to write the merged JSONL (default: combined.jsonl).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path.exists():
        print(f"Refusing to overwrite existing file: {output_path}", file=sys.stderr)
        return 1

    files_to_merge = []
    for name in args.files:
        path = input_dir / name
        if not path.is_file():
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        files_to_merge.append(path)

    if not files_to_merge:
        print("No input JSONL files found to merge.", file=sys.stderr)
        return 1

    with output_path.open("w", encoding="utf-8") as out_f:
        for source_path in files_to_merge:
            with source_path.open("r", encoding="utf-8") as src_f:
                for line in src_f:
                    if line.strip():
                        out_f.write(line.rstrip("\n") + "\n")

    print(f"Wrote {output_path} from {len(files_to_merge)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
