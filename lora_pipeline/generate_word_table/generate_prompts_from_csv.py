"""
python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /你的/输入.csv \
  --output_txt /你的/输出.txt \
  --num_prompts 100 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 1 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore
"""
import argparse
import csv
import os
import random
from typing import List, Set, Tuple


def load_columns_from_csv(input_csv: str, delimiter: str = ",", has_header: bool = False) -> List[List[str]]:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 为空，无法采样。")

    if has_header and len(rows) > 1:
        rows = rows[1:]
    elif has_header and len(rows) == 1:
        raise ValueError("CSV 只有表头，没有可采样内容。")

    max_cols = max(len(row) for row in rows)
    columns: List[List[str]] = [[] for _ in range(max_cols)]

    for row in rows:
        for col_idx in range(max_cols):
            value = row[col_idx].strip() if col_idx < len(row) else ""
            if value:
                columns[col_idx].append(value)

    columns = [col for col in columns if col]
    if not columns:
        raise ValueError("所有列都为空，无法采样。")

    return columns


def normalize_prompt(prompt: str) -> str:
    parts = [x.strip() for x in str(prompt).split(",")]
    parts = [x for x in parts if x]
    return ", ".join(parts)


def load_existing_prompts(output_txt: str) -> Tuple[List[str], Set[str]]:
    if not os.path.exists(output_txt):
        return [], set()

    existing_prompts: List[str] = []
    seen: Set[str] = set()
    with open(output_txt, "r", encoding="utf-8") as f:
        for line in f:
            text = normalize_prompt(line.strip())
            if text:
                if text in seen:
                    continue
                seen.add(text)
                existing_prompts.append(text)
    return existing_prompts, seen


def sample_single_prompt(
    columns: List[List[str]],
    min_columns: int,
    max_columns: int,
    min_terms_per_column: int,
    max_terms_per_column: int,
    replace_space_with_underscore: bool = False,
    rng: random.Random = random,
) -> str:
    total_cols = len(columns)
    col_pick_count = rng.randint(min_columns, max_columns)
    picked_col_indices = rng.sample(range(total_cols), k=col_pick_count)

    parts: List[str] = []
    for col_idx in picked_col_indices:
        col_values = columns[col_idx]
        upper = min(max_terms_per_column, len(col_values))
        lower = min(min_terms_per_column, upper)
        term_count = rng.randint(lower, upper)
        if term_count > 0:
            terms = rng.sample(col_values, k=term_count)
            if replace_space_with_underscore:
                terms = [t.replace(" ", "_") for t in terms]
            parts.extend(terms)

    return normalize_prompt(", ".join(parts))


def sample_non_empty_prompt(
    columns: List[List[str]],
    min_columns: int,
    max_columns: int,
    min_terms_per_column: int,
    max_terms_per_column: int,
    replace_space_with_underscore: bool = False,
    max_retries: int = 30,
    rng: random.Random = random,
) -> str:
    for _ in range(max_retries):
        prompt = sample_single_prompt(
            columns=columns,
            min_columns=min_columns,
            max_columns=max_columns,
            min_terms_per_column=min_terms_per_column,
            max_terms_per_column=max_terms_per_column,
            replace_space_with_underscore=replace_space_with_underscore,
            rng=rng,
        )
        if prompt:
            return prompt

    fallback_col = rng.choice(columns)
    fallback_term = rng.choice(fallback_col)
    if replace_space_with_underscore:
        return normalize_prompt(fallback_term.replace(" ", "_"))
    return normalize_prompt(fallback_term)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 CSV 多列词组中随机采样生成 prompt 文本，并自动去重。")
    parser.add_argument("--input_csv", type=str, required=True, help="输入 CSV 路径。")
    parser.add_argument("--output_txt", type=str, required=True, help="输出 TXT 路径。")
    parser.add_argument("--num_prompts", type=int, required=True, help="最终希望 TXT 中拥有的 prompt 总条数。")
    parser.add_argument("--min_columns", type=int, default=1, help="每条 prompt 最少采样列数。")
    parser.add_argument("--max_columns", type=int, default=None, help="每条 prompt 最多采样列数。默认等于总列数。")
    parser.add_argument(
        "--min_terms_per_column",
        type=int,
        default=0,
        help="每个被选中列最少采样词组数。",
    )
    parser.add_argument(
        "--max_terms_per_column",
        type=int,
        default=1,
        help="每个被选中列最多采样词组数。",
    )
    parser.add_argument("--delimiter", type=str, default=",", help="CSV 分隔符，默认逗号。")
    parser.add_argument("--has_header", action="store_true", help="如果 CSV 第一行是表头则开启。")
    parser.add_argument(
        "--replace-space-with-underscore",
        action="store_true",
        help="开启后，将采样到的词组中的空格替换为下划线。",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现。")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=100000,
        help="最大采样尝试次数，防止去重后组合不足导致死循环。",
    )
    return parser


def validate_args(args: argparse.Namespace, total_columns: int) -> None:
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts 必须大于 0。")
    if args.min_columns < 0:
        raise ValueError("--min_columns 不能小于 0。")
    if args.min_terms_per_column < 0:
        raise ValueError("--min_terms_per_column 不能小于 0。")
    if args.max_terms_per_column < 0:
        raise ValueError("--max_terms_per_column 不能小于 0。")
    if args.min_terms_per_column > args.max_terms_per_column:
        raise ValueError("--min_terms_per_column 不能大于 --max_terms_per_column。")

    max_columns = args.max_columns if args.max_columns is not None else total_columns
    if max_columns < 0:
        raise ValueError("--max_columns 不能小于 0。")
    if max_columns > total_columns:
        max_columns = total_columns
    if args.min_columns > max_columns:
        raise ValueError("--min_columns 不能大于 --max_columns（或总列数）。")

    args.max_columns = max_columns


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    columns = load_columns_from_csv(
        input_csv=args.input_csv,
        delimiter=args.delimiter,
        has_header=args.has_header,
    )
    validate_args(args, total_columns=len(columns))

    output_dir = os.path.dirname(args.output_txt)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    existing_prompts, seen = load_existing_prompts(args.output_txt)
    prompts: List[str] = list(existing_prompts)

    if len(prompts) > args.num_prompts:
        print(
            f"警告：现有 TXT 中已存在 {len(prompts)} 条 prompt，"
            f"大于指定的 --num_prompts {args.num_prompts}。将保留原文件内容，不做截断。"
        )
        print(f"当前文件路径: {args.output_txt}")
        return

    attempts = 0
    no_new_streak = 0
    while len(prompts) < args.num_prompts:
        if attempts >= args.max_attempts:
            raise RuntimeError(
                f"达到最大尝试次数 {args.max_attempts}，仍未凑齐 {args.num_prompts} 条唯一 prompt。\n"
                f"当前已有 {len(prompts)} 条唯一 prompt。\n"
                f"可能原因：可组合空间不足，或参数限制过严。"
            )

        prompt = sample_non_empty_prompt(
            columns=columns,
            min_columns=args.min_columns,
            max_columns=args.max_columns,
            min_terms_per_column=args.min_terms_per_column,
            max_terms_per_column=args.max_terms_per_column,
            replace_space_with_underscore=args.replace_space_with_underscore,
            rng=rng,
        )
        attempts += 1

        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            no_new_streak = 0
        else:
            no_new_streak += 1
            if no_new_streak >= min(100000, args.max_attempts):
                raise RuntimeError(
                    "连续大量采样均命中历史/重复，组合空间可能接近耗尽；"
                    "请放宽采样参数或降低目标数量。"
                )

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    newly_added = len(prompts) - len(existing_prompts)
    print(
        f"最终共 {len(prompts)} 条唯一 prompt，新增 {newly_added} 条，保存到: {args.output_txt}"
    )


if __name__ == "__main__":
    main()
