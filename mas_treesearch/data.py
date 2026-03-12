from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .profiles import profile_summary, resolve_dataset_profile


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_option_number(letter: str) -> Optional[int]:
    text = letter.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.isalpha():
        return ord(text.upper()) - ord("A") + 1
    return None


def _normalize_yes_no(answer: str) -> str:
    lowered = answer.strip().lower()
    if lowered in {"yes", "y", "true"}:
        return "yes"
    if lowered in {"no", "n", "false"}:
        return "no"
    match = re.search(r"\b(yes|no)\b", lowered)
    if match:
        return match.group(1)
    return lowered


def _normalize_answer(record: Dict[str, Any]) -> str:
    dataset_name = str(record.get("source_dataset", "")).strip()
    answer = str(record.get("answer", "")).strip()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    profile = resolve_dataset_profile(dataset_name)

    if profile.answer_format in {"option", "option_or_option_confidence"}:
        numeric = None
        if metadata and isinstance(metadata.get("answer_index"), int):
            numeric = int(metadata["answer_index"]) + 1
        if numeric is None:
            match = re.search(r"OPTION\s*-\s*(\d+)", answer, flags=re.IGNORECASE)
            if match:
                numeric = int(match.group(1))
        if numeric is None:
            numeric = _extract_option_number(answer)
        if numeric is not None:
            if "CONFIDENCE -" in str(record.get("question", "")).upper():
                confidence_match = re.search(r"CONFIDENCE\s*-\s*(\d+)", answer, flags=re.IGNORECASE)
                confidence = confidence_match.group(1) if confidence_match else "1"
                return f"OPTION - {numeric}\nCONFIDENCE - {confidence}"
            return f"OPTION - {numeric}"
        return answer

    if profile.answer_format == "yes_no":
        return _normalize_yes_no(answer)

    if profile.answer_format == "json_list":
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, list):
                return _compact_json(parsed)
        except Exception:
            pass
        if metadata and isinstance(metadata.get("answer_all"), list):
            return _compact_json(metadata["answer_all"])
        return answer

    return answer


def _format_question(record: Dict[str, Any]) -> str:
    dataset_name = str(record.get("source_dataset", "")).strip()
    question = str(record.get("question", "")).strip()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    profile = resolve_dataset_profile(dataset_name)

    if dataset_name == "mmlu_pro":
        options = metadata.get("options")
        if isinstance(options, list) and options:
            option_lines = [f"{idx}) {str(option).strip()}" for idx, option in enumerate(options, start=1)]
            return (
                f"{question}\n\nOptions:\n"
                + "\n".join(option_lines)
                + "\n\nAnswer with exactly one line: OPTION - <NUMBER>"
            )

    if dataset_name in {"mmlu", "popqa", "cqa"}:
        if "CONFIDENCE -" in question.upper():
            return question + "\n\nRemember: output exactly two lines in the required format."
        return question + "\n\nRemember: output exactly one line in the required format."

    if dataset_name == "gsm8k":
        return question + "\n\nReturn only the final numeric answer."

    if dataset_name == "normad":
        return question + "\n\nAnswer with exactly one token: yes or no."

    if dataset_name == "knowledge_crosswords":
        blanks = metadata.get("blanks")
        options = metadata.get("options")
        lines: List[str] = [question]
        if isinstance(blanks, list) and blanks:
            lines.append("")
            lines.append("Blank order:")
            for idx, blank in enumerate(blanks, start=1):
                lines.append(f"{idx}. {blank}")
        if isinstance(options, dict) and options:
            lines.append("")
            lines.append("Candidate options per blank:")
            for blank in blanks or options.keys():
                values = options.get(blank)
                if isinstance(values, list):
                    lines.append(f"{blank}: {', '.join(str(v) for v in values)}")
        lines.append("")
        lines.append("Return only a JSON array of filled values following the blank order.")
        return "\n".join(lines)

    if dataset_name == "nlgraph":
        return question + "\n\nIf a path exists, output the path clearly. Otherwise output No."

    if dataset_name == "qasper":
        title = str(metadata.get("title", "")).strip()
        abstract = str(metadata.get("abstract", "")).strip()
        parts = []
        if title:
            parts.append(f"Paper title: {title}")
        if abstract:
            parts.append(f"Paper abstract: {abstract}")
        parts.append(f"Question: {question}")
        parts.append("Return a short answer phrase only.")
        return "\n\n".join(parts)

    if dataset_name == "gaia":
        return question + "\n\nFollow the exact output format requested in the question."

    if profile.answer_format == "short_span":
        return question + "\n\nReturn only a short answer span."
    return question


def standardize_record(record: Dict[str, Any], split: str) -> Dict[str, Any]:
    dataset_name = str(record.get("source_dataset", "")).strip() or "unknown"
    profile = resolve_dataset_profile(dataset_name)
    original_question = str(record.get("question", "")).strip()
    original_answer = str(record.get("answer", "")).strip()
    metadata = dict(record.get("metadata") or {})

    question = _format_question(record)
    answer = _normalize_answer(record)
    metadata["mas_dataset_name"] = dataset_name
    metadata["mas_split"] = split
    metadata["mas_task_type"] = profile.task_type
    metadata["mas_answer_format"] = profile.answer_format
    metadata["mas_root_templates"] = list(profile.root_templates)
    metadata["mas_profile_notes"] = profile.notes

    standardized = {
        "id": str(record.get("id", "")).strip(),
        "source_dataset": dataset_name,
        "category": str(record.get("category", "")).strip(),
        "split": split,
        "question": question,
        "answer": answer,
        "original_question": original_question,
        "original_answer": original_answer,
        "task_type": profile.task_type,
        "answer_format": profile.answer_format,
        "metadata": metadata,
    }
    return standardized


def load_jsonl(path: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(record)
            if max_items is not None and len(items) >= max_items:
                break
    return items


def load_processed_split(
    data_root: str,
    dataset_name: str,
    split: str,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    path = os.path.join(data_root, dataset_name, f"{split}.jsonl")
    return load_jsonl(path, max_items=max_items)


def list_processed_datasets(data_root: str) -> List[str]:
    if not os.path.isdir(data_root):
        return []
    names = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    return sorted(name for name in names if not name.startswith("."))


def iter_unified_splits(unified_root: str, splits: Iterable[str]) -> Iterator[tuple[str, Dict[str, Any]]]:
    for split in splits:
        path = os.path.join(unified_root, f"{split}.jsonl")
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield split, json.loads(line)


def build_processed_datasets(
    unified_root: str,
    output_root: str,
    splits: Iterable[str] = ("train", "validation", "test"),
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, record in iter_unified_splits(unified_root, splits):
        dataset_name = str(record.get("source_dataset", "")).strip() or "unknown"
        grouped[dataset_name][split].append(standardize_record(record, split))

    os.makedirs(output_root, exist_ok=True)
    manifest: Dict[str, Any] = {
        "source_root": unified_root,
        "output_root": output_root,
        "datasets": {},
    }
    for dataset_name, split_map in grouped.items():
        dataset_dir = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        manifest["datasets"][dataset_name] = {
            "splits": {},
            "profile": profile_summary([dataset_name])[dataset_name],
        }
        for split in splits:
            items = split_map.get(split, [])
            out_path = os.path.join(dataset_dir, f"{split}.jsonl")
            with open(out_path, "w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest["datasets"][dataset_name]["splits"][split] = len(items)

    manifest_path = os.path.join(output_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
