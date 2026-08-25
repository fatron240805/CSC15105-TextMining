"""Persistent JSONL storage for resumable generation experiments."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_dotenv()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = resolve_repo_path(path)
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value
    config = expand(config)
    if "experiment" not in config or "models" not in config:
        raise ValueError("Config must contain 'experiment' and 'models'.")
    return config, config_path


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append and fsync so a stopped process keeps every completed sample."""
    output = resolve_repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    source = resolve_repo_path(path)
    if not source.exists():
        return []
    records = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_by(records: list[dict], key: str) -> dict[str, dict]:
    out = {}
    for record in records:
        if record.get(key):
            out[record[key]] = record
    return out


def run_dir(config: dict, model_config: dict) -> Path:
    base = resolve_repo_path(config.get("storage", {}).get(
        "runs_dir", "evaluation/generation/runs"))
    experiment_name = slug(config["experiment"]["name"])
    model_name = slug(f"{model_config['provider']}--{model_config['model']}")
    path = base / experiment_name / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def rewrite_cache_key(sample: dict, model_config: dict, rewrite_config: dict,
                      prompt_version: str) -> str:
    return canonical_hash({
        "sample_id": sample["sample_id"],
        "provider": model_config["provider"],
        "model": model_config["model"],
        "prompt_version": prompt_version,
        "params": {
            "temperature": rewrite_config["temperature"],
            "top_p": rewrite_config["top_p"],
            "max_tokens": rewrite_config["max_tokens"],
        },
    })


def judge_cache_key(sample: dict, rewritten: str, judge_config: dict) -> str:
    return canonical_hash({
        "sample_id": sample["sample_id"],
        "rewritten": rewritten,
        "provider": judge_config.get("provider", "gemini"),
        "model": judge_config["model"],
        "prompt_version": judge_config.get("prompt_version", "judge_v1"),
    })


def append_experiment(config: dict, record: dict) -> Path:
    path = resolve_repo_path(config.get("storage", {}).get(
        "experiment_file", "evaluation/generation/experiments.jsonl"))
    append_jsonl(path, record)
    return path
