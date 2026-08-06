#!/usr/bin/env python3
"""Replay one structured-memory reconciliation point against a fixed activity log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tty_agent.memory_replay import (
    load_activity_memory_events,
    load_structured_journal_batches,
    select_batch_events,
    write_journal_prefix,
)
from tty_agent.models import OpenAICompatibleAdapter
from tty_agent.structured_memory import StructuredMemoryConfig, StructuredMemorySubsystem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activity_log", type=Path)
    parser.add_argument("source_context", type=Path, help="Directory containing the original ops.jsonl and manifest")
    point = parser.add_mutually_exclusive_group(required=True)
    point.add_argument("--batch-index", type=int, help="Replay this original journal batch")
    point.add_argument(
        "--audit-after-batch",
        type=int,
        metavar="COUNT",
        help="Copy COUNT original batches, then run an eventless final audit",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent-id", default="memory-point-replay")
    parser.add_argument("--context-id", default="zork")
    parser.add_argument("--model", default="accounts/fireworks/models/qwen3p7-plus")
    parser.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    parser.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--audit-temperature",
        type=float,
        help="final cleanup-audit temperature; omitted inherits --temperature",
    )
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--prompt-append-file", type=Path)
    parser.add_argument(
        "--audit-prompt-append-file",
        "--commit-prompt-append-file",
        dest="audit_prompt_append_file",
        type=Path,
        help="append extra policy to the cleanup-only audit prompt",
    )
    parser.add_argument("--compaction-max-tokens", type=int, default=32_768)
    parser.add_argument("--memory-max-tokens", type=int, default=32_768)
    parser.add_argument("--utility-token-ceiling", type=int, default=262_144)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--label", default="")
    return parser


def _read_appendix(path: Path | None) -> str:
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt appendix is empty: {path}")
    return text


def _source_config(context: Path) -> StructuredMemoryConfig:
    manifest_path = context / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"could not load structured-memory config from {manifest_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"invalid structured-memory config in {manifest_path}")
    return StructuredMemoryConfig(**config)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extra_body(args: argparse.Namespace) -> dict[str, object]:
    body: dict[str, object] = {}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if args.top_p is not None:
        body["top_p"] = args.top_p
    return body


def _new_records(path: Path, prefix_count: int) -> list[dict[str, Any]]:
    batches = load_structured_journal_batches(path)
    return [record for batch in batches[prefix_count:] for record in batch.records]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"environment variable {args.api_key_env} is not set")

    source_journal = args.source_context / "ops.jsonl"
    batches = load_structured_journal_batches(source_journal)
    config = _source_config(args.source_context)
    events = load_activity_memory_events(args.activity_log)
    target = None
    if args.batch_index is not None:
        if args.batch_index < 0 or args.batch_index >= len(batches):
            raise ValueError(f"--batch-index must be between 0 and {len(batches) - 1}")
        prefix_count = args.batch_index
        target = batches[args.batch_index]
        selected, overlap = select_batch_events(events, batches, args.batch_index, config.overlap_events)
    else:
        assert args.audit_after_batch is not None
        prefix_count = args.audit_after_batch
        if prefix_count < 0 or prefix_count > len(batches):
            raise ValueError(f"--audit-after-batch must be between 0 and {len(batches)}")
        selected, overlap = [], []
    prompt_appendix = _read_appendix(args.prompt_append_file)
    audit_prompt_appendix = _read_appendix(args.audit_prompt_append_file)

    context_directory = args.output_root / args.agent_id / args.context_id
    if context_directory.exists() and any(context_directory.iterdir()):
        raise FileExistsError(f"output context is not empty: {context_directory}")
    output_journal = context_directory / "ops.jsonl"
    write_journal_prefix(source_journal, output_journal, prefix_count)

    subsystem = StructuredMemorySubsystem(
        args.output_root,
        config,
        reconcile_prompt_appendix=prompt_appendix,
        audit_prompt_appendix=audit_prompt_appendix,
    )
    model = OpenAICompatibleAdapter(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        audit_temperature=args.audit_temperature,
        max_tokens=args.compaction_max_tokens,
        extra_body=_extra_body(args),
        compaction_max_tokens=args.compaction_max_tokens,
        memory_max_tokens=args.memory_max_tokens,
        compaction_max_tokens_retry_ceiling=args.utility_token_ceiling,
        memory_max_tokens_retry_ceiling=args.utility_token_ceiling,
    )

    handle = subsystem.open_context(args.agent_id, args.context_id)
    started = time.monotonic()
    try:
        before = handle.memory.to_dict()
        # This is deliberately point-replay-only state injection. It restores
        # the exact context-only tail that the original live handle retained.
        handle._covered_tail = list(overlap)
        for event in selected:
            handle.observe(event)
        if target is None:
            outcome = handle.commit(model)
        elif target.batch == "commit":
            outcome = handle.commit(model)
        elif target.batch == "reconcile":
            outcome = handle.maybe_reconcile(model, force=True)
        else:
            raise ValueError(f"unsupported source batch type {target.batch!r}")
        if outcome is None:
            raise RuntimeError("point replay unexpectedly produced no reconciliation outcome")
        after = handle.memory.to_dict()
    finally:
        handle.close()

    result = {
        "schema_version": 1,
        "label": args.label,
        "activity_log": str(args.activity_log),
        "activity_log_sha256": _sha256(args.activity_log),
        "source_context": str(args.source_context),
        "source_journal_sha256": _sha256(source_journal),
        "source_prefix_batches": prefix_count,
        "source_batch": (
            {
                "index": target.index,
                "batch": target.batch,
                "source_steps": list(target.source_steps),
                "records": list(target.records),
            }
            if target is not None
            else None
        ),
        "sampling": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "audit_temperature": args.temperature if args.audit_temperature is None else args.audit_temperature,
            "top_p": args.top_p,
            "reasoning_effort": args.reasoning_effort,
            "compaction_max_tokens": args.compaction_max_tokens,
            "memory_max_tokens": args.memory_max_tokens,
            "utility_token_ceiling": args.utility_token_ceiling,
        },
        "prompt": {
            "append_file": str(args.prompt_append_file) if args.prompt_append_file else None,
            "appendix": prompt_appendix,
            "audit_append_file": str(args.audit_prompt_append_file) if args.audit_prompt_append_file else None,
            "audit_appendix": audit_prompt_appendix,
            "fingerprint": subsystem.fingerprints()["prompt"],
        },
        "overlap_steps": [event.step for event in overlap],
        "memory_before": before,
        "memory_after": after,
        "replay_records": _new_records(output_journal, prefix_count),
        "outcome": outcome.to_dict(),
        "model_response": model.last_parsed_response,
        "model_reasoning": model.last_reasoning,
        "request_metadata": model.last_request_metadata,
        "wall_seconds": time.monotonic() - started,
    }
    result_path = context_directory / "point-replay.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "outcome": outcome.to_dict()}, sort_keys=True))
    return 0 if outcome.status != "failed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
