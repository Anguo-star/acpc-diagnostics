"""Hash-bound protocol/addendum validation for Paper 1 operational runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


class ProtocolBindingError(RuntimeError):
    """Raised before any result generation when a frozen binding mismatches."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        return float(value)
    return value


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise ProtocolBindingError(f"path escapes repository root: {path}") from exc


def validate_frozen_execution(
    *,
    protocol_path: Path,
    addendum_path: Path,
    runner_path: Path,
    shard_id: str,
    checkpoint_path: Path,
    output_path: Path,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a complete frozen shard contract before model/data loading."""

    protocol_path = protocol_path.resolve()
    addendum_path = addendum_path.resolve()
    runner_path = runner_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_pre_execution":
        raise ProtocolBindingError("protocol status is not frozen_pre_execution")
    if protocol.get("immutable") is not True:
        raise ProtocolBindingError("protocol is not marked immutable")
    if protocol.get("execution_authorized") is not True:
        raise ProtocolBindingError("protocol does not authorize execution")
    if addendum.get("status") != "frozen_pre_execution":
        raise ProtocolBindingError("addendum status is not frozen_pre_execution")
    parent = addendum.get("parent_protocol", {})
    if parent.get("path") != _relative_to_root(protocol_path):
        raise ProtocolBindingError("addendum parent protocol path mismatch")
    if parent.get("sha256") != sha256(protocol_path):
        raise ProtocolBindingError("addendum parent protocol hash mismatch")

    runner_relative = _relative_to_root(runner_path)
    runner_hash = addendum.get("source_hashes", {}).get(runner_relative)
    if runner_hash != sha256(runner_path):
        raise ProtocolBindingError(f"runner hash mismatch: {runner_relative}")
    protocol_runner_hash = protocol.get("source_hashes", {}).get(runner_relative)
    if protocol_runner_hash != sha256(runner_path):
        raise ProtocolBindingError(f"protocol runner hash mismatch: {runner_relative}")

    shards = addendum.get("authorized_shards", [])
    matches = [shard for shard in shards if shard.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ProtocolBindingError(
            f"expected exactly one authorized shard {shard_id!r}; found {len(matches)}"
        )
    shard = matches[0]
    if shard.get("runner") != runner_relative:
        raise ProtocolBindingError("authorized shard runner mismatch")
    if shard.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ProtocolBindingError("authorized checkpoint hash mismatch")
    if shard.get("output_path") != _relative_to_root(output_path):
        raise ProtocolBindingError("authorized output path mismatch")
    expected_arguments = shard.get("arguments", {})
    actual_arguments = {key: _normalize(arguments.get(key)) for key in expected_arguments}
    normalized_expected = {
        key: _normalize(value) for key, value in expected_arguments.items()
    }
    if actual_arguments != normalized_expected:
        differences = {
            key: {
                "expected": normalized_expected.get(key),
                "actual": actual_arguments.get(key),
            }
            for key in normalized_expected
            if normalized_expected.get(key) != actual_arguments.get(key)
        }
        raise ProtocolBindingError(f"authorized shard argument mismatch: {differences}")

    result_root = ROOT / str(addendum.get("result_root", ""))
    try:
        output_path.relative_to(result_root.resolve())
    except ValueError as exc:
        raise ProtocolBindingError("output path escapes frozen result root") from exc
    return protocol, addendum, shard


def namespace_arguments(namespace: Any, names: Sequence[str]) -> dict[str, Any]:
    return {name: getattr(namespace, name) for name in names}
