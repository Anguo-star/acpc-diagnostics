#!/usr/bin/env python3
"""Extract the real PushT pixel examples used by the ACPC overview figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 - registers the compression filters used by the H5.
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT / "assets" / "paper1_figs" / "acpc_overview_inputs"
)

CLEAN_ROW = 1_009_793
DIFFERENT_ROW = 1_009_869
EXPECTED_EPISODE = 8_200
EXPECTED_CLEAN_STEP = 48
EXPECTED_DIFFERENT_STEP = 124
EXPECTED_CLEAN_FRAME_SHA256 = (
    "01a4daaa4aa0a2b1fea3dd373e4bcca8116f62db5f1239bbfe39f36556baa386"
)
EXPECTED_DIFFERENT_FRAME_SHA256 = (
    "ccceef2846ff48a2b957578e98fc431dc38744d8f96829dc8ee6d1cca370e7f1"
)

NOISE_STD = 0.08
NOISE_SEED = 10_110


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _frame_record(
    h5_file: h5py.File,
    row: int,
    *,
    expected_step: int,
    expected_sha256: str,
) -> dict[str, Any]:
    frame = np.asarray(h5_file["pixels"][row], dtype=np.uint8)
    episode = int(h5_file["episode_idx"][row])
    step = int(h5_file["step_idx"][row])
    state = np.asarray(h5_file["state"][row], dtype=np.float32)
    frame_sha256 = _sha256_bytes(frame.tobytes(order="C"))

    _require(frame.shape == (224, 224, 3), f"row {row}: unexpected frame shape")
    _require(episode == EXPECTED_EPISODE, f"row {row}: episode mismatch")
    _require(step == expected_step, f"row {row}: step mismatch")
    _require(
        frame_sha256 == expected_sha256,
        f"row {row}: pixel hash mismatch",
    )
    return {
        "row": row,
        "episode": episode,
        "step": step,
        "frame": frame,
        "frame_sha256": frame_sha256,
        "state": state.tolist(),
        "state_coordinates_2_5": state[2:5].tolist(),
    }


def build(h5_path: Path, out_dir: Path) -> dict[str, Any]:
    _require(h5_path.is_file(), f"missing PushT H5: {h5_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5_file:
        _require(
            tuple(h5_file["pixels"].shape) == (2_336_736, 224, 224, 3),
            "unexpected PushT pixels dataset shape",
        )
        _require(h5_file["pixels"].dtype == np.dtype("uint8"), "pixels must be uint8")
        clean = _frame_record(
            h5_file,
            CLEAN_ROW,
            expected_step=EXPECTED_CLEAN_STEP,
            expected_sha256=EXPECTED_CLEAN_FRAME_SHA256,
        )
        different = _frame_record(
            h5_file,
            DIFFERENT_ROW,
            expected_step=EXPECTED_DIFFERENT_STEP,
            expected_sha256=EXPECTED_DIFFERENT_FRAME_SHA256,
        )

    rng = np.random.default_rng(NOISE_SEED)
    clean_float = clean["frame"].astype(np.float32) / 255.0
    noise = rng.normal(0.0, NOISE_STD, clean_float.shape).astype(np.float32)
    noisy_float = np.clip(clean_float + noise, 0.0, 1.0)
    noisy = np.rint(noisy_float * 255.0).astype(np.uint8)

    outputs = {
        "clean": out_dir / "pusht_ep8200_step48_clean.png",
        "noisy": out_dir / "pusht_ep8200_step48_gaussian_std008.png",
        "different": out_dir / "pusht_ep8200_step124_different_state.png",
    }
    for name, array in (
        ("clean", clean["frame"]),
        ("noisy", noisy),
        ("different", different["frame"]),
    ):
        Image.fromarray(array).save(outputs[name])

    metadata = {
        "schema_version": "paper1-acpc-overview-inputs-1.0",
        "source": {
            "task": "PushT",
            "dataset_name": "pusht_expert_train",
            "h5_filename": h5_path.name,
            "h5_file_size_bytes": h5_path.stat().st_size,
            "pixels_shape": [2_336_736, 224, 224, 3],
            "pixels_dtype": "uint8",
        },
        "clean": {
            key: value
            for key, value in clean.items()
            if key != "frame"
        },
        "different_state": {
            key: value
            for key, value in different.items()
            if key != "frame"
        },
        "same_state_noise": {
            "source_row": CLEAN_ROW,
            "distribution": "independent Gaussian noise in display RGB [0,1]",
            "standard_deviation": NOISE_STD,
            "seed": NOISE_SEED,
            "clipping": "[0,1]",
            "uint8_conversion": "round(255*x)",
            "pixel_sha256": _sha256_bytes(noisy.tobytes(order="C")),
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "figure_role": (
            "Real task input examples. The clean and noisy images depict the "
            "same state; the third image is a different state from the same "
            "recorded episode. Latent diagrams in Figure 1 are schematic."
        ),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {metadata_path}")
    for path in outputs.values():
        print(f"wrote {path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to the PushT pusht_expert_train.h5 dataset.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build(args.h5.expanduser(), args.out_dir.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
