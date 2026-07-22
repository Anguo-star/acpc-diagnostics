"""Portable trajectory-block sampling shared by Paper 1 ACPC runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from utils import get_column_normalizer, get_img_preprocessor, resolve_h5_dataset_path


def choose_unique_episode_indices(
    clip_indices: Sequence[Sequence[int]],
    *,
    n_blocks: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Choose one seeded dataset clip per distinct episode."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    episodes = sorted({int(pair[0]) for pair in clip_indices})
    if len(episodes) < n_blocks:
        raise ValueError(
            f"requested {n_blocks} trajectory blocks but only "
            f"{len(episodes)} episodes are available"
        )

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(episodes), generator=generator)[:n_blocks]
    selected = [episodes[int(index)] for index in permutation]
    selected_set = set(selected)
    candidates: dict[int, list[tuple[int, int]]] = {
        episode: [] for episode in selected
    }
    for dataset_index, pair in enumerate(clip_indices):
        episode, start = int(pair[0]), int(pair[1])
        if episode in selected_set:
            candidates[episode].append((dataset_index, start))

    result: list[tuple[int, int, int]] = []
    for episode in selected:
        choices = candidates[episode]
        if not choices:
            raise RuntimeError(f"selected episode {episode} has no valid clip")
        position = int(
            torch.randint(len(choices), (1,), generator=generator).item()
        )
        dataset_index, start = choices[position]
        result.append((int(dataset_index), int(episode), int(start)))
    return result


@dataclass(frozen=True)
class TrajectoryBlock:
    block_index: int
    dataset_index: int
    episode_id: int
    start_step: int

    @property
    def block_id(self) -> str:
        return f"episode-{self.episode_id}"


def load_trajectory_blocks(
    *,
    dataset_name: str,
    n_blocks: int,
    history_size: int,
    future_steps: int,
    frameskip: int,
    img_size: int,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[TrajectoryBlock]]:
    """Load one transformed clip from each independently sampled episode."""
    num_steps = int(history_size) + int(future_steps)
    h5_path = resolve_h5_dataset_path(dataset_name)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        num_steps=num_steps,
        frameskip=int(frameskip),
        keys_to_load=["pixels", "action"],
        transform=None,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", int(img_size)),
        get_column_normalizer(dataset, "action", "action"),
    )
    chosen = choose_unique_episode_indices(
        dataset.clip_indices,
        n_blocks=int(n_blocks),
        seed=int(seed),
    )
    samples = [dataset[index] for index, _episode, _start in chosen]
    batch = {
        "pixels": torch.stack([sample["pixels"] for sample in samples]).to(device),
        "action": torch.nan_to_num(
            torch.stack([sample["action"] for sample in samples]), 0.0
        ).to(device),
    }
    blocks = [
        TrajectoryBlock(
            block_index=index,
            dataset_index=dataset_index,
            episode_id=episode,
            start_step=start,
        )
        for index, (dataset_index, episode, start) in enumerate(chosen)
    ]
    if len({block.episode_id for block in blocks}) != len(blocks):
        raise RuntimeError("trajectory block sampler returned duplicate episodes")
    return batch, blocks
