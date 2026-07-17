import argparse
import io
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import hydra
import lance
import numpy as np
import pyarrow as pa
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from sklearn import preprocessing

from eval import (
    _world_evaluate_compat,
    apply_inference_overrides,
    img_transform,
    infer_model_action_block,
)


TASKS = {
    "tworoom": {
        "eval_config": "tworoom",
        "lance_name": "lewm_tworoom.lance",
        "columns": ("pixels", "action", "proprio"),
    },
    "pusht": {
        "eval_config": "pusht",
        "lance_name": "lewm_pusht.lance",
        "columns": ("pixels", "action", "proprio", "state"),
    },
    "reacher": {
        "eval_config": "reacher",
        "lance_name": "lewm_reacher.lance",
        "columns": ("pixels", "action", "observation", "qpos", "qvel"),
    },
}


class BareLanceDataset:
    """Reader adapter for bare ``*.lance`` datasets used by Lance-format LeWM.

    The installed ``stable_worldmodel.data.LanceDataset`` expects a LanceDB
    table layout. The datasets used here are plain Lance datasets, so this
    adapter provides the small interface that ``World.evaluate`` needs.
    """

    def __init__(self, path: str | Path, columns: tuple[str, ...]):
        self.path = str(path)
        self._dataset = lance.dataset(self.path)
        self._keys = list(columns)
        self._source_columns = tuple(
            col for col in columns if col not in ("qpos", "qvel")
        )
        self.episode_col = "episode_idx"
        self.step_col = "step_idx"
        self._cache: dict[str, np.ndarray] = {}

        episode_idx = self.get_col_data(self.episode_col)
        change_positions = np.flatnonzero(np.diff(episode_idx) != 0) + 1
        self.offsets = np.concatenate([[0], change_positions]).astype(np.int64)
        self.lengths = np.diff(
            np.concatenate([self.offsets, [len(episode_idx)]])
        ).astype(np.int64)
        self.episode_ids = episode_idx[self.offsets].astype(np.int64)
        self._offset_by_episode = {
            int(ep_id): int(offset)
            for ep_id, offset in zip(self.episode_ids, self.offsets)
        }

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __len__(self) -> int:
        return int(self.lengths.sum())

    def _table_to_numpy(self, table, col_name: str) -> np.ndarray:
        col = table.column(col_name).combine_chunks()
        ctype = col.type
        if pa.types.is_fixed_size_list(ctype):
            dim = ctype.list_size
            flat = col.flatten()
            return flat.to_numpy(zero_copy_only=False).reshape(len(col), dim)
        if pa.types.is_binary(ctype) or pa.types.is_large_binary(ctype):
            return np.asarray(col.to_pylist(), dtype=object)
        return col.to_numpy(zero_copy_only=False)

    @staticmethod
    def _derive_qpos(observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation)[..., :2].copy()

    @staticmethod
    def _derive_qvel(observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation)[..., -2:].copy()

    def get_col_data(self, col_name: str) -> np.ndarray:
        if col_name in self._cache:
            return self._cache[col_name]

        if col_name in ("qpos", "qvel"):
            observation = self.get_col_data("observation")
            data = (
                self._derive_qpos(observation)
                if col_name == "qpos"
                else self._derive_qvel(observation)
            )
        else:
            table = self._dataset.scanner(columns=[col_name]).to_table()
            data = self._table_to_numpy(table, col_name)

        self._cache[col_name] = data
        return data

    def get_row_data(self, row_idx: int | list[int] | np.ndarray) -> dict:
        if isinstance(row_idx, (list, tuple, np.ndarray)):
            rows = [int(i) for i in row_idx]
        else:
            rows = [int(row_idx)]

        return {
            self.episode_col: self.get_col_data(self.episode_col)[rows],
            self.step_col: self.get_col_data(self.step_col)[rows],
        }

    @staticmethod
    def _decode_image(blob) -> torch.Tensor:
        with Image.open(io.BytesIO(bytes(blob))) as img:
            arr = np.array(img.convert("RGB"))
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _decode_images(self, blobs) -> torch.Tensor:
        return torch.stack([self._decode_image(blob) for blob in blobs])

    def load_chunk(self, episodes_idx, start, end) -> list[dict]:
        chunks = []
        for ep_id, s, e in zip(episodes_idx, start, end):
            offset = self._offset_by_episode[int(ep_id)]
            rows = list(range(offset + int(s), offset + int(e)))
            table = self._dataset.take(rows, columns=self._source_columns)

            steps = {}
            observation = None
            for col in self._source_columns:
                if col == "pixels":
                    steps[col] = self._decode_images(table.column(col).to_pylist())
                    continue

                arr = np.array(self._table_to_numpy(table, col), copy=True)
                if col == "observation":
                    observation = arr
                steps[col] = torch.as_tensor(arr)

            if "qpos" in self._keys:
                if observation is None:
                    raise KeyError("qpos derivation requires an observation column")
                steps["qpos"] = torch.as_tensor(self._derive_qpos(observation))
            if "qvel" in self._keys:
                if observation is None:
                    raise KeyError("qvel derivation requires an observation column")
                steps["qvel"] = torch.as_tensor(self._derive_qvel(observation))
            if "action" in steps:
                steps["action"] = steps["action"].reshape(int(e) - int(s), -1)
            chunks.append(steps)
        return chunks


def compose_eval_cfg(config_name: str, overrides: list[str]):
    config_dir = str(Path.cwd() / "config" / "eval")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides)


def load_model(config_path: Path, weights_path: Path):
    train_cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(train_cfg.model)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    return model, train_cfg


def first_available_cuda() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(args) -> None:
    task_spec = TASKS[args.task]
    overrides = [
        f"eval.num_eval={args.num_eval}",
        f"seed={args.seed}",
        f"policy={args.policy_name}",
        f"eval.corruption.type={args.corruption_type}",
        f"eval.corruption.std={args.corruption_std}",
        f"eval.corruption.kernel_size={args.corruption_kernel_size}",
        f"eval.corruption.factor={args.corruption_factor}",
        f"eval.corruption.apply_to=[{args.corruption_apply_to}]",
    ]
    overrides.extend(args.override)
    cfg = compose_eval_cfg(task_spec["eval_config"], overrides)
    with open_dict(cfg):
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": img_transform(cfg, "pixels"),
        "goal": img_transform(cfg, "goal"),
    }

    lance_path = args.dataset_root / task_spec["lance_name"]
    dataset = BareLanceDataset(lance_path, task_spec["columns"])

    process = {}
    for col in cfg.dataset.keys_to_cache:
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model, train_cfg = load_model(args.config_path, args.weights_path)
    apply_inference_overrides(model, cfg)

    model_action_block = infer_model_action_block(model, world)
    model_history_len = OmegaConf.select(train_cfg, "wm.history_size")
    if model_history_len is None:
        predictor = getattr(model, "predictor", None)
        pos_embedding = getattr(predictor, "pos_embedding", None)
        if torch.is_tensor(pos_embedding) and pos_embedding.ndim >= 2:
            model_history_len = int(pos_embedding.shape[1])

    if model_history_len is not None:
        setattr(model, "history_size", int(model_history_len))
    with open_dict(cfg):
        if model_action_block is not None:
            cfg.plan_config.action_block = model_action_block
        if model_history_len is not None:
            cfg.plan_config.history_len = int(model_history_len)

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    device = first_available_cuda()
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )

    episode_idx = dataset.get_col_data(dataset.episode_col)
    step_idx = dataset.get_col_data(dataset.step_col)
    max_start_idx = dataset.lengths - cfg.eval.goal_offset_steps - 1
    max_start_idx_by_episode = {
        int(ep): int(max_start)
        for ep, max_start in zip(dataset.episode_ids, max_start_idx)
    }
    max_start_per_row = np.array(
        [max_start_idx_by_episode[int(ep)] for ep in episode_idx]
    )
    valid_mask = step_idx <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) <= cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid starts: {len(valid_indices)} "
            f"for num_eval={cfg.eval.num_eval}"
        )

    rng = np.random.default_rng(cfg.seed)
    chosen = rng.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[chosen])
    row_data = dataset.get_row_data(random_episode_indices)
    eval_episodes = row_data[dataset.episode_col]
    eval_start_idx = row_data[dataset.step_col]

    world.set_policy(policy)
    start_time = time.time()
    metrics = _world_evaluate_compat(
        world,
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=args.video_dir,
    )
    elapsed = time.time() - start_time

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metrics = {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in metrics.items()
    }
    result = {
        "task": args.task,
        "metrics": serializable_metrics,
        "evaluation_time": elapsed,
        "lance_path": str(lance_path),
        "weights_path": str(args.weights_path),
    }
    with args.output_path.open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed} seconds\n")
        f.write("==== JSON ====\n")
        f.write(json.dumps(result, sort_keys=True))
        f.write("\n")

    print(metrics)
    print(f"evaluation_time: {elapsed} seconds")
    print(f"wrote: {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--corruption-std", default="0.0")
    parser.add_argument("--corruption-kernel-size", default="1")
    parser.add_argument("--corruption-factor", default="1.0")
    parser.add_argument("--corruption-apply-to", default="pixels")
    parser.add_argument(
        "override",
        nargs="*",
        help="Additional Hydra overrides for the eval config.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
