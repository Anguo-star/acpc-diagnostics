import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from utils import (
    AddGaussianBlur,
    AddNormalizedGaussianNoise,
    AddResize,
    resolve_h5_dataset_path,
)


def infer_model_action_block(model, world) -> int | None:
    action_encoder = getattr(model, "action_encoder", None)
    patch_embed = getattr(action_encoder, "patch_embed", None)
    in_channels = getattr(patch_embed, "in_channels", None)
    action_space = getattr(world.envs, "single_action_space", None)
    if in_channels is None or action_space is None:
        return None
    action_dim = int(np.prod(action_space.shape))
    if action_dim <= 0 or int(in_channels) % action_dim != 0:
        return None
    return int(in_channels) // action_dim


def infer_model_history_size(model, policy: str, cache_dir: str | Path) -> int | None:
    for attr in ("history_size", "history_len"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)

    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if torch.is_tensor(pos_embedding) and pos_embedding.ndim >= 2:
        return int(pos_embedding.shape[1])

    cfg_path = Path(cache_dir, policy).parent / "config.yaml"
    if cfg_path.exists():
        train_cfg = OmegaConf.load(cfg_path)
        value = OmegaConf.select(train_cfg, "wm.history_size")
        if value is not None:
            return int(value)

    return None


def _corruption_magnitude(cfg) -> float:
    """Return a non-negative scalar summarising how much corruption the
    config asks for (zero == no-op), so callers can short-circuit no-op
    cases.

    By type:
        gaussian_noise -> ``std`` (or max(std) for a list)
        gaussian_blur  -> ``kernel_size - 1`` (kernel_size==1 is a no-op)
        resize         -> ``1 - factor`` (factor==1 is a no-op)
    """
    if cfg is None:
        return 0.0
    ctype = cfg.get("type", "gaussian_noise")
    if ctype == "gaussian_noise":
        std = cfg.get("std", 0.0)
        if not isinstance(std, (str, bytes, int, float)) and hasattr(std, "__len__"):
            return max(float(v) for v in std)
        return float(std)
    if ctype == "gaussian_blur":
        ks = float(cfg.get("kernel_size", 0))
        return max(ks - 1.0, 0.0)
    if ctype == "resize":
        return max(1.0 - float(cfg.get("factor", 1.0)), 0.0)
    raise ValueError(f"Unsupported eval corruption type: {ctype}")


def _should_corrupt_target(cfg, target: str):
    corruption = cfg.eval.get("corruption")
    if corruption is None:
        return False
    if _corruption_magnitude(corruption) <= 0:
        return False
    apply_to = corruption.get("apply_to", ["pixels", "goal"])
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    return target in apply_to


def img_transform(cfg, target: str):
    steps = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ]

    if _should_corrupt_target(cfg, target):
        corruption = cfg.eval.corruption
        ctype = corruption.get("type", "gaussian_noise")
        if ctype == "gaussian_noise":
            std = float(corruption.get("std", 0.0))
            steps.append(AddNormalizedGaussianNoise(std, std))
        elif ctype == "gaussian_blur":
            ks = int(round(float(corruption.get("kernel_size", 0))))
            if ks <= 1:
                pass
            else:
                if ks % 2 == 0:
                    ks += 1
                steps.append(AddGaussianBlur(ks, ks))
        elif ctype == "resize":
            factor = float(corruption.get("factor", 1.0))
            steps.append(AddResize(factor, factor))
        else:
            raise ValueError(f"Unsupported eval corruption type: {ctype}")

    return transforms.Compose(steps)


def scan_episode_rows(dataset):
    """Return episode ids, row-to-episode indices, lengths, and step indices.

    ``np.maximum.at`` makes this a single O(N) pass over dataset rows.  The
    previous per-episode boolean scans were O(N * number_of_episodes) and
    became a bottleneck for the 300-episode Paper 1 evaluation grids.
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col_name))
    step_idx = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    episode_ids, row_to_episode = np.unique(episode_idx, return_inverse=True)
    max_step = np.full(len(episode_ids), -1, dtype=np.int64)
    np.maximum.at(max_step, row_to_episode, step_idx)
    return col_name, episode_ids, row_to_episode, max_step + 1, step_idx


def get_dataset(cfg, dataset_name):
    # Resolve the H5 path explicitly so we tolerate both stable-worldmodel layouts
    # (flat 0.0.6-wheel layout and post-PR-#221 `datasets/` subdir).
    dataset_path = Path(cfg.cache_dir) if cfg.cache_dir else None
    h5_path = resolve_h5_dataset_path(dataset_name, cache_dir=dataset_path)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        keys_to_cache=cfg.dataset.keys_to_cache,
    )
    return dataset


def _world_evaluate_compat(world, dataset, start_steps, goal_offset, eval_budget,
                           episodes_idx, callables, video):
    """Bridge across two swm releases:

    * 0.0.6 wheel (and earlier): ``World.evaluate_from_dataset(dataset, ...,
      goal_offset_steps=, video_path=)``.
    * Post-PR-#221 source: ``World.evaluate(dataset=, ..., goal_offset=,
      video=)`` (the old method became the private ``_evaluate_from_dataset``).

    Same args in / same dict out either way, so neither LeWM nor PLDM
    eval paths need to know which swm is installed.
    """
    if hasattr(getattr(world, "policy", None), "reset_history"):
        world.policy.reset_history()

    if hasattr(world, "evaluate_from_dataset"):
        return world.evaluate_from_dataset(
            dataset,
            start_steps=start_steps,
            goal_offset_steps=goal_offset,
            eval_budget=eval_budget,
            episodes_idx=episodes_idx,
            callables=callables,
            video_path=video,
        )
    return world.evaluate(
        dataset=dataset,
        start_steps=start_steps,
        goal_offset=goal_offset,
        eval_budget=eval_budget,
        episodes_idx=episodes_idx,
        callables=callables,
        video=video,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Evaluate a LeWM/PLDM checkpoint or the random policy."""
    num_eval = int(cfg.eval.num_eval)
    requested_batch_size = int(cfg.eval.get("batch_size", num_eval))
    if num_eval < 1:
        raise ValueError("eval.num_eval must be positive")
    if requested_batch_size < 1:
        raise ValueError("eval.batch_size must be positive")
    batch_size = min(requested_batch_size, num_eval)
    with open_dict(cfg):
        cfg.eval.batch_size = batch_size
        cfg.world.num_envs = batch_size

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg, "pixels"),
        "goal": img_transform(cfg, "goal"),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name, _, row_to_episode, episode_len, step_idx = scan_episode_rows(dataset)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        # AutoCostModel's default fallback is get_cache_dir(sub_folder='checkpoints'),
        # which adds a spurious 'checkpoints/' segment that train.py / train_pldm.py
        # do NOT use when saving. Pass STABLEWM_HOME explicitly so the lookup path
        # matches the save path.
        eval_cache_dir = cfg.cache_dir or str(swm.data.utils.get_cache_dir())
        model = swm.policy.AutoCostModel(cfg.policy, cache_dir=eval_cache_dir)

        model_action_block = infer_model_action_block(model, world)
        model_history_len = infer_model_history_size(model, cfg.policy, eval_cache_dir)
        if model_history_len is not None:
            setattr(model, "history_size", model_history_len)
        with open_dict(cfg):
            if model_action_block is not None:
                cfg.plan_config.action_block = model_action_block
            if model_history_len is not None:
                cfg.plan_config.history_len = model_history_len

        assert (
            cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
        ), "Planning horizon must be smaller than or equal to eval_budget"

        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # Sample valid starting rows. ``row_to_episode`` avoids a second scan or
    # per-row dictionary lookup when broadcasting each episode's limit.
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_per_row = max_start_idx[row_to_episode]

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = (max_start_per_row >= 0) & (step_idx <= max_start_per_row)
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.", flush=True)

    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Only {len(valid_indices)} valid starting points are available for "
            f"eval.num_eval={num_eval}."
        )
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices), size=num_eval, replace=False)

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices, flush=True)

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    world.set_policy(policy)

    start_time = time.time()
    save_video = bool(cfg.eval.get("save_video", False))
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    print(
        f"[eval] num_eval={num_eval} batch_size={batch_size} save_video={save_video}",
        flush=True,
    )

    if num_eval > batch_size:
        # Batch evaluation to avoid creating too many parallel envs
        all_successes = []
        all_seeds = []
        for batch_start in range(0, num_eval, batch_size):
            batch_end = min(batch_start + batch_size, num_eval)
            actual_bs = batch_end - batch_start
            batch_episodes = eval_episodes[batch_start:batch_end].tolist()
            batch_start_idx = eval_start_idx[batch_start:batch_end].tolist()

            # Pad to match world.num_envs if last batch is smaller
            if actual_bs < batch_size:
                pad = batch_size - actual_bs
                batch_episodes = batch_episodes + batch_episodes[-1:] * pad
                batch_start_idx = batch_start_idx + batch_start_idx[-1:] * pad

            batch_video_path = None
            if save_video:
                batch_video_path = results_path / f"batch_{batch_start}"
                batch_video_path.mkdir(parents=True, exist_ok=True)

            print(
                f"[eval] batch {batch_start}:{batch_end}/{num_eval} starting",
                flush=True,
            )

            batch_metrics = _world_evaluate_compat(
                world,
                dataset=dataset,
                start_steps=batch_start_idx,
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=batch_episodes,
                callables=callables,
                video=batch_video_path,
            )
            all_successes.extend(batch_metrics["episode_successes"][:actual_bs])
            batch_seeds = batch_metrics.get("seeds")
            if batch_seeds is not None:
                all_seeds.extend(batch_seeds[:actual_bs])
            print(
                f"[eval] batch {batch_start}:{batch_end}/{num_eval} done",
                flush=True,
            )

        metrics = {
            "success_rate": float(np.sum(all_successes)) / num_eval * 100.0,
            "episode_successes": np.array(all_successes),
            "seeds": np.array(all_seeds) if all_seeds else None,
        }
    else:
        print(f"[eval] batch 0:{num_eval}/{num_eval} starting", flush=True)
        metrics = _world_evaluate_compat(
            world,
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            video=results_path if save_video else None,
        )
        print(f"[eval] batch 0:{num_eval}/{num_eval} done", flush=True)
    end_time = time.time()

    print(metrics, flush=True)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
