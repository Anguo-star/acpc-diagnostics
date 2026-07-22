"""Model, dataset, and rollout helpers used by Paper 1 ACPC diagnostics."""

from __future__ import annotations

from typing import Dict, Mapping

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

from utils import get_column_normalizer, get_img_preprocessor, resolve_h5_dataset_path


def load_model(ckpt_path: str, device: str = "cpu"):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.to(device).eval().requires_grad_(False)


def infer_history_size(model) -> int:
    for attr in ("history_size", "history_len"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if not torch.is_tensor(pos_embedding) or pos_embedding.ndim < 2:
        raise ValueError("unable to infer history size from the checkpoint")
    return int(pos_embedding.shape[1])


def resolve_space_name(space: str | None, default: str = "normalized") -> str:
    space = (space or default).lower()
    return "normalized" if space == "sphere" else space


def get_embedding_space(
    outputs: Mapping[str, torch.Tensor], space: str
) -> torch.Tensor:
    space = resolve_space_name(space)
    if space == "raw":
        return outputs["emb_raw"]
    if space == "normalized":
        return outputs["emb"]
    raise ValueError(f"unsupported embedding space: {space}")


def get_model_spaces(model) -> Dict[str, str]:
    analysis = resolve_space_name(
        getattr(model, "analysis_prediction_space", "normalized")
    )
    context = resolve_space_name(
        getattr(model, "training_context_space", analysis)
    )
    rollout = resolve_space_name(
        getattr(model, "inference_rollout_state_space", "normalized")
    )
    cost = resolve_space_name(
        getattr(model, "inference_cost_space", "normalized")
    )
    return {
        "analysis_prediction_space": analysis,
        "training_context_space": context,
        "inference_rollout_state_space": rollout,
        "inference_cost_space": cost,
        "inference_cost_type": getattr(model, "inference_cost_type", "mse").lower(),
    }


def load_dataset_samples(
    *,
    dataset_name: str,
    state_key: str | None,
    n_sequences: int,
    history_size: int,
    future_steps: int,
    frameskip: int,
    img_size: int,
    seed: int,
    device: str,
):
    keys_to_load = ["pixels", "action"]
    if state_key:
        keys_to_load.append(state_key)
    h5_path = resolve_h5_dataset_path(dataset_name)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        num_steps=int(history_size) + int(future_steps),
        frameskip=int(frameskip),
        keys_to_load=keys_to_load,
        transform=None,
    )
    transforms = [
        get_img_preprocessor("pixels", "pixels", int(img_size)),
        get_column_normalizer(dataset, "action", "action"),
    ]
    if state_key:
        transforms.append(get_column_normalizer(dataset, state_key, state_key))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:n_sequences].tolist()
    samples = [dataset[index] for index in indices]
    batch: Dict[str, torch.Tensor] = {
        "pixels": torch.stack([sample["pixels"] for sample in samples]).to(device),
        "action": torch.nan_to_num(
            torch.stack([sample["action"] for sample in samples]), 0.0
        ).to(device),
    }
    if state_key:
        batch["state"] = torch.stack(
            [sample[state_key] for sample in samples]
        ).to(device)
    return batch


@torch.no_grad()
def encode_sequences(
    model, batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    info = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    outputs = {
        "emb": info["emb"],
        "emb_raw": info.get("emb_raw", info["emb"]),
        "act_emb": info["act_emb"],
        "pixels": batch["pixels"],
        "action": batch["action"],
    }
    if "state" in batch:
        outputs["state"] = batch["state"]
    return outputs


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if float(denominator) < 1e-12:
        return 0.0
    return float((x @ y) / denominator)


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    def rankdata(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values, stable=True)
        sorted_values = values[order]
        ranks = torch.empty(order.numel(), dtype=torch.float32, device=values.device)
        start = 0
        while start < order.numel():
            end = start + 1
            while end < order.numel() and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * float(start + end - 1)
            start = end
        return ranks

    return _pearson_corr(rankdata(x.float().reshape(-1)), rankdata(y.float().reshape(-1)))


def sample_random_future_actions(
    future_action: torch.Tensor, *, n_trials: int, seed: int
) -> torch.Tensor:
    batch_size, horizon, action_dim = future_action.shape
    pool = future_action.reshape(-1, action_dim).cpu()
    if pool.size(0) == 0:
        raise ValueError("need at least one future action to sample candidates")
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randint(
        0,
        pool.size(0),
        (batch_size, int(n_trials), horizon),
        generator=generator,
    )
    return pool[indices].to(future_action.device)


def _safe_quantile(value: torch.Tensor, quantile: float) -> float:
    if value.numel() == 0:
        return float("nan")
    return float(torch.quantile(value.float().cpu(), quantile))


def _shift_stats(
    clean: torch.Tensor, perturbed: torch.Tensor
) -> Dict[str, float]:
    clean = clean.reshape(-1, clean.size(-1))
    perturbed = perturbed.reshape(-1, perturbed.size(-1))
    clean_norm = F.normalize(clean, dim=-1, eps=1e-8)
    perturbed_norm = F.normalize(perturbed, dim=-1, eps=1e-8)
    cosine = (clean_norm * perturbed_norm).sum(dim=-1).clamp(-1.0, 1.0)
    cosine_distance = (1.0 - cosine).clamp_min(0.0)
    angle_degrees = torch.rad2deg(torch.acos(cosine))
    l2_distance = torch.linalg.vector_norm(perturbed - clean, dim=-1)
    return {
        "cos_dist_median": _safe_quantile(cosine_distance, 0.5),
        "cos_dist_p90": _safe_quantile(cosine_distance, 0.9),
        "angle_deg_median": _safe_quantile(angle_degrees, 0.5),
        "angle_deg_p90": _safe_quantile(angle_degrees, 0.9),
        "l2_median": _safe_quantile(l2_distance, 0.5),
        "l2_p90": _safe_quantile(l2_distance, 0.9),
    }


def _clean_nn_dist(embedding: torch.Tensor) -> Dict[str, float]:
    embedding = embedding.reshape(-1, embedding.size(-1))
    if embedding.size(0) < 2:
        return {"cos": float("nan"), "l2": float("nan")}
    normalized = F.normalize(embedding, dim=-1, eps=1e-8)
    cosine_distance = 1.0 - normalized @ normalized.T
    l2_distance = torch.cdist(embedding, embedding, p=2)
    diagonal = torch.eye(
        embedding.size(0), dtype=torch.bool, device=embedding.device
    )
    cosine_nn = cosine_distance.masked_fill(diagonal, float("inf")).min(dim=1).values
    l2_nn = l2_distance.masked_fill(diagonal, float("inf")).min(dim=1).values
    return {
        "cos": _safe_quantile(cosine_nn.clamp_min(0.0), 0.5),
        "l2": _safe_quantile(l2_nn, 0.5),
    }


@torch.no_grad()
def _open_loop_target_shift(
    model,
    clean_emb: torch.Tensor,
    perturbed_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
) -> Dict[str, torch.Tensor]:
    _, sequence_length, _ = clean_emb.shape
    if sequence_length <= history_size:
        return {
            "clean_pred": clean_emb[:, :0],
            "noisy_pred": perturbed_emb[:, :0],
        }
    clean_predictions = []
    perturbed_predictions = []
    for start in range(sequence_length - history_size):
        stop = start + history_size
        clean_predictions.append(
            model.predict(clean_emb[:, start:stop], act_emb[:, start:stop])[:, -1]
        )
        perturbed_predictions.append(
            model.predict(
                perturbed_emb[:, start:stop], act_emb[:, start:stop]
            )[:, -1]
        )
    return {
        "clean_pred": torch.stack(clean_predictions, dim=1),
        "noisy_pred": torch.stack(perturbed_predictions, dim=1),
    }


@torch.no_grad()
def _autoregressive_rollout(
    model,
    init_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    n_steps: int,
) -> torch.Tensor:
    chain = init_emb.clone()
    for step in range(int(n_steps)):
        action_window = act_emb[:, step : step + history_size]
        if action_window.size(1) < history_size:
            break
        prediction = model.predict(
            chain[:, -history_size:], action_window
        )[:, -1:]
        chain = torch.cat([chain, prediction], dim=1)
    return chain
