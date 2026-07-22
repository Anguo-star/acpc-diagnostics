#!/usr/bin/env bash

# Paper 1 train/eval entry point for the two released model families.
#
# Required environment variables:
#   dataset_name       pusht | tworoom | reacher | cube
#   trainer_file       train.py | train_pldm.py
#   config             lewm | pldm (a .yaml suffix is accepted)
#   output_model_name  suffix; the task name is prepended automatically
#   num_eval           total episodes across all evaluation seeds
#   STABLEWM_HOME      data/checkpoint root (or the task-specific lewm-* root)
#
# Training controls:
#   skip_train=1
#   seed=<int>
#   image_noise_std_min=<float>
#   image_noise_std_max=<float>
#   image_noise_noise_prob=<float>
#   image_noise_apply_to_val=True|False
#   wandb_enabled=True|False
#
# Evaluation controls:
#   post_train_eval_mode=full|origin|none
#   eval_corruption_type=gaussian_noise|gaussian_blur|resize
#   eval_corruption_stds="0.0 0.03 0.05 0.08"
#   eval_blur_kernel_sizes="1 3 7 11 15"
#   eval_resize_factors="1.0 0.75 0.5 0.25"
#   eval_corruption_apply_to=1|2|3|4|5 (or pixels/goal/pixels+goal)
#   eval_seeds=3, eval_base_seed=42, eval_epoch=<checkpoint epoch>
#   eval_gpus="0 1", eval_batch_size=100, eval_max_concurrency=<positive int>
#   eval_resume=1, eval_save_video=0, eval_timeout_seconds=7200
#   skip_eval_sweep=1, ckpt_override=/absolute/path/to/*_object.ckpt

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for required_name in dataset_name trainer_file config output_model_name num_eval STABLEWM_HOME; do
    if [ -z "${!required_name:-}" ]; then
        echo "[error] required environment variable is empty: ${required_name}" >&2
        exit 2
    fi
done

if ! [[ "${num_eval}" =~ ^[0-9]+$ ]] || [ "${num_eval}" -lt 1 ]; then
    echo "[error] num_eval must be a positive integer; got '${num_eval}'" >&2
    exit 2
fi

config_name="${config##*/}"
config_name="${config_name%.yaml}"
config_name="${config_name%.yml}"
trainer_basename="$(basename "${trainer_file}")"
case "${config_name}:${trainer_basename}" in
    lewm:train.py|pldm:train_pldm.py) ;;
    *)
        echo "[error] supported trainer/config pairs are train.py+lewm and train_pldm.py+pldm; got ${trainer_file}+${config}" >&2
        exit 2
        ;;
esac

case "${dataset_name}" in
    tworoom)
        data="tworoom"
        dataset_dirname="tworooms"
        ;;
    pusht)
        data="pusht"
        dataset_dirname="pusht"
        ;;
    cube)
        data="ogb"
        dataset_dirname="cube"
        ;;
    reacher)
        data="dmc"
        dataset_dirname="reacher"
        ;;
    *)
        echo "[error] unknown dataset_name '${dataset_name}'" >&2
        exit 2
        ;;
esac

input_root="${STABLEWM_HOME%/}"
if [[ "$(basename "${input_root}")" == lewm-* ]]; then
    task_root="$(dirname "${input_root}")/lewm-${dataset_dirname}"
else
    task_root="${input_root}/lewm-${dataset_dirname}"
fi
export STABLEWM_HOME="${task_root}"

dataset_cfg="${SCRIPT_DIR}/config/train/data/${data}.yaml"
if [ ! -f "${dataset_cfg}" ]; then
    echo "[error] training data config not found: ${dataset_cfg}" >&2
    exit 2
fi
default_frameskip="$(awk '/^[[:space:]]*frameskip:/ {sub(/#.*/, ""); sub(/.*:[[:space:]]*/, ""); print; exit}' "${dataset_cfg}")"
frameskip="${frameskip:-${default_frameskip:-5}}"

train_cfg="${SCRIPT_DIR}/config/train/${config_name}.yaml"
if [ ! -f "${train_cfg}" ]; then
    echo "[error] training config not found: ${train_cfg}" >&2
    exit 2
fi
config_max_epochs="$(awk '
    /^[^[:space:]]/ { in_trainer=($1=="trainer:") }
    in_trainer && /^[[:space:]]*max_epochs:/ {
        sub(/#.*/, ""); sub(/.*:[[:space:]]*/, ""); print; exit
    }
' "${train_cfg}")"
if [ -z "${config_max_epochs}" ]; then
    echo "[error] trainer.max_epochs not found in ${train_cfg}" >&2
    exit 2
fi
eval_epoch="${eval_epoch:-${config_max_epochs}}"

final_model_name="${dataset_name}_${output_model_name}"

CMD_ARGS=()
add_override() {
    local key="$1"
    local value="${2:-}"
    if [ -n "${value}" ]; then
        CMD_ARGS+=("${key}=${value}")
    fi
}

add_override "data" "${data}"
add_override "data.dataset.frameskip" "${frameskip}"
add_override "seed" "${seed:-}"
add_override "output_model_name" "${final_model_name}"
add_override "subdir" "ckpt/${final_model_name}"
add_override "image_noise.std_min" "${image_noise_std_min:-}"
add_override "image_noise.std_max" "${image_noise_std_max:-}"
add_override "image_noise.noise_prob" "${image_noise_noise_prob:-}"
add_override "image_noise.apply_to_val" "${image_noise_apply_to_val:-}"
add_override "wandb.enabled" "${wandb_enabled:-}"

if [ "${skip_train:-0}" = "1" ]; then
    echo "[train] skipped (skip_train=1)"
else
    echo "[train] ${trainer_basename} config=${config_name} task=${dataset_name} model=${final_model_name}"
    python "${trainer_file}" --config-name="${config_name}" "${CMD_ARGS[@]}"
    train_status=$?
    if [ "${train_status}" -ne 0 ]; then
        echo "[train] failed with status ${train_status}" >&2
        exit "${train_status}"
    fi
fi

post_train_eval_mode="${post_train_eval_mode:-full}"
case "${post_train_eval_mode}" in
    full|origin|none) ;;
    *)
        echo "[error] post_train_eval_mode must be full, origin, or none; got '${post_train_eval_mode}'" >&2
        exit 2
        ;;
esac

if [ "${post_train_eval_mode}" = "none" ]; then
    echo "[done] evaluation skipped (post_train_eval_mode=none)"
    exit 0
fi
if [ "${skip_eval_sweep:-0}" = "1" ]; then
    echo "[done] evaluation skipped (skip_eval_sweep=1)"
    exit 0
fi
if [ "${skip_diagnostics:-1}" != "1" ]; then
    echo "[note] run_trainer.sh performs behavioral evaluation only; run ACPC/IR/SR diagnostics with the paper1 scripts."
fi

if [ -n "${ckpt_override:-}" ]; then
    ckpt_abs="$(realpath -m -- "${ckpt_override}")"
    case "${ckpt_abs}" in
        *_object.ckpt) ckpt_policy="${ckpt_abs%_object.ckpt}" ;;
        *)
            echo "[error] ckpt_override must end in _object.ckpt: ${ckpt_abs}" >&2
            exit 2
            ;;
    esac
else
    ckpt_policy="ckpt/${final_model_name}/${final_model_name}_epoch_${eval_epoch}"
    ckpt_abs="${STABLEWM_HOME}/${ckpt_policy}_object.ckpt"
fi
if [ ! -f "${ckpt_abs}" ]; then
    echo "[error] checkpoint not found: ${ckpt_abs}" >&2
    exit 2
fi

results_dir="${STABLEWM_HOME}/ckpt/${final_model_name}/eval_results"
mkdir -p "${results_dir}"

eval_seeds="${eval_seeds:-3}"
eval_resume="${eval_resume:-0}"
eval_save_video="${eval_save_video:-0}"
eval_timeout_seconds="${eval_timeout_seconds:-0}"
eval_batch_size="${eval_batch_size:-100}"

for positive_name in eval_seeds eval_batch_size; do
    positive_value="${!positive_name}"
    if ! [[ "${positive_value}" =~ ^[0-9]+$ ]] || [ "${positive_value}" -lt 1 ]; then
        echo "[error] ${positive_name} must be a positive integer; got '${positive_value}'" >&2
        exit 2
    fi
done
if ! [[ "${eval_timeout_seconds}" =~ ^[0-9]+$ ]]; then
    echo "[error] eval_timeout_seconds must be a non-negative integer; got '${eval_timeout_seconds}'" >&2
    exit 2
fi
for binary_name in eval_resume eval_save_video; do
    binary_value="${!binary_name}"
    if [ "${binary_value}" != "0" ] && [ "${binary_value}" != "1" ]; then
        echo "[error] ${binary_name} must be 0 or 1; got '${binary_value}'" >&2
        exit 2
    fi
done
if [ "${eval_save_video}" = "1" ]; then
    eval_save_video_bool=true
else
    eval_save_video_bool=false
fi

if [ -z "${eval_base_seed:-}" ]; then
    eval_cfg="${SCRIPT_DIR}/config/eval/${dataset_name}.yaml"
    eval_base_seed="$(awk '/^seed:/ {sub(/#.*/, ""); sub(/.*:[[:space:]]*/, ""); print; exit}' "${eval_cfg}")"
fi
if ! [[ "${eval_base_seed}" =~ ^[0-9]+$ ]]; then
    echo "[error] eval_base_seed must be a non-negative integer; got '${eval_base_seed}'" >&2
    exit 2
fi

per_seed_num_eval=$((num_eval / eval_seeds))
if [ "${per_seed_num_eval}" -lt 1 ]; then
    echo "[error] num_eval=${num_eval} is smaller than eval_seeds=${eval_seeds}" >&2
    exit 2
fi
if [ $((per_seed_num_eval * eval_seeds)) -ne "${num_eval}" ]; then
    echo "[eval][warn] num_eval=${num_eval} is not divisible by eval_seeds=${eval_seeds}; using ${per_seed_num_eval} episodes per seed"
fi

if [ -z "${eval_gpus:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        eval_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' ')"
    else
        eval_gpus="0"
    fi
fi
read -ra gpu_array <<< "${eval_gpus}"
n_gpus=${#gpu_array[@]}
if [ "${n_gpus}" -lt 1 ]; then
    echo "[error] eval_gpus did not contain a GPU id" >&2
    exit 2
fi
eval_max_concurrency="${eval_max_concurrency:-1}"
if ! [[ "${eval_max_concurrency}" =~ ^[0-9]+$ ]] || [ "${eval_max_concurrency}" -lt 1 ]; then
    echo "[error] eval_max_concurrency must be a positive integer; got '${eval_max_concurrency}'" >&2
    exit 2
fi
if [ "${eval_max_concurrency}" -gt "${n_gpus}" ]; then
    eval_max_concurrency="${n_gpus}"
fi

normalize_eval_corruption_apply_to() {
    local raw="${1:-1}"
    local compact="${raw//[[:space:]]/}"
    case "${compact}" in
        ""|1|pixel|pixels|obs|observation) echo "pixels" ;;
        2|goal) echo "goal" ;;
        3|both|pixels+goal|pixels_goal|pixels-goal|pixelsgoal) echo "pixels+goal" ;;
        4|pixels,pixels+goal|pixels,pixels_goal) echo "pixels,pixels+goal" ;;
        5|all) echo "pixels,goal,pixels+goal" ;;
        *)
            local token
            local normalized
            local values=()
            IFS=',' read -ra tokens <<< "${compact}"
            for token in "${tokens[@]}"; do
                case "${token}" in
                    pixel|pixels|obs|observation) normalized="pixels" ;;
                    goal) normalized="goal" ;;
                    both|pixels+goal|pixels_goal|pixels-goal|pixelsgoal) normalized="pixels+goal" ;;
                    *)
                        echo "[error] invalid eval_corruption_apply_to token '${token}'" >&2
                        return 1
                        ;;
                esac
                values+=("${normalized}")
            done
            local IFS=','
            echo "${values[*]}"
            ;;
    esac
}

eval_corruption_type="${eval_corruption_type:-gaussian_noise}"
eval_corruption_apply_to_raw="${eval_corruption_apply_to:-1}"
if ! eval_corruption_apply_to="$(normalize_eval_corruption_apply_to "${eval_corruption_apply_to_raw}")"; then
    exit 2
fi
eval_corruption_stds="${eval_corruption_stds-0.0 0.03 0.05 0.08}"
eval_blur_kernel_sizes="${eval_blur_kernel_sizes-1 3 7 11 15}"
eval_resize_factors="${eval_resize_factors-1.0 0.75 0.5 0.25}"

case "${eval_corruption_type}" in
    gaussian_noise)
        sweep_magnitudes="${eval_corruption_stds}"
        tag_prefix="std"
        origin_magnitude="0.0"
        ;;
    gaussian_blur)
        sweep_magnitudes="${eval_blur_kernel_sizes}"
        tag_prefix="blur_ks"
        origin_magnitude="1"
        ;;
    resize)
        sweep_magnitudes="${eval_resize_factors}"
        tag_prefix="rs_factor"
        origin_magnitude="1.0"
        ;;
    *)
        echo "[error] unknown eval_corruption_type '${eval_corruption_type}'" >&2
        exit 2
        ;;
esac
if [ "${post_train_eval_mode}" = "origin" ]; then
    sweep_magnitudes="${origin_magnitude}"
fi

seed_suffix() {
    if [ "${eval_seeds}" -gt 1 ]; then
        echo "_seed$1"
    fi
}

jobs=()
for magnitude in ${sweep_magnitudes}; do
    is_origin="$(awk -v magnitude="${magnitude}" -v origin="${origin_magnitude}" 'BEGIN {print (magnitude+0 == origin+0) ? 1 : 0}')"
    for ((seed_offset=0; seed_offset<eval_seeds; seed_offset++)); do
        current_seed=$((eval_base_seed + seed_offset))
        suffix="$(seed_suffix "${current_seed}")"
        if [ "${is_origin}" = "1" ]; then
            jobs+=("origin${suffix}|${origin_magnitude}|none|${current_seed}|${eval_corruption_type}")
        else
            IFS=',' read -ra apply_modes <<< "${eval_corruption_apply_to}"
            for apply_mode in "${apply_modes[@]}"; do
                label="${apply_mode//+/_}_${tag_prefix}${magnitude}${suffix}"
                jobs+=("${label}|${magnitude}|${apply_mode}|${current_seed}|${eval_corruption_type}")
            done
        fi
    done
done

eval_artifact_complete() {
    local label="$1"
    local metrics_path="${results_dir}/${label}_metrics.txt"
    local log_path="${results_dir}/${label}.log"
    [ -s "${metrics_path}" ] &&
        [ -s "${log_path}" ] &&
        grep -qF "==== RESULTS ====" "${metrics_path}" &&
        grep -qF "evaluation_time:" "${metrics_path}" &&
        grep -qF "'success_rate':" "${log_path}"
}

run_one_eval() {
    local job="$1"
    local gpu="$2"
    local parts
    IFS='|' read -ra parts <<< "${job}"
    local label="${parts[0]}"
    local magnitude="${parts[1]}"
    local apply_mode="${parts[2]}"
    local eval_seed="${parts[3]}"
    local corruption_type="${parts[4]}"
    local metrics_path="${results_dir}/${label}_metrics.txt"
    local log_path="${results_dir}/${label}.log"

    if [ "${eval_resume}" = "1" ] && eval_artifact_complete "${label}"; then
        echo "[eval] resume skip label=${label}"
        return 0
    fi

    if [ "${eval_resume}" = "1" ] && { [ -e "${metrics_path}" ] || [ -e "${log_path}" ]; }; then
        local interrupted_at
        interrupted_at="$(date -u +%Y%m%dT%H%M%SZ)"
        if [ -e "${metrics_path}" ]; then
            mv -- "${metrics_path}" "${metrics_path}.interrupted_${interrupted_at}"
        fi
        if [ -e "${log_path}" ]; then
            mv -- "${log_path}" "${log_path}.interrupted_${interrupted_at}"
        fi
        echo "[eval] archived incomplete artifact(s) for label=${label}"
    fi

    local args=(
        "--config-name=${dataset_name}.yaml"
        "policy=${ckpt_policy}"
        "seed=${eval_seed}"
        "eval.num_eval=${per_seed_num_eval}"
        "eval.batch_size=${eval_batch_size}"
        "eval.save_video=${eval_save_video_bool}"
        "output.filename=${metrics_path}"
    )
    if [ "${apply_mode}" != "none" ]; then
        args+=("eval.corruption.type=${corruption_type}")
        case "${corruption_type}" in
            gaussian_noise) args+=("eval.corruption.std=${magnitude}") ;;
            gaussian_blur) args+=("eval.corruption.kernel_size=${magnitude}") ;;
            resize) args+=("eval.corruption.factor=${magnitude}") ;;
        esac
        args+=("eval.corruption.apply_to=[${apply_mode//+/,}]")
    fi

    local eval_command=(python -u eval.py "${args[@]}")
    if [ "${eval_timeout_seconds}" -gt 0 ]; then
        eval_command=(timeout --signal=TERM --kill-after=60s "${eval_timeout_seconds}s" "${eval_command[@]}")
    fi

    echo "[eval] start gpu=${gpu} label=${label} type=${corruption_type} magnitude=${magnitude} seed=${eval_seed}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${eval_command[@]}" >"${log_path}" 2>&1
    local status=$?
    if [ "${status}" -eq 0 ] && eval_artifact_complete "${label}"; then
        echo "[eval] done  gpu=${gpu} label=${label}"
        return 0
    fi
    if [ "${status}" -eq 0 ]; then
        status=65
    fi
    echo "[eval] FAIL  gpu=${gpu} label=${label} status=${status}; see ${log_path}" >&2
    return "${status}"
}

echo "[eval] checkpoint=${ckpt_abs} jobs=${#jobs[@]} seeds=${eval_seeds} episodes_per_seed=${per_seed_num_eval} batch_size=${eval_batch_size} concurrency=${eval_max_concurrency}"
failed_jobs=0
job_index=0
while [ "${job_index}" -lt "${#jobs[@]}" ]; do
    pids=()
    while [ "${#pids[@]}" -lt "${eval_max_concurrency}" ] && [ "${job_index}" -lt "${#jobs[@]}" ]; do
        gpu="${gpu_array[$((job_index % n_gpus))]}"
        run_one_eval "${jobs[$job_index]}" "${gpu}" &
        pids+=("$!")
        job_index=$((job_index + 1))
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed_jobs=$((failed_jobs + 1))
        fi
    done
done

summary_file="${results_dir}/summary.txt"
python3 - "${results_dir}" "${summary_file}" "${final_model_name}" "${ckpt_abs}" "${dataset_name}" <<'PYEOF'
import ast
import csv
import math
import re
import statistics
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
model_name, checkpoint, dataset_name = sys.argv[3:6]
groups = {}
array_pattern = re.compile(r"\b(?:np\.)?array\((?:[^()]|\([^()]*\))*\)", re.DOTALL)

for metrics_path in sorted(results_dir.glob("*_metrics.txt")):
    label = metrics_path.name[:-len("_metrics.txt")]
    match = re.match(r"^(.*?)(?:_seed(\d+))?$", label)
    group, seed = match.group(1), match.group(2)
    text = metrics_path.read_text(encoding="utf-8", errors="replace")
    candidates = re.findall(r"^metrics:\s*(\{.*?\})(?=\nevaluation_time:|\Z)", text, re.MULTILINE | re.DOTALL)
    if not candidates:
        continue
    try:
        metrics = ast.literal_eval(array_pattern.sub("None", candidates[-1]))
    except (SyntaxError, ValueError):
        continue
    if isinstance(metrics, dict):
        groups.setdefault(group, []).append((seed, metrics))

csv_rows = [("group", "n_seeds", "seeds", "metric", "mean", "std", "sem", "values")]
lines = [
    f"===== {model_name} eval summary =====",
    f"ckpt: {checkpoint}",
    f"dataset: {dataset_name}",
    "",
]
for group in sorted(groups):
    runs = groups[group]
    seeds = ",".join(seed or "-" for seed, _ in runs)
    lines.append(f"== {group} (n_seeds={len(runs)}, seeds={seeds}) ==")
    keys = sorted({key for _, metrics in runs for key in metrics})
    for key in keys:
        values = [
            float(metrics[key])
            for _, metrics in runs
            if isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
        ]
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        lines.append(f"  {key}: mean={mean:.4f} std={std:.4f} sem={sem:.4f} raw={values}")
        csv_rows.append((
            group,
            len(runs),
            seeds,
            key,
            f"{mean:.6f}",
            f"{std:.6f}",
            f"{sem:.6f}",
            ";".join(f"{value:.6f}" for value in values),
        ))
    lines.append("")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
with (results_dir / "eval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerows(csv_rows)
PYEOF

echo "[done] results=${results_dir} summary=${summary_file}"
if [ "${failed_jobs}" -gt 0 ]; then
    echo "[error] ${failed_jobs} evaluation job(s) failed" >&2
    exit 1
fi
