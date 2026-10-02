#!/usr/bin/env bash
# Re-evaluate the selected V1 auxiliary checkpoints on their complete validation
# split. This performs no optimisation and never opens the held-out test loader.
#
# Interactive use:
#   bash jobs/ClusterA/revalidate_auxiliary_full_validation.sh chimericity
#   bash jobs/ClusterA/revalidate_auxiliary_full_validation.sh oxidized_met
#   bash jobs/ClusterA/revalidate_auxiliary_full_validation.sh retention_time

set -eo pipefail

TASK="${1:-}"
MODE="${2:-all}"
if [[ ! "${TASK}" =~ ^(chimericity|oxidized_met|retention_time)$ ]]; then
    echo "Usage: $0 {chimericity|oxidized_met|retention_time}" >&2
    exit 2
fi
if [[ "${TASK}" == "retention_time" && ! "${MODE}" =~ ^(all|ordinal|regression)$ ]]; then
    echo "Retention-time mode must be all, ordinal, or regression." >&2
    exit 2
fi

cd /path/to/dIon
PYTHON_BIN="${PYTHON_BIN:-python}"
HYBRID_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/epoch=219-dinov2_val_loss_epoch=5.71.ckpt

case "${TASK}" in
    chimericity)
        MASTER_HYBRID=configs/master_chimericity_hye.yaml
        MASTER_RANDOM=configs/master_chimericity_hye_random.yaml
        MASTER_BINNED=configs/master_chimericity_hye_binned.yaml
        DOWNSTREAM_CONFIG=configs/downstream/chimericity_hye.yaml
        WANDB_PROJECT=ms2-chimericity
        ;;
    oxidized_met)
        MASTER_HYBRID=configs/master_oxidized_met.yaml
        MASTER_RANDOM=configs/master_oxidized_met_random.yaml
        MASTER_BINNED=configs/master_oxidized_met_binned.yaml
        DOWNSTREAM_CONFIG=configs/downstream/oxidized_met.yaml
        WANDB_PROJECT=ms2-oxidized-met
        ;;
    retention_time)
        MASTER_HYBRID=configs/master_retention_time_probe.yaml
        MASTER_RANDOM=configs/master_retention_time_random.yaml
        MASTER_BINNED=configs/master_retention_time_binned.yaml
        WANDB_PROJECT=ms2-retention-time
        ;;
esac

run_one() {
    local label="$1"
    local checkpoint_glob="$2"
    local master_config="$3"
    local downstream_config="$4"
    local freeze_encoder="$5"
    local conditioning="$6"
    local use_mass="$7"
    local use_charge="$8"
    shift 8
    local -a extra_args=("$@")
    local checkpoint_dir checkpoint

    checkpoint_dir=$(compgen -G "${checkpoint_glob}" | head -n 1 || true)
    if [[ -z "${checkpoint_dir}" ]]; then
        echo "No checkpoint directory matches ${checkpoint_glob}" >&2
        return 2
    fi

    checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' -print -quit)
    if [[ -z "${checkpoint}" ]]; then
        echo "No selected checkpoint under ${checkpoint_dir}" >&2
        return 2
    fi

    local run_name="${TASK}_full_validation_${label}_$(date +%Y%m%dT%H%M%S)"
    echo "Evaluating ${label}: ${checkpoint}"
    WANDB_NAME="${run_name}" WANDB_RUN_GROUP="${TASK}_auxiliary_full_validation_v1" \
        "${PYTHON_BIN}" -u -m src.main \
        --config "${master_config}" \
        --downstream_config "${downstream_config}" \
        --downstream_weights "${checkpoint}" \
        "${extra_args[@]}" \
        --freeze_encoder "${freeze_encoder}" \
        --precursor_conditioning "${conditioning}" \
        --use_mass "${use_mass}" \
        --use_charge "${use_charge}" \
        --accelerator gpu \
        --num_devices 1 \
        --num_nodes 1 \
        --strategy ddp \
        --precision bf16-mixed \
        --num_workers 8 \
        --pin_mem 1 \
        --eval_only 1 \
        --validate_on_end 1 \
        --test_on_end 0 \
        --save_top_k 0 \
        --save_last 0 \
        --embedding_dir "/tmp/${run_name}/embedding_cache" \
        --output_dir "/tmp/${run_name}" \
        --log_dir "/tmp/${run_name}/logs" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity user \
        --log_wandb 1
}

if [[ "${TASK}" == "retention_time" ]]; then
    MODES=(ordinal regression)
    if [[ "${MODE}" != "all" ]]; then
        MODES=("${MODE}")
    fi
    for mode in "${MODES[@]}"; do
        DECODER_ARGS=()
        if [[ "${mode}" == "regression" ]]; then
            DECODER_ARGS=(--decoder_model linear_regression_head)
        fi
        downstream_config="configs/downstream/retention_time_${mode}_probe.yaml"
        [[ "${mode}" == "ordinal" ]] && downstream_config=configs/downstream/retention_time_probe.yaml
        run_one "random_${mode}_frozen" \
            "/path/to/checkpoints/retention_time/retention_time_random_${mode}_frozen_17499916_$([[ "${mode}" == ordinal ]] && echo 0 || echo 1)_"* \
            "${MASTER_RANDOM}" "${downstream_config}" 1 conditioned 1 1 "${DECODER_ARGS[@]}"
        run_one "hybrid300e219_conditioned_${mode}_frozen" \
            "/path/to/checkpoints/retention_time/retention_time_hybrid300e219_conditioned_${mode}_frozen_17499916_$([[ "${mode}" == ordinal ]] && echo 2 || echo 3)_"* \
            "${MASTER_HYBRID}" "${downstream_config}" 1 conditioned 1 1 --encoder_weights "${HYBRID_CKPT}" "${DECODER_ARGS[@]}"
        run_one "hybrid300e219_null_${mode}_frozen" \
            "/path/to/checkpoints/retention_time/retention_time_hybrid300e219_null_${mode}_frozen_17499916_$([[ "${mode}" == ordinal ]] && echo 4 || echo 5)_"* \
            "${MASTER_HYBRID}" "${downstream_config}" 1 null 1 1 --encoder_weights "${HYBRID_CKPT}" "${DECODER_ARGS[@]}"
        run_one "binned1024_precursor_metadata_${mode}_frozen" \
            "/path/to/checkpoints/retention_time/retention_time_binned1024_precursor_metadata_${mode}_frozen_17499916_$([[ "${mode}" == ordinal ]] && echo 6 || echo 7)_"* \
            "${MASTER_BINNED}" "${downstream_config}" 1 conditioned 1 1 "${DECODER_ARGS[@]}"
        run_one "binned1024_spectrum_only_${mode}_frozen" \
            "/path/to/checkpoints/retention_time/retention_time_binned1024_spectrum_only_${mode}_frozen_17499916_$([[ "${mode}" == ordinal ]] && echo 8 || echo 9)_"* \
            "${MASTER_BINNED}" "${downstream_config}" 1 conditioned 0 0 "${DECODER_ARGS[@]}"
    done
else
    run_one "scratch_conditioned_finetune" \
        "/path/to/checkpoints/${TASK}/${TASK}_scratch_conditioned_finetune_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_0_"* \
        "${MASTER_RANDOM}" "${DOWNSTREAM_CONFIG}" 0 conditioned 1 1
    run_one "hybrid300e219_conditioned_frozen" \
        "/path/to/checkpoints/${TASK}/${TASK}_hybrid300e219_conditioned_frozen_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_1_"* \
        "${MASTER_HYBRID}" "${DOWNSTREAM_CONFIG}" 1 conditioned 1 1 --encoder_weights "${HYBRID_CKPT}"
    run_one "hybrid300e219_null_frozen" \
        "/path/to/checkpoints/${TASK}/${TASK}_hybrid300e219_null_frozen_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_2_"* \
        "${MASTER_HYBRID}" "${DOWNSTREAM_CONFIG}" 1 null 1 1 --encoder_weights "${HYBRID_CKPT}"
    run_one "hybrid300e219_conditioned_finetune" \
        "/path/to/checkpoints/${TASK}/${TASK}_hybrid300e219_conditioned_finetune_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_3_"* \
        "${MASTER_HYBRID}" "${DOWNSTREAM_CONFIG}" 0 conditioned 1 1 --encoder_weights "${HYBRID_CKPT}"
    run_one "hybrid300e219_null_finetune" \
        "/path/to/checkpoints/${TASK}/${TASK}_hybrid300e219_null_finetune_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_4_"* \
        "${MASTER_HYBRID}" "${DOWNSTREAM_CONFIG}" 0 null 1 1 --encoder_weights "${HYBRID_CKPT}"
    run_one "binned1024_precursor_metadata_frozen" \
        "/path/to/checkpoints/${TASK}/${TASK}_binned1024_precursor_metadata_frozen_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_5_"* \
        "${MASTER_BINNED}" "${DOWNSTREAM_CONFIG}" 1 conditioned 1 1
    run_one "binned1024_spectrum_only_frozen" \
        "/path/to/checkpoints/${TASK}/${TASK}_binned1024_spectrum_only_frozen_1749990$([[ "${TASK}" == chimericity ]] && echo 2 || echo 3)_6_"* \
        "${MASTER_BINNED}" "${DOWNSTREAM_CONFIG}" 1 conditioned 0 0
fi
