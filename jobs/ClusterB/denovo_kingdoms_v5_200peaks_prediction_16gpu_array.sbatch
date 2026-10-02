#!/usr/bin/env bash
# Derived from denovo_mskb_final_encoderld_16gpu_array.sbatch; eval-only Kingdoms variant.
#SBATCH --job-name=kingdoms-v5-predict
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err
set -euo pipefail
PERSONAL=/path/to/work; REPO=$PERSONAL/repos/dion-runs/denovo-mskb-final; DATA=/path/to/data; ROOT=$DATA/denovo_kingdoms/run_disjoint_v1_species_cap100k; SIF=$PERSONAL/containers/dion-ngc-26.06-slingshot.sif
case $SLURM_ARRAY_TASK_ID in
0) L=hybrid300; P=200; B=100; C=$PERSONAL/runs/dion/checkpoints/denovo_dnlv1/denovo_dnlv1_encoderld_hybrid300_conditioned_2344260_2/checkpoint_06_55_38_836668__13_09_26/epoch=37-denovo_tf_val_pep_prec=0.60.ckpt;;
1) L=scratch_encoderld; P=200; B=100; C=$PERSONAL/runs/dion/checkpoints/denovo_dnlv1/denovo_dnlv1_encoderld_scratch_encoderld_conditioned_2346267_0/checkpoint_04_27_32_377495__13_09_26/epoch=41-denovo_tf_val_pep_prec=0.57.ckpt;;
esac
cd $REPO; test -f "$C"; test -f "$ROOT/manifest.json"; R=$PERSONAL/runs/dion/results/denovo_kingdoms_species_cap100k/200_peaks_prediction_mode/v5/${L}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$R/logs"
srun --ntasks=16 --ntasks-per-node=4 --mpi=pmix --kill-on-bad-exit=1 apptainer exec --nv --bind "$PERSONAL:$PERSONAL" --bind "$DATA:$DATA" "$SIF" python -u -m src.main --config config_cluster_b/master_denovo_dion_hybrid_mskb_final.yaml --downstream_config config_cluster_b/downstream/denovo_kingdoms_200peaks_prediction.yaml --downstream_weights "$C" --downstream_root_dir "$ROOT" --output_dir "$R/checkpoint" --log_dir "$R/logs" --wandb_project denovo-kingdoms-v5-prediction-mode-predict --wandb_entity user --log_wandb 0 --accelerator gpu --num_devices 4 --num_nodes 4 --strategy ddp --num_workers 4 --pin_mem 1 --batch_size "$B" --max_peaks "$P" --disable_cudnn_sdp 1 --eval_only 1 --validate_on_end 0 --test_on_end 1 --limit_test_batches 1.0 --save_top_k 0 --save_last 0
