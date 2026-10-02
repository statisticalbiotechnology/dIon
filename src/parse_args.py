import argparse
import os
import datetime
import json
import time
import yaml


def _int_or_float(value: str):
    """Parse numeric CLI values while preserving ints when possible."""
    v = value.strip()
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid numeric value: '{value}'"
        ) from exc


def get_args_parser(conf_parser):
    parser = argparse.ArgumentParser(
        "Train unsupervised transformers", parents=[conf_parser]
    )
    # Model parameters
    parser.add_argument(
        "--encoder_model",
        default="abc",
        type=str,
        help="Name of the encoder model to train",
    )
    parser.add_argument(
        "--decoder_model",
        default="",
        type=str,
        help="Name of the decoder model to train",
    )
    parser.add_argument(
        "--encoder_weights",
        default=None,
        type=str,
        help="Path to checkpoint of previously trained encoder weights",
    )
    parser.add_argument(
        "--embedding_readout",
        default="backbone",
        choices=["backbone", "dino_bottleneck", "dino_logits"],
        help=(
            "Embedding representation for standalone DINO evaluation: pooled "
            "EMA-teacher backbone features, the normalized DINO-head bottleneck, "
            "or full DINO prototype logits."
        ),
    )
    parser.add_argument(
        "--counterfactual_distance",
        default="cosine",
        choices=["cosine", "euclidean", "jensen_shannon"],
        help=(
            "Distance used only by evaluate_precursor_counterfactual.py. "
            "Jensen-Shannon requires --embedding_readout dino_logits."
        ),
    )
    parser.add_argument(
        "--counterfactual_output_report",
        default="",
        type=str,
        help="Optional report path used only by evaluate_precursor_counterfactual.py.",
    )
    parser.add_argument(
        "--embedding_baseline",
        default="model",
        choices=["model", "precursor_metadata", "binned_spectrum"],
        help=(
            "Embedding source for standalone evaluation. precursor_metadata uses "
            "only standardized log precursor mass and charge. binned_spectrum uses "
            "the fixed metadata-free intensity-binned peak vector; neither needs a checkpoint."
        ),
    )
    parser.add_argument(
        "--downstream_weights",
        default="",
        type=str,
        help="Path to checkpoint of previously trained downstream weights",
    )
    parser.add_argument(
        "--pretraining_task",
        default="dion",
        choices=[
            "dion",
            "dummy",
        ],
        type=str,
        help="Which pretraining strategy to use",
    )
    parser.add_argument(
        "--pretrain_config",
        default="configs/pretrain/dion_noibot_nokoleo_default.yaml",
        type=str,
        help="Path of the pretraining config",
    )
    parser.add_argument(
        "--downstream_config",
        default="configs/downstream/denovo_mskb.yaml",
        type=str,
        help="Path of the downstream config",
    )
    parser.add_argument(
        "--probing_config",
        default="",
        type=str,
        help="Path to the probing tasks config file (e.g. embedding_probes.yaml)",
    )
    parser.add_argument(
        "--probe_on_fit_start",
        default=0,
        type=int,
        help="Bool (0/1): run linear probing once at fit start before training.",
    )
    parser.add_argument(
        "--probe_every_n_steps",
        default=1000,
        type=int,
        help="Run linear probing every N training steps.",
    )
    parser.add_argument(
        "--downstream_task",
        default="none",
        choices=["denovo_tf", "denovo_random", "sqa", "chimericity", "oxidized_met", "retention_time", "metric_learning", "none"],
        type=str,
        help="Which finetuning task to perform",
    )
    parser.add_argument(
        "--downstream_encoder",
        default="best",
        choices=["best", "last"],
        type=str,
        help="If pretraining, use the best/last encoder checkpoint achieved during pretraining",
    )
    parser.add_argument(
        "--watch_model",
        default=0,
        type=int,
        help="Bool (0/1): toggle logging of weights to WandB",
    )
    parser.add_argument(
        "--pretrain",
        default=1,
        type=int,
        help="Bool (0/1): toggle pretraining",
    )
    parser.add_argument(
        "--use_mass",
        default=0,
        type=int,
        help="Bool (0/1): input precursor mass",
    )
    parser.add_argument(
        "--use_energy",
        default=0,
        type=int,
        help="Bool (0/1): input energy",
    )
    parser.add_argument(
        "--use_charge",
        default=0,
        type=int,
        help="Bool (0/1): input precursor charge",
    )
    parser.add_argument(
        "--precursor_conditioning",
        default="conditioned",
        choices=["conditioned", "null"],
        help=(
            "Encoder precursor mode for downstream/evaluation inputs. null uses "
            "mass=0 and the learned charge-0 null embedding."
        ),
    )
    parser.add_argument(
        "--max_charge",
        default=10,
        type=int,
        help="The maximally allowed precursor charge.",
    )
    parser.add_argument(
        "--min_peaks",
        default=0,
        type=int,
        help="Drop batch members shorter than this.",
    )
    parser.add_argument(
        "--mask_zero_tokens",
        default=1,
        type=int,
        help="Bool (0/1): mask the attention for 'null' tokens",
    )
    parser.add_argument(
        "--max_peaks",
        default=300,
        type=int,
        help="The maximally allowed number of peaks. Spectra that have more peaks are subsampled by the maximum intensities.",
    )
    parser.add_argument(
        "--peak_filter_method",
        default="default",
        choices=["default", "casanovo", "top_intensity_minmax"],
        help="Method used to filter the peaks. Choose 'default' for basic subsampling and intensity scaling, "
        "or 'casanovo' for Casanovo-specific filtering, or 'top_intensity_minmax' for unwindowed top-intensity selection. ",
    )
    parser.add_argument(
        "--intensity_scaling",
        default="basepeak",
        choices=["minmax", "basepeak", "none"],
        help="Intensity scaling applied after default peak subsampling. "
        "Ignored for Casanovo filtering.",
    )
    parser.add_argument(
        "--min_mz",
        default=0,
        type=float,
        help="Minimum m/z value allowed. Peaks with lower m/z are discarded.",
    )
    parser.add_argument(
        "--max_mz",
        default=2500,
        type=float,
        help="Maximum m/z value allowed. Peaks with higher m/z are discarded.",
    )
    parser.add_argument(
        "--min_intensity",
        default=0.01,
        type=float,
        help="(Casanovo-specific) Minimum intensity value in (percent of the most intense peak) allowed. ",
    )
    parser.add_argument(
        "--remove_precursor_tol",
        default=2.0,
        type=float,
        help="(Casanovo-specific) Tolerance (in Da) to remove peaks within the specified distance from the precursor m/z. "
        "Set to <=0 to disable this filtering.",
    )
    parser.add_argument(
        "--max_length",
        default=30,
        type=int,
        help="The maximally allowed length of peptides. Longer peptides will be truncated to this. ",
    )
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--max_steps", default=-1, type=int, help="Maximum optimizer steps; -1 disables the limit.")
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--anneal_lr",
        type=int,
        default=0,
        help="Bool (0/1): Turn on cosine annealing lr",
    )
    parser.add_argument(
        "--lr_warmup",
        type=int,
        default=0,
        help="Bool (0/1): Turn on linear warmup lr",
    )
    parser.add_argument(
        "--lr_start",
        type=float,
        default=1e-7,
        help="Starting learning rate",
    )
    parser.add_argument(
        "--lr_end",
        type=float,
        default=2e-4,
        help="Final learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=2e4,
        help="Number of steps to increase lr for",
    )
    parser.add_argument(
        "--lr_decay",
        type=int,
        default=0,
        help="Bool (0/1): Turn on lr decay",
    )
    parser.add_argument(
        "--lr_decay_start_step",
        type=int,
        default=100000,
        help="Starting step for lr decay",
    )
    parser.add_argument(
        "--lr_decay_end_step",
        type=int,
        default=120000,
        help="Final step for lr decay",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.999,
        help="Number multiplied to learning rate each step",
    )

    parser.add_argument(
        "--scale_lr_by_batchsize",
        type=int,
        default=0,
        help="Bool (0/1): Turns on MAE-style lr scaling which multiplies lr by (eff_batch_size / 256)",
    )
    parser.add_argument(
        "--cls_token",
        type=int,
        default=0,
        help="Bool (0/1): Adds a learned CLS token to the input of the encoders",
    )

    parser.add_argument(
        "--cross_attend",
        type=int,
        default=1,
        help="Bool (0/1): 1 = Decoder cross-attends encoder output",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=_int_or_float,
        default=1.0,
        help="Train batch limit: int for num batches, float in [0,1] for fraction.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=0,
        help="Maximum number of batches to run in validation",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=_int_or_float,
        default=1.0,
        help="Test batch limit: int for num batches, float in [0,1] for fraction.",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=1,
        help="Limit validation frequency; used as argument to downstream trainer",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_root_dir",
        default="../../datasets/instanovo_data_subset",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--downstream_root_dir",
        default="/path/to/data/proteomics/foundational_model/ninespecies_xy/",
        type=str,
        help="dataset path for the denovo task",
    )
    parser.add_argument(
        "--downstream_train_path",
        default="",
        type=str,
        help="dataset path for the denovo task",
    )
    parser.add_argument(
        "--downstream_val_path",
        default="",
        type=str,
        help="dataset path for the denovo task",
    )
    parser.add_argument(
        "--downstream_test_path",
        default="",
        type=str,
        help="dataset path for the denovo task",
    )
    parser.add_argument(
        "--output_dir",
        default="outs/checkpoint",
        help="path where to save",
    )
    parser.add_argument("--log_dir", default="outs/log", help="path where to log")
    parser.add_argument(
        "--barebones",
        type=int,
        default=0,
        help="Bool (0/1): barebones mode",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="saves top-K checkpoints based on 'val_loss' metric",
    )
    parser.add_argument(
        "--save_last",
        type=int,
        default=1,
        help="Bool (0/1): saves checkpoint from last epoch",
    )
    parser.add_argument(
        "--every_n_epochs",
        type=int,
        default=None,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=0,
        help="(int) If >0, early stop with patience = this int",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume", default=0, type=int, help="Bool (0/1): resume from checkpoint"
    )
    parser.add_argument(
        "--eval_only",
        default=0,
        type=int,
        help="Bool (0/1): skip training and do downstream eval",
    )
    parser.add_argument(
        "--validate_on_end",
        default=0,
        type=int,
        help="Bool (0/1): do an additional validation pass after training",
    )
    parser.add_argument(
        "--test_on_end",
        default=1,
        type=int,
        help="Bool (0/1): test pass after training",
    )
    parser.add_argument(
        "--remove_ckpt",
        default=0,
        type=int,
        help="Bool (0/1): remove final checkpoint to save space",
    )
    # parser.add_argument("--start_epoch", default=0, type=int, help="start epoch") # I think lightning detects the starting epoch from the checkpoint
    parser.add_argument(
        "--num_workers",
        default=-1,
        type=int,
        help="If >= 0, set global num_workers, else let the task dict control num_workers for each task",
    )
    parser.add_argument(
        "--batch_size",
        default=-1,
        type=int,
        help="If >= 0, set global batch_size, else let the task dict control batch_size for each task",
    )
    parser.add_argument(
        "--pin_mem",
        type=int,
        default=0,
        help="Bool (0/1): Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    # distributed training parameters
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default="gpu",
        help="Specify the accelerator type",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16-mixed", "bf16-mixed", "32-true", "64-true"],
        default="32-true",
        help=" Double precision, full precision (32), 16bit mixed precision or bfloat16 mixed precision",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        choices=["medium", "high", None],
        default=None,
        help="To fully exploit NVIDIA A100 GPUs, set torch.set_float32_matmul_precision('medium' | 'high') which trades off precision for performance",
    )
    parser.add_argument(
        "--disable_cudnn_sdp",
        type=int,
        default=0,
        help="Bool (0/1): disable cuDNN SDPA so PyTorch selects another fused attention backend.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ddp", "deepspeed"],
        default="ddp",
        help="Specify the distributed strategy",
    )
    parser.add_argument(
        "--num_devices", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes")

    parser.add_argument("--clip_grad", type=float, default=None, help="")
    parser.add_argument(
        "--log_wandb", default=1, type=int, help="Disable WandB logging by setting to 0"
    )
    parser.add_argument(
        "--wandb_project", default=None, help="Specify project name to log using WandB"
    )
    parser.add_argument(
        "--wandb_entity", default=None, help="Entity to log as on WandB"
    )
    parser.add_argument(
        "--loss_type",
        choices=["mse", "ce", "bce"],
        default="mse",
        help="MSE, CE or Binary CE (latter only makes sense for binary targets)",
    )
    parser.add_argument(
        "--profile_flops",
        type=int,
        default=0,
        help="Bool (0/1): Measure forward pass FLOPs on the first train batch",
    )
    parser.add_argument(
        "--subset",
        default=0,
        type=float,
        help="Fraction in [0, 1]. Train on a subset of the data for debugging purposes. 0 = full data",
    )
    parser.add_argument(
        "--freeze_encoder",
        type=int,
        default=0,
        help="Bool (0/1): If set, freeze encoder (and use cached embeddings in SQA)",
    )
    parser.add_argument(
        "--embedding_dir",
        default="/path/to/project/embeddings_cache",
        type=str,
        help="Directory to store/load precomputed encoder embeddings for downstream tasks",
    )
    parser.add_argument(
        "--external_embedding_cache",
        default="",
        type=str,
        help=(
            "Complete split-indexed frozen embedding cache produced by "
            "materialize_external_downstream_embeddings.py. This is only valid "
            "for frozen SQA-derived downstream tasks."
        ),
    )

    return parser


def sanity_checks(args):
    # add sanity checks, i.e. for args that should be mutually exclusive here
    ...
    # make sure int booleans are bool
    args.log_wandb = bool(args.log_wandb)
    args.save_last = bool(args.save_last)
    args.use_mass = bool(args.use_mass)
    args.use_charge = bool(args.use_charge)
    args.use_energy = bool(args.use_energy)
    args.mask_zero_tokens = bool(args.mask_zero_tokens)
    args.anneal_lr = bool(args.anneal_lr)
    args.scale_lr_by_batchsize = bool(args.scale_lr_by_batchsize)
    args.resume = bool(args.resume)
    args.eval_only = bool(args.eval_only)
    args.validate_on_end = bool(args.validate_on_end)
    args.test_on_end = bool(args.test_on_end)
    args.pin_mem = bool(args.pin_mem)
    # args.data_in_memory = bool(args.data_in_memory)
    args.profile_flops = bool(args.profile_flops)
    args.pretrain = bool(args.pretrain)
    args.watch_model = bool(args.watch_model)
    args.cls_token = bool(args.cls_token)
    args.cross_attend = bool(args.cross_attend)
    args.freeze_encoder = bool(args.freeze_encoder)
    args.probe_on_fit_start = bool(args.probe_on_fit_start)
    if args.probe_every_n_steps < 1:
        raise ValueError("probe_every_n_steps must be >= 1")


def uniquify_path(path):
    # append a number
    counter = 1
    try_path = path
    if not os.path.exists(try_path):
        return path
    else:
        while os.path.exists(try_path):
            try_path = path + "__" + str(counter)
            counter += 1

        return try_path


def parse_args_and_config():
    ### Priority: provided command line args > config values > argparse defaults

    # parse the config arg only
    conf_parser = argparse.ArgumentParser("Config parser", add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        help="config path",
    )
    conf_args, remaining_args = conf_parser.parse_known_args()

    # open config file and set default args to those included in the config
    try:
        with open(conf_args.config, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print("Error occurred while loading the configuration file:")
        print(e)

    parser = get_args_parser(conf_parser)
    parser.set_defaults(**config)

    # parse the rest of the args and override defaults/config
    args = parser.parse_args(remaining_args)
    sanity_checks(args)

    if bool(args.pretrain_config):
        try:
            with open(args.pretrain_config, "r") as f:
                pretrain_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(
                f"Error occurred while loading the configuration file: {args.downstream_config}"
            )
            print(e)
    else:
        pretrain_config = None

    if bool(args.downstream_config):
        try:
            with open(args.downstream_config, "r") as f:
                ds_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(
                f"Error occurred while loading the configuration file: {args.downstream_config}"
            )
            print(e)
    else:
        ds_config = None

    # ─── new probing config ────────────────────────────────────────────────────
    if bool(args.probing_config):
        try:
            with open(args.probing_config, "r") as f:
                probing_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(
                f"Error occurred while loading the probing config {args.probing_config}:"
            )
            print(e)
            probing_config = None
    else:
        probing_config = None
    # ───────────────────────────────────────────────────────────────────────────

    return args, pretrain_config, ds_config, probing_config


def _output_dir_rendezvous_path(output_dir: str) -> str | None:
    """Return a job-unique pre-DDP rendezvous path, when Slurm launched >1 rank."""
    job_id = os.environ.get("SLURM_JOB_ID")
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    if job_id and world_size > 1:
        return f"{output_dir}.paths_{job_id}.json"
    return None


def create_output_dirs(args, is_main_process=True):
    """Create one timestamped run directory and publish it to every launch rank.

    Lightning initializes process groups after this function runs. Under ``srun``,
    rank zero previously received the timestamped directory while other ranks kept
    the unsuffixed path, so their checkpoint callbacks disagreed about ``dirpath``.
    A tiny Slurm-job-scoped file rendezvous gives every rank the same two paths
    before Lightning constructs those callbacks.
    """
    rendezvous_path = _output_dir_rendezvous_path(args.output_dir)
    if is_main_process:
        now = datetime.datetime.now()
        cur_time = now.strftime("_%H_%M_%S_%f__%d_%m_%y")
        args.output_dir = uniquify_path(args.output_dir + cur_time)
        os.makedirs(args.output_dir, exist_ok=False)
        args.log_dir = uniquify_path(args.log_dir + cur_time)
        os.makedirs(args.log_dir, exist_ok=False)

        if rendezvous_path:
            payload = {"output_dir": args.output_dir, "log_dir": args.log_dir}
            temp_path = f"{rendezvous_path}.{os.getpid()}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp_path, rendezvous_path)
    elif rendezvous_path:
        deadline = time.monotonic() + 120
        while not os.path.exists(rendezvous_path):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for rank zero to publish distributed run paths: "
                    f"{rendezvous_path}"
                )
            time.sleep(0.1)
        with open(rendezvous_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        args.output_dir = payload["output_dir"]
        args.log_dir = payload["log_dir"]


if __name__ == "__main__":
    # test code
    args = parse_args_and_config()
    # create_output_dirs(args)
