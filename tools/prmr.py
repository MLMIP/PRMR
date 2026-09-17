import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_visible_devices():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-id", default=None)
    args, _ = parser.parse_known_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


configure_visible_devices()

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from diffusion_model import DiffusionModel  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import get_dist_info, init_dist, load_checkpoint  # noqa: E402
from mmcv.utils import DictAction  # noqa: E402
from mmseg.apis import single_gpu_prmr  # noqa: E402
from mmseg.datasets import build_dataloader, build_dataset  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "local_configs"
    / "segformer"
    / "B5"
    / "segformer.b5.1024x1024.acdc.160k.py"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT/ "ckpt" /"segformer.b5.1024x1024.city.160k.pth"
DEFAULT_OUT = PROJECT_ROOT / "work_dirs" / "res.pkl"
DEFAULT_DIFFUSION_PARAMS = PROJECT_ROOT / "ckpt" / "params_uncondition.yml"
DEFAULT_DIFFUSION_CHECKPOINT = PROJECT_ROOT / "ckpt" / "diffusion_model.pt"
DEFAULT_NUM_ROUNDS = 3


def set_random_seed(seed=1, deterministic=False):
    """Set random seed for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_ema_model(model):
    ema_model = deepcopy(model)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.detach_()
        ema_param.data.copy_(param.data)
    return ema_model


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    """Update teacher parameters with exponential moving average."""
    if iteration is not None:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    return ema_model


def set_multi_scale_aug(test_cfg):
    if test_cfg.type in ["CityscapesDataset", "ACDCDataset"]:
        img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    elif test_cfg.type == "ADE20KDataset":
        img_ratios = [0.75, 0.875, 1.0, 1.125, 1.25]
    else:
        img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    test_cfg.pipeline[1].img_ratios = img_ratios
    test_cfg.pipeline[1].flip = True


def validate_args(args):
    has_operation = any(
        [args.out, args.eval, args.format_only, args.show, args.show_dir]
    )
    if not has_operation:
        raise ValueError(
            "Please specify at least one operation with --out, --eval, "
            "--format-only, --show or --show-dir."
        )

    if args.eval and "None" in args.eval:
        args.eval = None
    if args.eval and args.format_only:
        raise ValueError("--eval and --format-only cannot be both specified.")
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRMR test-time adaptation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="test config file path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CONFIG), help="segmentation checkpoint file")
    parser.add_argument("--diffusion-params", default=str(DEFAULT_DIFFUSION_PARAMS))
    parser.add_argument("--diffusion-checkpoint", default=str(DEFAULT_DIFFUSION_CHECKPOINT))
    parser.add_argument("--gpu-id", default=None, help="CUDA_VISIBLE_DEVICES value")

    parser.add_argument("--aug-test", action="store_true", help="Use flip and multi-scale aug")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output result file in pickle format")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help=(
            "Format the output results without performing evaluation. Useful "
            "when submitting results to a test server."
        ),
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=["mIoU"],
        help='evaluation metrics, e.g. "mIoU" or "cityscapes"',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple workers",
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="custom options")
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--model_lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--time_step", type=float, default=180)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--memory_bank_threshold", type=float, default=0.7)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--wandb_login", type=str)
    parser.add_argument("--wandb_project", type=str, default="3-round")
    parser.add_argument("--wandb_name", type=str, default="debug")
    parser.add_argument("--wandb_key", default=os.getenv("WANDB_API_KEY"))

    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return args


def build_data_loaders(datasets, cfg, distributed):
    return [
        build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        for dataset in datasets
    ]


def main():
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(
        name=args.wandb_name,
        project=args.wandb_project,
        entity=args.wandb_login,
        mode="offline",
        save_code=False,
        config=args,
    )

    print("PRMR parameters:")
    print(f"seed: {args.seed}")
    print(f"time_step: {args.time_step}")
    print(f"threshold: {args.threshold}")

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if True:
        for split_name in ["test", "test1", "test2", "test3"]:
            set_multi_scale_aug(getattr(cfg.data, split_name))

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    datasets = [
        build_dataset(cfg.data.test),
        build_dataset(cfg.data.test1),
        build_dataset(cfg.data.test2),
        build_dataset(cfg.data.test3),
    ]

    diffusion_model = DiffusionModel(args.diffusion_params, args.diffusion_checkpoint)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint["meta"]["CLASSES"]
    model.PALETTE = checkpoint["meta"]["PALETTE"]

    efficient_test = True
    if args.eval_options is not None:
        efficient_test = args.eval_options.get("efficient_test", False)

    model = MMDataParallel(model, device_ids=[0])
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)

    total_miou = 0.0
    for round_idx in range(args.num_rounds):
        mean_miou = 0.0
        print(f"revisiting {round_idx}")
        data_loaders = build_data_loaders(datasets, cfg, distributed)

        for dataset, data_loader in zip(datasets, data_loaders):
            outputs = single_gpu_prmr(
                args,
                model,
                diffusion_model,
                data_loader,
                args.show,
                args.show_dir,
                efficient_test,
                anchor,
                ema_model,
                anchor_model,
            )

            rank, _ = get_dist_info()
            if rank != 0:
                continue

            if args.out:
                print(f"\nwriting results to {args.out}")
                mmcv.dump(outputs, args.out)

            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                results = dataset.evaluate(outputs, args.eval, **kwargs)
                miou = results["mIoU"]
                mean_miou += miou
                wandb.log({"mIoU": miou})

        round_miou = mean_miou / len(datasets)
        total_miou += round_miou
        wandb.log({"mean_mIoU": round_miou})

    wandb.log({"All_mIoU": total_miou / args.num_rounds})
    wandb.finish()


if __name__ == "__main__":
    main()
