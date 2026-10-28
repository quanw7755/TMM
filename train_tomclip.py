"""
  @Author: 王权
  @FileName: train_tomclip.py
  @DateTime: 2026/8/10 21:52
  @SoftWare: PyCharm
"""
from clip.clip import load, tokenize, available_models

import argparse
import copy
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from dataset import *
from dataset.mpdd import MPDDDataset
from util.utils import eval_all_class
from util.geometry_eval import build_normal_geometry, diagnose_geometry_dataset


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float().clamp(1e-6, 1.0 - 1e-6)
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs.float(), targets.float(), reduction=reduction)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name or filename)
    logger.setLevel(level_dict[verbosity])
    logger.propagate = False

    # Avoid duplicate handlers when the script is re-entered interactively.
    logger.handlers.clear()

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def print_args(logger, args):
    logger.info("--------args----------")
    for k in list(vars(args).keys()):
        logger.info("{}: {}".format(k, vars(args)[k]))
    logger.info("--------args----------\n")


def patch_alignment_loss(img_tokens, gts):
    """
    Original AF-CLIP pairwise patch-alignment objective, with safer tensor
    cloning. This is kept for warm-up/backward compatibility, but the new
    geometry-aware stage can disable it via --patch_align_after_geo 0.
    """
    gts = gts.reshape(img_tokens[0].size(0), -1)
    new_gts = gts.clone()

    if (new_gts == 0).sum().item() == 0:
        return img_tokens[0].new_tensor(0.0)

    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()
    mask = torch.matmul(
        new_gts.reshape(b, l, 1),
        new_gts.reshape(b, 1, l),
    )

    total_sim = img_tokens[0].new_tensor(0.0)
    valid_layers = 0
    for img_token in img_tokens:
        patch_token = F.normalize(img_token[:, 1:, :], dim=-1)
        sim_matrix = torch.matmul(patch_token, patch_token.transpose(1, 2))

        cross = sim_matrix[mask == -1]
        same = sim_matrix[mask == 1]
        if cross.numel() == 0 or same.numel() == 0:
            continue

        value = cross.mean() - same.mean()
        total_sim = total_sim + F.relu(value)
        valid_layers += 1

    if valid_layers == 0:
        return img_tokens[0].new_tensor(0.0)
    return total_sim / valid_layers


def resize_gt_to_map(gts, map_tensor):
    """Resize Bx1xHxW GT masks to the spatial size of a predicted map."""
    if gts.dim() == 3:
        gts = gts.unsqueeze(1)
    gts = F.interpolate(
        gts.float(),
        size=map_tensor.shape[-2:],
        mode="nearest",
    )
    return (gts >= 0.5).float()


def geometry_training_loss(aux, labels, gts, args):
    """
    Geometry-aware refinement objective.

    Normal term:
        uses patches from image-level normal samples only.

    Separation term:
        asks real anomalous pixels to have a larger off-manifold score than
        pure-normal patches by at least --geo_sep_margin.

    No objective forces anomaly vectors to be fully orthogonal to the tangent
    space; only a relative separation is required.
    """
    if "off_ratio" not in aux:
        zero = aux["semantic_patch"].new_tensor(0.0)
        return zero, {
            "geo_normal": 0.0,
            "geo_sep": 0.0,
            "normal_mean": float("nan"),
            "anomaly_mean": float("nan"),
        }

    geo_patch = aux["geo_patch"]  # [B, L]
    b, l = geo_patch.shape
    side = int(math.sqrt(l))
    if side * side != l:
        raise ValueError(f"Patch count {l} is not square.")

    if gts.dim() == 3:
        gts = gts.unsqueeze(1)
    gt_patch = F.interpolate(
        gts.float(),
        size=(side, side),
        mode="nearest",
    ).reshape(b, -1)

    labels = labels.reshape(-1)
    pure_normal_image = (labels == 0).unsqueeze(1).expand(-1, l)
    anomaly_mask = gt_patch >= 0.5

    # Normality fitting/refinement never uses background patches from an
    # anomalous image as normal supervision.
    normal_values = geo_patch[pure_normal_image]
    anomaly_values = geo_patch[anomaly_mask]

    if normal_values.numel() > 0:
        normal_loss = normal_values.pow(2).mean()
        normal_mean = normal_values.mean()
    else:
        normal_loss = geo_patch.new_tensor(0.0)
        normal_mean = geo_patch.new_tensor(0.0)

    if normal_values.numel() > 0 and anomaly_values.numel() > 0:
        anomaly_mean = anomaly_values.mean()
        sep_loss = F.relu(
            float(args.geo_sep_margin) + normal_mean - anomaly_mean
        )
    else:
        anomaly_mean = geo_patch.new_tensor(float("nan"))
        sep_loss = geo_patch.new_tensor(0.0)

    loss = normal_loss + float(args.geo_sep_weight) * sep_loss
    stats = {
        "geo_normal": float(normal_loss.detach().item()),
        "geo_sep": float(sep_loss.detach().item()),
        "normal_mean": float(normal_mean.detach().item()),
        "anomaly_mean": float(anomaly_mean.detach().item()),
    }
    return loss, stats


def save_checkpoint(path, clip_model, args, epoch, logger=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": int(epoch),
        "args": vars(args),
        "state_prompt_embedding": clip_model.state_prompt_embedding.detach().cpu(),
        "adaptor_state_dict": {
            k: v.detach().cpu() for k, v in clip_model.adaptor.state_dict().items()
        },
        "geometry": clip_model.get_geometry_state() if clip_model.geo_ready else None,
    }
    torch.save(state, path)
    if logger is not None:
        logger.info(f"Saved checkpoint: {path}")


def load_new_checkpoint(path, clip_model, device, logger=None):
    checkpoint = torch.load(path, map_location="cpu")

    prompt = checkpoint["state_prompt_embedding"]
    clip_model.state_prompt_embedding.data.copy_(prompt.to(device))
    clip_model.adaptor.load_state_dict(checkpoint["adaptor_state_dict"], strict=True)

    if checkpoint.get("geometry") is not None:
        clip_model.load_geometry_state(checkpoint["geometry"])

    if logger is not None:
        logger.info(
            f"Loaded checkpoint: {path}; geometry_ready={clip_model.geo_ready}"
        )
    return checkpoint


def load_legacy_weight_dir(
    clip_model,
    weight_dir,
    dataset_name,
    device,
    logger=None,
):
    """
    Load legacy AF-CLIP weights.

    Expected directory structure:

        weight_dir/
        ├── mvtec_prompt.pt
        └── mvtec_adaptor.pt

    or:

        weight_dir/
        ├── visa_prompt.pt
        └── visa_adaptor.pt
    """

    # =========================================================
    # 0. Basic validation
    # =========================================================

    if not isinstance(
        weight_dir,
        (str, bytes, os.PathLike)
    ):
        raise TypeError(
            "`weight_dir` must be a filesystem path, "
            f"but got {type(weight_dir)}. "
            "This usually means the arguments of "
            "load_legacy_weight_dir() were passed "
            "in the wrong order."
        )

    weight_dir = os.fspath(
        weight_dir
    )

    prompt_path = os.path.join(
        weight_dir,
        f"{dataset_name}_prompt.pt"
    )

    adaptor_path = os.path.join(
        weight_dir,
        f"{dataset_name}_adaptor.pt"
    )

    if logger is not None:
        logger.info(
            "[Legacy Weight] "
            f"Weight directory: {weight_dir}"
        )

        logger.info(
            "[Legacy Weight] "
            f"Prompt path: {prompt_path}"
        )

        logger.info(
            "[Legacy Weight] "
            f"Adaptor path: {adaptor_path}"
        )

    # =========================================================
    # 1. File existence check
    # =========================================================

    if not os.path.isfile(
        prompt_path
    ):
        raise FileNotFoundError(
            "Legacy AF-CLIP prompt checkpoint "
            f"not found:\n{prompt_path}"
        )

    if not os.path.isfile(
        adaptor_path
    ):
        raise FileNotFoundError(
            "Legacy AF-CLIP adaptor checkpoint "
            f"not found:\n{adaptor_path}"
        )

    # =========================================================
    # 2. Load prompt
    # =========================================================

    try:
        prompt_obj = torch.load(
            prompt_path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception as error:

        if logger is not None:
            logger.warning(
                "[Legacy Weight] "
                "Prompt could not be loaded with "
                "weights_only=True. Falling back to "
                "legacy loading. "
                f"Reason: {error}"
            )

        prompt_obj = torch.load(
            prompt_path,
            map_location="cpu",
            weights_only=False,
        )

    # ---------------------------------------------------------
    # Handle prompt formats
    # ---------------------------------------------------------

    if isinstance(
        prompt_obj,
        torch.nn.Parameter
    ):
        prompt_tensor = (
            prompt_obj
            .detach()
            .clone()
        )

    elif torch.is_tensor(
        prompt_obj
    ):
        prompt_tensor = (
            prompt_obj
            .detach()
            .clone()
        )

    elif isinstance(
        prompt_obj,
        dict
    ):

        if (
            "state_prompt_embedding"
            in prompt_obj
        ):
            prompt_tensor = (
                prompt_obj[
                    "state_prompt_embedding"
                ]
                .detach()
                .clone()
            )

        elif "prompt" in prompt_obj:
            prompt_tensor = (
                prompt_obj[
                    "prompt"
                ]
                .detach()
                .clone()
            )

        else:
            raise RuntimeError(
                "Unsupported prompt checkpoint "
                "dictionary format. "
                f"Available keys: "
                f"{list(prompt_obj.keys())}"
            )

    else:
        raise TypeError(
            "Unsupported prompt checkpoint "
            f"type: {type(prompt_obj)}"
        )

    # ---------------------------------------------------------
    # Shape check
    # ---------------------------------------------------------

    expected_shape = tuple(
        clip_model
        .state_prompt_embedding
        .shape
    )

    loaded_shape = tuple(
        prompt_tensor.shape
    )

    if (
        loaded_shape
        != expected_shape
    ):
        raise RuntimeError(
            "Prompt checkpoint shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Loaded:   {loaded_shape}\n"
            "Check whether prompt_len and the "
            "checkpoint configuration match."
        )

    # ---------------------------------------------------------
    # Copy prompt weights
    # ---------------------------------------------------------

    with torch.no_grad():

        clip_model.state_prompt_embedding.copy_(
            prompt_tensor.to(
                device=device,
                dtype=(
                    clip_model
                    .state_prompt_embedding
                    .dtype
                ),
            )
        )

    # =========================================================
    # 3. Load adaptor
    # =========================================================

    # Legacy AF-CLIP stores the complete Adaptor module.
    # This checkpoint is assumed to be trusted/self-generated.
    adaptor_obj = torch.load(
        adaptor_path,
        map_location="cpu",
        weights_only=False,
    )

    # ---------------------------------------------------------
    # Case A:
    # torch.save(model.adaptor, ...)
    # ---------------------------------------------------------

    if isinstance(
        adaptor_obj,
        torch.nn.Module
    ):
        adaptor_state = (
            adaptor_obj.state_dict()
        )

    # ---------------------------------------------------------
    # Case B:
    # torch.save(model.adaptor.state_dict(), ...)
    # ---------------------------------------------------------

    elif isinstance(
        adaptor_obj,
        dict
    ):

        if (
            "state_dict"
            in adaptor_obj
        ):
            adaptor_state = (
                adaptor_obj[
                    "state_dict"
                ]
            )

        elif (
            "adaptor"
            in adaptor_obj
        ):

            candidate = (
                adaptor_obj[
                    "adaptor"
                ]
            )

            if isinstance(
                candidate,
                torch.nn.Module
            ):
                adaptor_state = (
                    candidate.state_dict()
                )

            elif isinstance(
                candidate,
                dict
            ):
                adaptor_state = (
                    candidate
                )

            else:
                raise TypeError(
                    "Unsupported adaptor entry "
                    f"type: {type(candidate)}"
                )

        else:
            # Assume this dictionary itself is state_dict.
            adaptor_state = (
                adaptor_obj
            )

    else:
        raise TypeError(
            "Unsupported adaptor checkpoint "
            f"type: {type(adaptor_obj)}"
        )

    # =========================================================
    # 4. Load adaptor into current TOMCLIP model
    # =========================================================

    load_result = (
        clip_model
        .adaptor
        .load_state_dict(
            adaptor_state,
            strict=False,
        )
    )

    missing_keys = (
        load_result.missing_keys
    )

    unexpected_keys = (
        load_result.unexpected_keys
    )

    clip_model.adaptor.to(
        device
    )

    # =========================================================
    # 5. Validation
    # =========================================================

    if len(
        missing_keys
    ) > 0:
        raise RuntimeError(
            "Adaptor checkpoint has missing "
            f"parameters:\n{missing_keys}"
        )

    if len(
        unexpected_keys
    ) > 0:
        raise RuntimeError(
            "Adaptor checkpoint has unexpected "
            f"parameters:\n{unexpected_keys}"
        )

    # =========================================================
    # 6. Weight diagnostics
    # =========================================================

    prompt_norm = (
        clip_model
        .state_prompt_embedding
        .detach()
        .float()
        .norm()
        .item()
    )

    adaptor_norm = 0.0

    for parameter in (
        clip_model
        .adaptor
        .parameters()
    ):
        adaptor_norm += (
            parameter
            .detach()
            .float()
            .norm()
            .item()
        )

    if logger is not None:

        logger.info(
            "[Legacy Weight] "
            "AF-CLIP weights loaded "
            "successfully."
        )

        logger.info(
            "[Legacy Weight] "
            f"Prompt shape: "
            f"{loaded_shape}"
        )

        logger.info(
            "[Legacy Weight] "
            f"Prompt norm: "
            f"{prompt_norm:.6f}"
        )

        logger.info(
            "[Legacy Weight] "
            "Adaptor parameter norm sum: "
            f"{adaptor_norm:.6f}"
        )

        logger.info(
            "[Legacy Weight] "
            "Missing adaptor keys: "
            f"{missing_keys}"
        )

        logger.info(
            "[Legacy Weight] "
            "Unexpected adaptor keys: "
            f"{unexpected_keys}"
        )

def build_datasets(args, clip_transform, target_transform):
    test_dataset_mvtec = MVTecDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_isic = ISICDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_clinic = ClinicDBDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_colon = ColonDBDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_visa = VisaDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_btad = BTADDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_mpdd = MPDDDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_brainmri = BrainMRIDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_br35h = Br35HDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_dagm = DAGMDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )
    test_dataset_kvasir = KvasirDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform,
    )

    return {
        "mvtec": test_dataset_mvtec,
        "visa": test_dataset_visa,
        "btad": test_dataset_btad,
        "mpdd": test_dataset_mpdd,
        "dagm": test_dataset_dagm,
        "isic": test_dataset_isic,
        "clinic": test_dataset_clinic,
        "colon": test_dataset_colon,
        "brainmri": test_dataset_brainmri,
        "br35h": test_dataset_br35h,
        "kvasir": test_dataset_kvasir,
    }


def train_one_epoch(clip_model, train_dataloader, optimizer, epoch, args, device):
    clip_model.train()

    total_loss = []
    semantic_losses = []
    align_losses = []
    geo_losses = []
    geo_normal_losses = []
    geo_sep_losses = []

    for items in tqdm(train_dataloader, desc=f"Train epoch {epoch}"):
        imgs, labels, gts = items[:3]
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        _, _, img_tokens, aux = clip_model.detect_forward_seg(
            imgs,
            args,
            return_aux=True,
        )

        semantic_cls = aux["semantic_cls"]
        semantic_map = aux["semantic_map"]
        gt_sem = resize_gt_to_map(gts, semantic_map)

        loss_sem = (
            focal_loss(semantic_cls, labels)
            + float(args.lambda1)
            * (
                focal_loss(semantic_map, gt_sem)
                + l1_loss(semantic_map, gt_sem)
            )
        )

        use_align = (
            epoch <= int(args.geo_warmup_epochs)
            or bool(args.patch_align_after_geo)
            or not bool(args.use_geometry)
        )
        if use_align and float(args.lambda2) > 0:
            loss_align = patch_alignment_loss(img_tokens, gt_sem)
        else:
            loss_align = loss_sem.new_tensor(0.0)

        if bool(args.use_geometry) and clip_model.geo_ready:
            loss_geo, geo_stats = geometry_training_loss(
                aux,
                labels,
                gts,
                args,
            )
        else:
            loss_geo = loss_sem.new_tensor(0.0)
            geo_stats = {
                "geo_normal": 0.0,
                "geo_sep": 0.0,
            }

        loss = (
            loss_sem
            + float(args.lambda2) * loss_align
            + float(args.geo_loss_weight) * loss_geo
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        semantic_losses.append(loss_sem.item())
        align_losses.append(loss_align.item())
        geo_losses.append(loss_geo.item())
        geo_normal_losses.append(geo_stats["geo_normal"])
        geo_sep_losses.append(geo_stats["geo_sep"])

    return {
        "loss": float(np.mean(total_loss)),
        "semantic": float(np.mean(semantic_losses)),
        "align": float(np.mean(align_losses)),
        "geometry": float(np.mean(geo_losses)),
        "geo_normal": float(np.mean(geo_normal_losses)),
        "geo_sep": float(np.mean(geo_sep_losses)),
    }


def run_geometry_diagnostics(
    clip_model,
    test_dataset_dict,
    args,
    logger,
    device,
):
    # Mirrors the metadata assumptions already present in util/utils.py.
    pixel_unavailable = {
        "his",
        "chest",
        "oct17",
        "br35h",
        "headct",
    }

    all_results = {}
    for dataset_name, test_ds in test_dataset_dict.items():
        if dataset_name in pixel_unavailable:
            logger.info(
                f"[Geometry Diagnostic] Skip {dataset_name}: no pixel GT."
            )
            continue
        try:
            all_results[dataset_name] = diagnose_geometry_dataset(
                clip_model,
                dataset_name,
                test_ds,
                args,
                device,
                logger=logger,
            )
        except Exception as exc:
            logger.exception(
                f"[Geometry Diagnostic] Failed on {dataset_name}: {exc}"
            )

    output_path = args.geo_diag_json
    if output_path is None:
        output_path = os.path.join(
            args.log_dir,
            f"geometry_diagnostic_{args.dataset}_s{args.seed}.json",
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, allow_nan=True)
    logger.info(f"Saved geometry diagnostics: {output_path}")
    return all_results


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(
        os.path.join(
            args.log_dir,
            f"tomclip_{args.dataset}_{args.fewshot}_s{args.seed}.txt",
        )
    )
    print_args(logger, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    clip_model, clip_transform = load(
        name=args.model,
        jit=(not args.model in available_models()),
        device=device,
        download_root=args.clip_download_dir,
    )

    clip_transform.transforms[0] = transforms.Resize(
        size=(args.img_size, args.img_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )
    clip_transform.transforms[1] = transforms.CenterCrop(
        size=(args.img_size, args.img_size)
    )
    target_transform = transforms.Compose([
        transforms.Resize(
            size=clip_transform.transforms[0].size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ])

    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    clip_model = clip_model.to(device)
    clip_model.insert(
        args=args,
        tokenizer=tokenize,
        device=device,
    )

    all_test_dataset_dict = build_datasets(
        args,
        clip_transform,
        target_transform,
    )

    if len(args.test_dataset) < 1:
        test_dataset_dict = dict(all_test_dataset_dict)
    else:
        test_dataset_dict = {
            ds_name: all_test_dataset_dict[ds_name]
            for ds_name in args.test_dataset
        }

    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]

    # Preserve the source protocol of the uploaded AF-CLIP training script.
    if args.dataset == "mvtec":
        train_dataset = all_test_dataset_dict["mvtec"]
    else:
        train_dataset = all_test_dataset_dict["visa"]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loaded_checkpoint = None
    if args.checkpoint is not None:
        loaded_checkpoint = load_new_checkpoint(
            args.checkpoint,
            clip_model,
            device,
            logger,
        )
    elif args.weight is not None:

        load_legacy_weight_dir(
            clip_model=clip_model,
            weight_dir=args.weight,
            dataset_name=args.dataset,
            device=device,
            logger=logger,
        )

    # -------------------------------------------------------------
    # P0 mode: build geometry on an existing semantic model without training.
    # -------------------------------------------------------------
    if bool(args.diagnostic_only):
        if bool(args.use_geometry) and not clip_model.geo_ready:
            build_normal_geometry(
                clip_model,
                train_dataloader,
                args,
                device,
                logger,
            )
        if not clip_model.geo_ready:
            raise RuntimeError(
                "--diagnostic_only requires --use_geometry 1 and fitted geometry."
            )
        run_geometry_diagnostics(
            clip_model,
            test_dataset_dict,
            args,
            logger,
            device,
        )
        if bool(args.eval_after_diagnostic):
            clip_model.eval()
            for dataset_name, test_ds in test_dataset_dict.items():
                logger.info(
                    "---------------------------{}------------------------------".format(
                        dataset_name
                    )
                )
                eval_all_class(
                    clip_model,
                    dataset_name,
                    test_ds,
                    args,
                    logger,
                    device,
                )
        return

    # -------------------------------------------------------------
    # Evaluation-only mode.
    # -------------------------------------------------------------
    if bool(args.eval_only):
        if bool(args.use_geometry) and not clip_model.geo_ready:
            logger.info(
                "Checkpoint has no geometry; fitting geometry from auxiliary normal data."
            )
            build_normal_geometry(
                clip_model,
                train_dataloader,
                args,
                device,
                logger,
            )
        clip_model.eval()
    else:
        optimizer = torch.optim.Adam(
            clip_model.get_trainable_parameters(),
            lr=args.lr,
            betas=(0.5, 0.999),
        )

        # If training resumes from a geometry-ready checkpoint, geometry-aware
        # refinement starts immediately. Otherwise epoch 1 is semantic warm-up.
        for epoch in range(1, args.epochs + 1):
            stats = train_one_epoch(
                clip_model,
                train_dataloader,
                optimizer,
                epoch,
                args,
                device,
            )
            logger.info(
                "Epoch {}/{} | Loss {:.6f} | Sem {:.6f} | Align {:.6f} | "
                "Geo {:.6f} (normal {:.6f}, sep {:.6f})".format(
                    epoch,
                    args.epochs,
                    stats["loss"],
                    stats["semantic"],
                    stats["align"],
                    stats["geometry"],
                    stats["geo_normal"],
                    stats["geo_sep"],
                )
            )

            # Build the coordinate system after semantic warm-up.
            if (
                bool(args.use_geometry)
                and not clip_model.geo_ready
                and epoch >= int(args.geo_warmup_epochs)
            ):
                build_normal_geometry(
                    clip_model,
                    train_dataloader,
                    args,
                    device,
                    logger,
                )
                if bool(args.freeze_prompt_after_geo):
                    clip_model.freeze_state_prompt()
                    logger.info(
                        "Geometry built; state prompt is frozen for geometric refinement."
                    )

        # The adaptor may move during the geometry-aware stage. Refit the final
        # tangent spaces so inference geometry matches the final representation.
        if bool(args.use_geometry):
            logger.info(
                "Rebuilding final tangent geometry using the final adaptor."
            )
            build_normal_geometry(
                clip_model,
                train_dataloader,
                args,
                device,
                logger,
            )

        clip_model.eval()

        checkpoint_path = args.save_checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"tomclip_{args.dataset}_s{args.seed}.pt",
            )
        save_checkpoint(
            checkpoint_path,
            clip_model,
            args,
            args.epochs,
            logger,
        )

    if bool(args.run_geo_diagnostic) and clip_model.geo_ready:
        run_geometry_diagnostics(
            clip_model,
            test_dataset_dict,
            args,
            logger,
            device,
        )

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info(
            "---------------------------{}------------------------------".format(
                dataset_name
            )
        )
        eval_all_class(
            clip_model,
            dataset_name,
            test_ds,
            args,
            logger,
            device,
        )
        logger.info(
            "-------------------------------------------------------------"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tangent-Off-Manifold CLIP based on the uploaded AF-CLIP code"
    )

    # Original AF-CLIP arguments.
    parser.add_argument(
        "--clip_download_dir",
        type=str,
        default="./download/clip/",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mvtec",
        choices=["mvtec", "visa"],
    )
    parser.add_argument("--model", type=str, default="ViT-L/14@336px")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=12)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--suffix", type=str, default="defect")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument(
        "--feature_layers",
        nargs="+",
        type=int,
        default=[6, 12, 18, 24],
    )
    parser.add_argument(
        "--test_dataset",
        nargs="+",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="legacy AF-CLIP weight directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="new TOM-CLIP checkpoint file",
    )
    parser.add_argument("--vis", type=int, default=0)
    parser.add_argument("--vis_dir", type=str, default="./vis_results/")
    parser.add_argument(
        "--memory_layers",
        nargs="+",
        type=int,
        default=[6, 12, 18, 24],
    )
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    # Geometry architecture.
    parser.add_argument("--use_geometry", type=int, default=1)
    parser.add_argument(
        "--geo_layers",
        nargs="+",
        type=int,
        default=[6, 12, 18, 24],
    )
    parser.add_argument("--geo_scale", type=int, default=1, choices=[1, 3, 5])
    parser.add_argument("--geo_rank", type=int, default=32)
    parser.add_argument(
        "--geo_center_shrink",
        type=float,
        default=1.0,
        help="0=text-anchored linear subspace; 1=full affine normal center",
    )
    parser.add_argument("--geo_pca_oversample", type=int, default=8)
    parser.add_argument("--geo_max_patches", type=int, default=20000)
    parser.add_argument("--geo_patches_per_batch", type=int, default=2048)

    # Inference fusion.
    parser.add_argument("--geo_weight", type=float, default=0.30)
    parser.add_argument("--geo_topk", type=float, default=0.05)
    parser.add_argument(
        "--legacy_logit_sum",
        type=int,
        default=1,
        help="1 preserves the original 12-branch logit summation exactly",
    )

    # Geometry-aware training.
    parser.add_argument("--geo_warmup_epochs", type=int, default=1)
    parser.add_argument("--geo_loss_weight", type=float, default=0.10)
    parser.add_argument("--geo_sep_weight", type=float, default=1.0)
    parser.add_argument("--geo_sep_margin", type=float, default=0.05)
    parser.add_argument("--freeze_prompt_after_geo", type=int, default=1)
    parser.add_argument(
        "--patch_align_after_geo",
        type=int,
        default=0,
        help="0 replaces pairwise alignment with geometry after warm-up",
    )

    # Diagnostics / execution modes.
    parser.add_argument("--diagnostic_only", type=int, default=1)
    parser.add_argument("--eval_after_diagnostic", type=int, default=1)
    parser.add_argument("--run_geo_diagnostic", type=int, default=1)
    parser.add_argument("--geo_diag_max_items", type=int, default=200000)
    parser.add_argument("--geo_diag_json", type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0)

    # Saving.
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--save_checkpoint", type=str, default=None)

    args = parser.parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
