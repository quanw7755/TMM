import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from clip.geometry import fit_tangent_subspace


def _log(logger, message: str):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def _sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.size(0) <= n:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:n]
    return x[idx]


@torch.no_grad()
def build_normal_geometry(
    clip_model,
    dataloader,
    args,
    device,
    logger=None,
):
    """
    Estimate one tangent subspace per selected CLIP layer from auxiliary
    normal images only.

    Important protocol choice:
      * Only image-level normal samples (label == 0) enter the fit.
      * Anomalous images are not allowed to contribute "normal background"
        patches, avoiding contextual contamination.
      * The semantic normal text feature is frozen as the spherical anchor.

    Returns the same geometry state that is loaded into the model.
    """
    was_training = clip_model.training
    clip_model.eval()

    geo_layers = [int(x) for x in args.geo_layers]
    max_patches = int(args.geo_max_patches)
    patches_per_batch = int(args.geo_patches_per_batch)

    collected = {layer: [] for layer in geo_layers}
    counts = {layer: 0 for layer in geo_layers}

    text_features = F.normalize(clip_model.encode_state_prompt(), dim=-1)
    normal_anchor = text_features[0].detach().float()

    _log(
        logger,
        f"[Geometry] Collecting auxiliary normal patches: layers={geo_layers}, "
        f"rank={args.geo_rank}, max_patches/layer={max_patches}, "
        f"scale={args.geo_scale}",
    )

    for items in tqdm(dataloader, desc="Collect normal geometry"):
        imgs, labels = items[:2]
        labels = labels.reshape(-1)
        normal_mask = labels == 0
        if normal_mask.sum().item() == 0:
            continue

        normal_imgs = imgs[normal_mask].to(device, non_blocking=True)
        layer_features = clip_model.extract_geometry_features(normal_imgs, args)

        for layer in geo_layers:
            if counts[layer] >= max_patches:
                continue

            z = layer_features[layer].reshape(-1, layer_features[layer].size(-1))
            remaining = max_patches - counts[layer]
            take = min(patches_per_batch, remaining, z.size(0))
            if take <= 0:
                continue

            z = _sample_rows(z, take).detach().float().cpu()
            collected[layer].append(z)
            counts[layer] += z.size(0)

        if all(counts[layer] >= max_patches for layer in geo_layers):
            break

    for layer in geo_layers:
        if counts[layer] < max(int(args.geo_rank) + 2, 64):
            raise RuntimeError(
                f"Insufficient normal patches for layer {layer}: {counts[layer]}. "
                "Increase auxiliary normal data or lower --geo_rank."
            )

    layer_states = {}
    fit_stats = {}

    for layer in geo_layers:
        normal_features = torch.cat(collected[layer], dim=0)
        # Fit sequentially on GPU to keep CPU/GPU memory predictable.
        normal_features = normal_features.to(device, non_blocking=True)
        center, basis, stats = fit_tangent_subspace(
            normal_features,
            normal_anchor.to(device),
            rank=int(args.geo_rank),
            center_shrink=float(args.geo_center_shrink),
            pca_oversample=int(args.geo_pca_oversample),
        )
        layer_states[layer] = {
            "center": center.detach(),
            "basis": basis.detach(),
        }
        fit_stats[layer] = stats
        _log(
            logger,
            "[Geometry][L{}] n={} rank={} explained_energy={:.4f} "
            "mean_off_ratio={:.4f} mean_geodesic={:.4f}".format(
                layer,
                int(stats["num_samples"]),
                int(stats["rank"]),
                stats["explained_energy"],
                stats["mean_off_ratio"],
                stats["mean_geodesic"],
            ),
        )

        del normal_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clip_model.set_geometry(
        normal_anchor.to(device),
        layer_states,
        fit_stats=fit_stats,
    )

    if was_training:
        clip_model.train()
    else:
        clip_model.eval()

    return clip_model.get_geometry_state()


def _summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    x = x.float().reshape(-1)
    if x.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
        }
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device))
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "q25": float(q[0].item()),
        "median": float(q[1].item()),
        "q75": float(q[2].item()),
    }


def _bounded_append(storage, values: torch.Tensor, max_items: int):
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return
    existing = sum(x.numel() for x in storage)
    remaining = max_items - existing
    if remaining <= 0:
        return
    if values.numel() > remaining:
        idx = torch.randperm(values.numel())[:remaining]
        values = values[idx]
    storage.append(values)


@torch.no_grad()
def diagnose_geometry_dataset(
    clip_model,
    dataset_name,
    test_dataset,
    args,
    device,
    logger=None,
    max_items_per_group: Optional[int] = None,
):
    """
    Test the core empirical hypothesis on datasets that provide pixel masks:

        E[R_perp | anomaly patch] > E[R_perp | normal patch].

    The function loops through all categories using the dataset's existing
    update(category) interface. It reports both direction-only R_perp and the
    actual off-manifold distance score used for prediction.
    """
    if not clip_model.geo_ready:
        raise RuntimeError("Geometry diagnostics require fitted geometry.")

    max_items = int(max_items_per_group or args.geo_diag_max_items)

    normal_ratio = []
    anomaly_ratio = []
    normal_score = []
    anomaly_score = []
    normal_off_energy = []
    anomaly_off_energy = []

    was_training = clip_model.training
    clip_model.eval()

    for category in test_dataset.categories:
        test_dataset.update(category)
        dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        for items in tqdm(dataloader, desc=f"GeoDiag {dataset_name}/{category}", leave=False):
            imgs, _, gt = items[:3]
            imgs = imgs.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True).float()

            # Force the geometry branch on for diagnostics without permanently
            # changing the caller's argparse object.
            old_use_geometry = getattr(args, "use_geometry", 0)
            args.use_geometry = 1
            _, _, _, aux = clip_model.detect_forward_seg(
                imgs,
                args,
                return_aux=True,
            )
            args.use_geometry = old_use_geometry

            ratio = aux["off_ratio"]
            score = aux["geo_patch"]
            off_energy = aux["off_energy"]

            b, l = ratio.shape
            side = int(math.sqrt(l))
            if side * side != l:
                raise ValueError(f"Patch count {l} is not square.")

            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt_patch = F.interpolate(
                gt,
                size=(side, side),
                mode="nearest",
            ).reshape(b, -1)

            normal_mask = gt_patch < 0.5
            anomaly_mask = gt_patch >= 0.5

            _bounded_append(normal_ratio, ratio[normal_mask], max_items)
            _bounded_append(anomaly_ratio, ratio[anomaly_mask], max_items)
            _bounded_append(normal_score, score[normal_mask], max_items)
            _bounded_append(anomaly_score, score[anomaly_mask], max_items)
            _bounded_append(normal_off_energy, off_energy[normal_mask], max_items)
            _bounded_append(anomaly_off_energy, off_energy[anomaly_mask], max_items)

    def cat_or_empty(items):
        return torch.cat(items) if items else torch.empty(0)

    nr = cat_or_empty(normal_ratio)
    ar = cat_or_empty(anomaly_ratio)
    ns = cat_or_empty(normal_score)
    a_s = cat_or_empty(anomaly_score)
    ne = cat_or_empty(normal_off_energy)
    ae = cat_or_empty(anomaly_off_energy)

    result = {
        "normal_off_ratio": _summarize_tensor(nr),
        "anomaly_off_ratio": _summarize_tensor(ar),
        "normal_geo_score": _summarize_tensor(ns),
        "anomaly_geo_score": _summarize_tensor(a_s),
        "normal_off_energy": _summarize_tensor(ne),
        "anomaly_off_energy": _summarize_tensor(ae),
    }

    if nr.numel() > 0 and ar.numel() > 0:
        y_ratio = np.concatenate([
            np.zeros(nr.numel(), dtype=np.int64),
            np.ones(ar.numel(), dtype=np.int64),
        ])
        s_ratio = np.concatenate([nr.numpy(), ar.numpy()])
        result["off_ratio_patch_AUROC"] = float(roc_auc_score(y_ratio, s_ratio))

        y_score = np.concatenate([
            np.zeros(ns.numel(), dtype=np.int64),
            np.ones(a_s.numel(), dtype=np.int64),
        ])
        s_score = np.concatenate([ns.numpy(), a_s.numpy()])
        result["geo_score_patch_AUROC"] = float(roc_auc_score(y_score, s_score))

        result["ratio_mean_gap"] = (
            result["anomaly_off_ratio"]["mean"]
            - result["normal_off_ratio"]["mean"]
        )
        result["score_mean_gap"] = (
            result["anomaly_geo_score"]["mean"]
            - result["normal_geo_score"]["mean"]
        )

    _log(logger, f"[Geometry Diagnostic][{dataset_name}] {result}")

    if was_training:
        clip_model.train()
    return result
