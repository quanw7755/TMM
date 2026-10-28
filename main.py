from clip.clip import (
    load,
    tokenize,
    available_models
)

import torch
from dataset import *
from torchvision import transforms
import argparse
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import copy

from dataset.mpdd import MPDDDataset
from util.utils import eval_all_class

# 2026/9/7   12：39 第二次修改    好
# 2026/9/11   17：02  tta修改   好
# 2026/9/13   12：54 修改  Spatial-DR Predictor   低于base
# 2026/9/13   18：37  修改只增加诊断，不改变最终     低于base
#  2026/9/13   23：22  修改
# 2026/9/15   12：07  fusion修改
# 2026/9/18  17：55   一版最好。
# 2026/9/21  11：37   之前的内容。
# ============================================================
# Reproducibility
# ============================================================

def setup_seed(seed):

    if seed == -1:
        seed = random.randint(
            0,
            1000
        )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


# ============================================================
# Original AF-CLIP losses
# ============================================================

def focal_loss(
    inputs,
    targets,
    alpha=-1,
    gamma=4,
    reduction="mean"
):
    inputs = inputs.float()
    targets = targets.float()

    ce_loss = F.binary_cross_entropy(
        inputs,
        targets,
        reduction="none"
    )

    p_t = (
        inputs * targets
        +
        (1 - inputs)
        * (1 - targets)
    )

    loss = (
        ce_loss
        *
        ((1 - p_t) ** gamma)
    )

    if alpha >= 0:

        alpha_t = (
            alpha * targets
            +
            (1 - alpha)
            * (1 - targets)
        )

        loss = (
            alpha_t
            * loss
        )

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(
    inputs,
    targets,
    reduction="mean"
):
    return F.l1_loss(
        inputs,
        targets,
        reduction=reduction
    )


# ============================================================
# Logger
# ============================================================

def get_logger(
    filename,
    verbosity=1,
    name=None
):
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING
    }

    formatter = logging.Formatter(
        "[%(filename)s]"
        "[line:%(lineno)d]"
        "[%(levelname)s] "
        "%(message)s"
    )

    logger = logging.getLogger(
        name
    )

    logger.setLevel(
        level_dict[verbosity]
    )

    # Avoid duplicated handlers
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(
        filename,
        "w"
    )

    fh.setFormatter(
        formatter
    )

    logger.addHandler(
        fh
    )

    sh = logging.StreamHandler()

    sh.setFormatter(
        formatter
    )

    logger.addHandler(
        sh
    )

    return logger


def print_args(
    logger,
    args
):
    logger.info(
        '--------args----------'
    )

    for k in list(
        vars(args).keys()
    ):
        logger.info(
            '{}: {}'.format(
                k,
                vars(args)[k]
            )
        )

    logger.info(
        '--------args----------\n'
    )


# ============================================================
# Original AF-CLIP patch alignment loss
# ============================================================

def patch_alignment_loss(
    img_tokens,
    labels,
    gts
):
    gts = gts.reshape(
        img_tokens[0].size(0),
        -1
    )

    labels = labels.reshape(
        labels.size(0),
        1
    )

    # --------------------------------------------------------
    # Avoid sharing storage with gts.
    # --------------------------------------------------------
    new_gts = gts.clone()

    if (
        len(
            new_gts[
                new_gts == 0
            ]
        )
        ==
        0
    ):
        return 0

    new_gts[
        new_gts == 0
    ] = -1

    b, l = new_gts.size()

    mask = torch.matmul(
        new_gts.reshape(
            b,
            l,
            1
        ),
        new_gts.reshape(
            b,
            1,
            l
        )
    )

    total_sim = 0

    for img_token in img_tokens:

        img_token = img_token[
            :,
            1:,
            :
        ]

        img_token = F.normalize(
            img_token,
            dim=-1
        )

        sim = torch.matmul(
            img_token,
            img_token.permute(
                0,
                2,
                1
            )
        )

        # Guard against a batch containing only one class.
        different_mask = (
            mask == -1
        )

        same_mask = (
            mask == 1
        )

        if (
            different_mask.sum() == 0
            or
            same_mask.sum() == 0
        ):
            continue

        sim_loss = (
            sim[
                different_mask
            ].mean()
            -
            sim[
                same_mask
            ].mean()
        )

        sim_loss = F.relu(
            sim_loss
        )

        total_sim = (
            total_sim
            +
            sim_loss
        )

    if isinstance(
        total_sim,
        int
    ):
        return img_tokens[0].new_tensor(
            0.0
        )

    return (
        total_sim
        /
        len(img_tokens)
    )


# ============================================================
# NEW:
# Spatial world prediction loss
# ============================================================
# ============================================================
# Signed Residual Fusion losses
# ============================================================

def signed_fusion_dice_loss(
    logits,
    targets,
    eps=1e-6
):
    """
    Soft Dice loss for source anomaly masks.
    """

    probs = torch.sigmoid(
        logits.float()
    )

    targets = (
        targets
        .float()
    )

    intersection = (
        probs
        *
        targets
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        probs.sum(
            dim=(1, 2, 3)
        )
        +
        targets.sum(
            dim=(1, 2, 3)
        )
    )

    dice = (
        2.0
        *
        intersection
        +
        eps
    ) / (
        denominator
        +
        eps
    )

    return (
        1.0
        -
        dice
    ).mean()

def _safe_mean(
    values,
    reference_tensor
):
    """
    Safe mean for possibly empty masks.
    """

    if values.numel() == 0:

        return reference_tensor.new_tensor(
            0.0
        )

    return values.mean()


def signed_error_correction_loss(
    fused_logit,
    correction,
    semantic_score,
    targets,
    fn_margin=0.30,
    fp_margin=0.30,
    fn_weight=0.50,
    fp_weight=0.50,
    keep_weight=1.00,
    magnitude_weight=0.02
):
    """
    Train the Signed Head as an ERROR CORRECTOR instead of
    another anomaly detector.

    Main principle
    --------------
    1. Semantic false negative:
           correction should be positive.

    2. Semantic false positive:
           correction should be negative.

    3. Semantic already correct and confident:
           correction should stay near zero.

    This is specifically designed to protect a strong
    AF-CLIP baseline.
    """

    targets = (
        targets
        .float()
    )

    semantic_score = (
        semantic_score
        .float()
        .detach()
    )

    correction = correction.float()

    # ========================================================
    # 1. Balanced BCE
    #
    # Do NOT use pos_weight=10.
    # ========================================================

    bce_map = F.binary_cross_entropy_with_logits(
        fused_logit.float(),
        targets,
        reduction="none"
    )

    anomaly_mask = (
        targets
        >
        0.5
    )

    normal_mask = (
        ~anomaly_mask
    )

    bce_anomaly = _safe_mean(
        bce_map[
            anomaly_mask
        ],
        fused_logit
    )

    bce_normal = _safe_mean(
        bce_map[
            normal_mask
        ],
        fused_logit
    )

    if (
        anomaly_mask.any()
        and
        normal_mask.any()
    ):

        loss_balanced = (
            0.5
            *
            bce_anomaly
            +
            0.5
            *
            bce_normal
        )

    elif anomaly_mask.any():

        loss_balanced = (
            bce_anomaly
        )

    else:

        loss_balanced = (
            bce_normal
        )

    # ========================================================
    # 2. Semantic false negative rescue
    #
    # y = 1, p_sem < 0.5
    #
    # Require:
    #
    #     delta >= fn_margin
    # ========================================================

    fn_mask = (
        anomaly_mask
        &
        (
            semantic_score
            <
            0.5
        )
    )

    if fn_mask.any():

        fn_error = F.relu(
            fn_margin
            -
            correction[
                fn_mask
            ]
        )

        loss_fn = (
            fn_error
            .pow(2)
            .mean()
        )

    else:

        loss_fn = (
            fused_logit
            .new_tensor(
                0.0
            )
        )

    # ========================================================
    # 3. Semantic false positive suppression
    #
    # y = 0, p_sem > 0.5
    #
    # Require:
    #
    #     delta <= -fp_margin
    # ========================================================

    fp_mask = (
        normal_mask
        &
        (
            semantic_score
            >
            0.5
        )
    )

    if fp_mask.any():

        fp_error = F.relu(
            correction[
                fp_mask
            ]
            +
            fp_margin
        )

        loss_fp = (
            fp_error
            .pow(2)
            .mean()
        )

    else:

        loss_fp = (
            fused_logit
            .new_tensor(
                0.0
            )
        )

    # ========================================================
    # 4. Keep already-correct Semantic predictions unchanged
    #
    # Strong correct anomaly:
    #     y=1 and p_sem>0.7
    #
    # Strong correct normal:
    #     y=0 and p_sem<0.3
    # ========================================================

    keep_mask = (
        (
            anomaly_mask
            &
            (
                semantic_score
                >
                0.7
            )
        )
        |
        (
            normal_mask
            &
            (
                semantic_score
                <
                0.3
            )
        )
    )

    if keep_mask.any():

        loss_keep = (
            correction[
                keep_mask
            ]
            .abs()
            .mean()
        )

    else:

        loss_keep = (
            fused_logit
            .new_tensor(
                0.0
            )
        )

    # ========================================================
    # 5. Global conservative regularization
    # ========================================================

    loss_magnitude = (
        correction
        .abs()
        .mean()
    )

    # ========================================================
    # 6. Final objective
    # ========================================================

    loss = (
        loss_balanced
        +
        fn_weight
        *
        loss_fn
        +
        fp_weight
        *
        loss_fp
        +
        keep_weight
        *
        loss_keep
        +
        magnitude_weight
        *
        loss_magnitude
    )

    return {
        "loss":
            loss,

        "balanced_bce":
            loss_balanced,

        "fn":
            loss_fn,

        "fp":
            loss_fp,

        "keep":
            loss_keep,

        "magnitude":
            loss_magnitude,
    }
def tri_state_focal_loss(
    action_logits,
    action_target,
    gamma=2.0
):
    """
    action_target:
        0 = suppress
        1 = keep
        2 = raise

    Dynamic inverse-sqrt class balancing is used because
    semantic errors are much rarer than KEEP pixels.
    """

    if action_target.ndim != 3:
        raise ValueError(
            "action_target must be [B,H,W], "
            f"got {tuple(action_target.shape)}"
        )

    num_classes = 3

    target_flat = (
        action_target
        .reshape(-1)
        .long()
    )

    counts = torch.bincount(
        target_flat,
        minlength=num_classes
    ).float()

    class_weights = (
        1.0
        /
        torch.sqrt(
            counts
            +
            1.0
        )
    )

    class_weights = (
        class_weights
        /
        class_weights.mean()
    )

    class_weights = class_weights.clamp(
        0.25,
        4.0
    )

    ce = F.cross_entropy(
        action_logits,
        action_target.long(),
        reduction="none"
    )

    probs = torch.softmax(
        action_logits,
        dim=1
    )

    pt = probs.gather(
        1,
        action_target
        .unsqueeze(1)
        .long()
    ).squeeze(1)

    focal = (
        (
            1.0
            -
            pt
        ).pow(
            gamma
        )
        *
        ce
    )

    pixel_weights = class_weights[
        action_target.long()
    ]

    return (
        pixel_weights
        *
        focal
    ).mean()

def hard_pair_ranking_loss(
    final_logit,
    semantic_score,
    targets,
    margin=0.5,
    max_pairs=256
):
    """
    Directly optimize hard semantic ranking errors.

    Hard anomaly:
        anomaly pixels with LOW semantic anomaly score.

    Hard normal:
        normal pixels with HIGH semantic anomaly score.
    """

    targets = (
        targets
        >
        0.5
    )

    semantic_flat = (
        semantic_score
        .reshape(-1)
    )

    final_flat = (
        final_logit
        .reshape(-1)
    )

    target_flat = (
        targets
        .reshape(-1)
    )

    pos_mask = (
        target_flat
    )

    neg_mask = (
        ~target_flat
    )

    if (
        not pos_mask.any()
        or
        not neg_mask.any()
    ):
        return final_logit.new_tensor(
            0.0
        )

    semantic_pos = semantic_flat[
        pos_mask
    ]

    semantic_neg = semantic_flat[
        neg_mask
    ]

    final_pos = final_flat[
        pos_mask
    ]

    final_neg = final_flat[
        neg_mask
    ]

    k_pos = min(
        int(max_pairs),
        int(
            semantic_pos.numel()
        )
    )

    k_neg = min(
        int(max_pairs),
        int(
            semantic_neg.numel()
        )
    )

    # lowest anomaly scores = hardest FN-like positives
    pos_indices = torch.topk(
        semantic_pos,
        k=k_pos,
        largest=False
    ).indices

    # highest normal scores = hardest FP-like negatives
    neg_indices = torch.topk(
        semantic_neg,
        k=k_neg,
        largest=True
    ).indices

    hard_pos = final_pos[
        pos_indices
    ]

    hard_neg = final_neg[
        neg_indices
    ]

    pairwise_margin = (
        margin
        -
        hard_pos[:, None]
        +
        hard_neg[None, :]
    )

    return F.softplus(
        pairwise_margin
    ).mean()

def tri_state_action_margin_loss(
    action_logits,
    fn_mask,
    fp_mask,
    margin=0.50
):
    """
    Force the correct intervention action to win by a margin.

    Tri-State channel definition:

        0 = SUPPRESS
        1 = KEEP
        2 = RAISE

    For semantic false negatives:
        RAISE must exceed both KEEP and SUPPRESS.

    For semantic false positives:
        SUPPRESS must exceed both KEEP and RAISE.

    This is stronger than merely requiring the correct
    class to be the argmax.
    """

    suppress_logit = (
        action_logits[
            :,
            0:1
        ]
    )

    keep_logit = (
        action_logits[
            :,
            1:2
        ]
    )

    raise_logit = (
        action_logits[
            :,
            2:3
        ]
    )

    zero = (
        action_logits
        .new_tensor(
            0.0
        )
    )

    # ========================================================
    # False negative:
    #
    # RAISE > max(KEEP, SUPPRESS) + margin
    # ========================================================

    if fn_mask.any():

        competing_fn = torch.maximum(
            keep_logit,
            suppress_logit
        )

        fn_margin_error = F.relu(
            float(
                margin
            )
            -
            (
                raise_logit
                -
                competing_fn
            )
        )

        loss_fn_margin = (
            fn_margin_error[
                fn_mask
            ]
            .mean()
        )

    else:

        loss_fn_margin = zero

    # ========================================================
    # False positive:
    #
    # SUPPRESS > max(KEEP, RAISE) + margin
    # ========================================================

    if fp_mask.any():

        competing_fp = torch.maximum(
            keep_logit,
            raise_logit
        )

        fp_margin_error = F.relu(
            float(
                margin
            )
            -
            (
                suppress_logit
                -
                competing_fp
            )
        )

        loss_fp_margin = (
            fp_margin_error[
                fp_mask
            ]
            .mean()
        )

    else:

        loss_fp_margin = zero

    # Give FN / FP equal importance when both exist.
    if (
        fn_mask.any()
        and
        fp_mask.any()
    ):

        loss = (
            0.5
            *
            loss_fn_margin
            +
            0.5
            *
            loss_fp_margin
        )

    elif fn_mask.any():

        loss = loss_fn_margin

    elif fp_mask.any():

        loss = loss_fp_margin

    else:

        loss = zero

    return {
        "loss":
            loss,

        "fn_margin":
            loss_fn_margin,

        "fp_margin":
            loss_fp_margin,
    }

def boundary_gradient_loss(
    final_score,
    targets
):
    """
    Lightweight boundary-aware loss.

    It only supervises local gradients near GT boundaries,
    rather than turning the correction head into another
    full segmentation network.
    """

    targets = (
        targets
        .float()
    )

    # ========================================================
    # Boundary band
    # ========================================================

    dilated = F.max_pool2d(
        targets,
        kernel_size=3,
        stride=1,
        padding=1
    )

    eroded = (
        -
        F.max_pool2d(
            -targets,
            kernel_size=3,
            stride=1,
            padding=1
        )
    )

    boundary = (
        dilated
        -
        eroded
    ).clamp(
        0.0,
        1.0
    )

    # ========================================================
    # x-gradient
    # ========================================================

    pred_dx = (
        final_score[
            :,
            :,
            :,
            1:
        ]
        -
        final_score[
            :,
            :,
            :,
            :-1
        ]
    ).abs()

    gt_dx = (
        targets[
            :,
            :,
            :,
            1:
        ]
        -
        targets[
            :,
            :,
            :,
            :-1
        ]
    ).abs()

    boundary_x = torch.maximum(
        boundary[
            :,
            :,
            :,
            1:
        ],
        boundary[
            :,
            :,
            :,
            :-1
        ]
    )

    # ========================================================
    # y-gradient
    # ========================================================

    pred_dy = (
        final_score[
            :,
            :,
            1:,
            :
        ]
        -
        final_score[
            :,
            :,
            :-1,
            :
        ]
    ).abs()

    gt_dy = (
        targets[
            :,
            :,
            1:,
            :
        ]
        -
        targets[
            :,
            :,
            :-1,
            :
        ]
    ).abs()

    boundary_y = torch.maximum(
        boundary[
            :,
            :,
            1:,
            :
        ],
        boundary[
            :,
            :,
            :-1,
            :
        ]
    )

    denom_x = (
        boundary_x.sum()
        +
        1e-6
    )

    denom_y = (
        boundary_y.sum()
        +
        1e-6
    )

    loss_x = (
        (
            pred_dx
            -
            gt_dx
        ).abs()
        *
        boundary_x
    ).sum() / denom_x

    loss_y = (
        (
            pred_dy
            -
            gt_dy
        ).abs()
        *
        boundary_y
    ).sum() / denom_y

    return (
        loss_x
        +
        loss_y
    ) * 0.5

def signed_fusion_bce_loss(
    logits,
    targets,
    max_pos_weight=10.0
):
    """
    Batch-adaptive class-balanced BCE.

    Anomalous pixels are sparse, so ordinary BCE tends to
    learn a nearly all-normal correction.
    """

    targets = (
        targets
        .float()
    )

    positive = targets.sum()

    negative = (
        targets.numel()
        -
        positive
    )

    if positive.item() > 0:

        pos_weight = (
            negative
            /
            positive.clamp_min(
                1.0
            )
        )

        pos_weight = (
            pos_weight
            .clamp(
                min=1.0,
                max=float(
                    max_pos_weight
                )
            )
            .detach()
        )

    else:

        pos_weight = logits.new_tensor(
            1.0
        )

    return F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        pos_weight=
        pos_weight
    )

def world_prediction_loss(
    predicted,
    target,
    l1_weight=0.5
):
    if predicted.shape != target.shape:

        raise ValueError(
            "World prediction/target shape mismatch: "
            f"predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}"
        )

    predicted_n = F.normalize(
        predicted.float(),
        dim=-1,
        eps=1e-6
    )

    target_n = F.normalize(
        target.detach().float(),
        dim=-1,
        eps=1e-6
    )

    cosine_loss = (
        1.0
        -
        (
            predicted_n
            *
            target_n
        ).sum(
            dim=-1
        )
    ).mean()

    smooth_l1 = F.smooth_l1_loss(
        predicted_n,
        target_n,
        reduction="mean"
    )

    return (
        cosine_loss
        +
        l1_weight
        *
        smooth_l1
    )



# ============================================================
# Signed Residual Fusion:
# source-supervised helper / training / checkpoint
# ============================================================

def _extract_world_source_z_map(
    clip_model,
    diagnostic_output
):
    """
    Read the source-standardized World residual map.

    Prefer the map returned by detect_forward_diagnostic().
    For compatibility with older diagnostic implementations,
    reconstruct it from world_geo_score when necessary.
    """

    if (
        "world_source_z_map"
        in diagnostic_output
    ):
        return (
            diagnostic_output[
                "world_source_z_map"
            ]
            .float()
        )

    if (
        "world_geo_score"
        not in diagnostic_output
        or
        "world_geo_map"
        not in diagnostic_output
    ):
        raise RuntimeError(
            "Signed fusion requires either "
            "'world_source_z_map', or both "
            "'world_geo_score' and 'world_geo_map'."
        )

    raw_score = (
        diagnostic_output[
            "world_geo_score"
        ]
        .float()
    )

    source_z = (
        clip_model
        .world_geometry
        .patch_zscore(
            raw_score
        )
    )

    B = source_z.size(0)

    h = int(
        diagnostic_output[
            "world_geo_map"
        ]
        .shape[-2]
    )

    w = int(
        diagnostic_output[
            "world_geo_map"
        ]
        .shape[-1]
    )

    if source_z.size(1) != h * w:
        raise RuntimeError(
            "World source-z shape mismatch: "
            f"N={source_z.size(1)}, "
            f"map={h}x{w}={h*w}."
        )

    return source_z.reshape(
        B,
        1,
        h,
        w
    )


@torch.no_grad()
def build_signed_fusion_inputs(
    clip_model,
    imgs,
    args
):
    """
    Build Identity + HFlip inputs for Signed Residual Fusion.

    Returned tensors
    ----------------
    semantic_score:
        TTA-averaged semantic anomaly probability.

    world_z:
        TTA-averaged source-standardized World residual.

    world_evidence:
        TTA-averaged Soft-Source evidence.

    tta_disagreement:
        weighted semantic view disagreement.

    Notes
    -----
    This helper intentionally reads Semantic / World evidence
    before the final legacy/selective pixel fusion. Thus the
    learned fusion head is trained from the independent experts,
    not from a previously fused prediction.
    """

    identity_weight = float(
        getattr(
            args,
            "tta_identity_weight",
            0.5
        )
    )

    identity_weight = min(
        max(
            identity_weight,
            0.0
        ),
        1.0
    )

    flip_weight = (
        1.0
        -
        identity_weight
    )

    # ========================================================
    # Identity view
    # ========================================================

    output_id = (
        clip_model
        .detect_forward_diagnostic(
            imgs,
            args
        )
    )

    semantic_id = (
        output_id[
            "semantic_map"
        ]
        .float()
    )

    world_z_id = (
        _extract_world_source_z_map(
            clip_model,
            output_id
        )
    )

    evidence_id = (
        output_id[
            "world_evidence_map"
        ]
        .float()
    )

    # ========================================================
    # HFlip view
    # ========================================================

    imgs_flip = torch.flip(
        imgs,
        dims=[3]
    )

    output_flip = (
        clip_model
        .detect_forward_diagnostic(
            imgs_flip,
            args
        )
    )

    semantic_flip = torch.flip(
        output_flip[
            "semantic_map"
        ]
        .float(),
        dims=[3]
    )

    world_z_flip = torch.flip(
        _extract_world_source_z_map(
            clip_model,
            output_flip
        ),
        dims=[3]
    )

    evidence_flip = torch.flip(
        output_flip[
            "world_evidence_map"
        ]
        .float(),
        dims=[3]
    )

    # ========================================================
    # Align all inputs to the semantic map grid
    # ========================================================

    target_size = (
        semantic_id.shape[-2:]
    )

    def _align_map(
        x
    ):
        if (
            x.shape[-2:]
            ==
            target_size
        ):
            return x

        return F.interpolate(
            x.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

    semantic_flip = _align_map(
        semantic_flip
    )

    world_z_id = _align_map(
        world_z_id
    )

    world_z_flip = _align_map(
        world_z_flip
    )

    evidence_id = _align_map(
        evidence_id
    )

    evidence_flip = _align_map(
        evidence_flip
    )

    # ========================================================
    # TTA consensus
    # ========================================================

    semantic_score = (
        identity_weight
        *
        semantic_id
        +
        flip_weight
        *
        semantic_flip
    )

    world_z = (
        identity_weight
        *
        world_z_id
        +
        flip_weight
        *
        world_z_flip
    )

    world_evidence = (
        identity_weight
        *
        evidence_id
        +
        flip_weight
        *
        evidence_flip
    )

    # Weighted mean absolute deviation from TTA consensus.
    tta_disagreement = (
        identity_weight
        *
        torch.abs(
            semantic_id
            -
            semantic_score
        )
        +
        flip_weight
        *
        torch.abs(
            semantic_flip
            -
            semantic_score
        )
    )

    return {
        "semantic_score":
            semantic_score.detach(),

        "world_z":
            world_z.detach(),

        "world_evidence":
            world_evidence.detach(),

        "tta_disagreement":
            tta_disagreement.detach(),
    }


def train_signed_fusion_head(
    clip_model,
    train_dataloader,
    args,
    device,
    logger
):
    """
    Train Tri-State Contextual Error Corrector V3.1.

    Main change from V3:

        Training now uses EXACTLY the same action intervention
        formula as inference:

            action_strength
                ->
            action_gate
                ->
            normalized action_direction
                ->
            bounded correction

    Frozen:
        CLIP backbone
        AF-CLIP prompt/adaptor
        Spatial-DR predictor
        residual geometry

    Trainable:
        signed_fusion_head only
    """

    if (
        clip_model.signed_fusion_head
        is None
    ):

        raise RuntimeError(
            "signed_fusion_head is not initialized."
        )

    if not hasattr(
        clip_model,
        "build_tri_state_intervention"
    ):

        raise RuntimeError(
            "clip/model.py must implement "
            "build_tri_state_intervention()."
        )

    logger.info(
        "========== Train Tri-State Error Corrector V3.1 =========="
    )

    # ========================================================
    # 1. Freeze complete model
    # ========================================================

    for parameter in (
        clip_model.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    # ========================================================
    # 2. Unfreeze Tri-State Head only
    # ========================================================

    for parameter in (
        clip_model
        .signed_fusion_head
        .parameters()
    ):

        parameter.requires_grad_(
            True
        )

    optimizer = torch.optim.AdamW(
        clip_model
        .get_signed_fusion_parameters(),

        lr=float(
            getattr(
                args,
                "signed_fusion_lr",
                1e-3
            )
        ),

        weight_decay=float(
            getattr(
                args,
                "signed_fusion_weight_decay",
                1e-4
            )
        )
    )

    # ========================================================
    # Configuration
    # ========================================================

    epochs = int(
        getattr(
            args,
            "signed_fusion_epochs",
            3
        )
    )

    tri_gamma = float(
        getattr(
            args,
            "tri_focal_gamma",
            2.0
        )
    )

    tri_class_weight = float(
        getattr(
            args,
            "tri_class_weight",
            5.0
        )
    )

    rank_weight = float(
        getattr(
            args,
            "tri_rank_weight",
            0.25
        )
    )

    action_margin_weight = float(
        getattr(
            args,
            "tri_action_margin_weight",
            0.50
        )
    )

    action_margin = float(
        getattr(
            args,
            "tri_action_margin",
            0.50
        )
    )

    keep_weight = float(
        getattr(
            args,
            "tri_keep_weight",
            0.50
        )
    )

    boundary_weight = float(
        getattr(
            args,
            "tri_boundary_weight",
            0.10
        )
    )

    magnitude_weight = float(
        getattr(
            args,
            "tri_magnitude_weight",
            0.01
        )
    )

    rank_margin = float(
        getattr(
            args,
            "tri_rank_margin",
            0.50
        )
    )

    rank_pairs = int(
        getattr(
            args,
            "tri_rank_pairs",
            256
        )
    )

    action_threshold = float(
        getattr(
            args,
            "safe_action_threshold",
            0.55
        )
    )

    action_temperature = float(
        getattr(
            args,
            "safe_action_temperature",
            0.10
        )
    )

    training_correction_cap = float(
        getattr(
            args,
            "safe_max_correction",
            0.30
        )
    )

    logger.info(
        "Tri-State V3.1 configuration: "
        f"epochs={epochs}, "
        f"lr={getattr(args, 'signed_fusion_lr', 1e-3):.6f}, "
        f"TriWeight={tri_class_weight:.3f}, "
        f"RankWeight={rank_weight:.3f}, "
        f"ActionMarginWeight={action_margin_weight:.3f}, "
        f"ActionMargin={action_margin:.3f}, "
        f"ActionThreshold={action_threshold:.3f}, "
        f"ActionTemperature={action_temperature:.3f}, "
        f"TrainCap={training_correction_cap:.3f}"
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        clip_model.eval()

        clip_model.signed_fusion_head.train()

        total_values = []
        tri_values = []
        rank_values = []
        margin_values = []
        keep_values = []
        boundary_values = []
        magnitude_values = []

        delta_fn_values = []
        delta_fp_values = []
        delta_keep_values = []

        fn_action_acc_values = []
        fp_action_acc_values = []

        fn_gate_values = []
        fp_gate_values = []
        keep_gate_values = []

        for items in tqdm(
            train_dataloader,
            desc=
                f"Tri-State V3.1 Epoch {epoch}"
        ):

            imgs, _, gts = (
                items[:3]
            )

            imgs = imgs.to(
                device,
                non_blocking=True
            )

            gts = (
                gts
                .to(
                    device,
                    non_blocking=True
                )
                .float()
            )

            if gts.ndim == 3:

                gts = gts.unsqueeze(
                    1
                )

            # =================================================
            # Frozen Semantic + World inputs
            # =================================================

            with torch.no_grad():

                fusion_inputs = (
                    build_signed_fusion_inputs(
                        clip_model,
                        imgs,
                        args
                    )
                )

            semantic_score = (
                fusion_inputs[
                    "semantic_score"
                ]
                .detach()
                .float()
            )

            world_z = (
                fusion_inputs[
                    "world_z"
                ]
                .detach()
                .float()
            )

            world_evidence = (
                fusion_inputs[
                    "world_evidence"
                ]
                .detach()
                .float()
            )

            disagreement = (
                fusion_inputs[
                    "tta_disagreement"
                ]
                .detach()
                .float()
            )

            # =================================================
            # GT -> patch grid
            # =================================================

            gts_patch = F.interpolate(
                gts,
                size=
                    semantic_score.shape[-2:],
                mode="nearest"
            )

            gts_patch = (
                gts_patch
                >
                0.5
            ).float()

            anomaly_mask = (
                gts_patch
                >
                0.5
            )

            normal_mask = (
                ~anomaly_mask
            )

            # =================================================
            # Semantic error masks
            # =================================================

            fn_mask = (
                anomaly_mask
                &
                (
                    semantic_score
                    <
                    0.5
                )
            )

            fp_mask = (
                normal_mask
                &
                (
                    semantic_score
                    >
                    0.5
                )
            )

            # =================================================
            # Tri-State labels
            #
            # 0 = SUPPRESS
            # 1 = KEEP
            # 2 = RAISE
            # =================================================

            action_target = torch.ones(
                (
                    semantic_score.size(0),
                    semantic_score.size(2),
                    semantic_score.size(3)
                ),
                dtype=torch.long,
                device=device
            )

            action_target[
                fp_mask.squeeze(1)
            ] = 0

            action_target[
                fn_mask.squeeze(1)
            ] = 2

            # =================================================
            # Head forward
            # =================================================

            result = (
                clip_model
                .apply_signed_residual_fusion(
                    semantic_score=
                        semantic_score,

                    world_z=
                        world_z,

                    world_evidence=
                        world_evidence,

                    tta_disagreement=
                        disagreement
                )
            )

            action_logits = (
                result[
                    "action_logits"
                ]
            )

            semantic_logit = (
                result[
                    "semantic_logit"
                ]
            )

            # =================================================
            # IMPORTANT:
            #
            # Use the SAME intervention formula as inference.
            # =================================================

            intervention = (
                clip_model
                .build_tri_state_intervention(
                    tri_result=
                        result,

                    action_threshold=
                        action_threshold,

                    action_temperature=
                        action_temperature,

                    correction_cap=
                        training_correction_cap,

                    consistency_trust=
                        None
                )
            )

            correction = (
                intervention[
                    "safe_delta"
                ]
            )

            action_gate = (
                intervention[
                    "action_gate"
                ]
            )

            final_logit = (
                semantic_logit
                +
                correction
            )

            final_score = torch.sigmoid(
                final_logit
            )

            # =================================================
            # 1. Tri-State focal classification
            # =================================================

            loss_tri = (
                tri_state_focal_loss(
                    action_logits=
                        action_logits,

                    action_target=
                        action_target,

                    gamma=
                        tri_gamma
                )
            )

            # =================================================
            # 2. Explicit action-margin supervision
            # =================================================

            margin_result = (
                tri_state_action_margin_loss(
                    action_logits=
                        action_logits,

                    fn_mask=
                        fn_mask,

                    fp_mask=
                        fp_mask,

                    margin=
                        action_margin
                )
            )

            loss_action_margin = (
                margin_result[
                    "loss"
                ]
            )

            # =================================================
            # 3. Hard pair ranking
            #
            # Ranking now sees the ACTUAL inference-style
            # corrected logit.
            # =================================================

            loss_rank = (
                hard_pair_ranking_loss(
                    final_logit=
                        final_logit,

                    semantic_score=
                        semantic_score,

                    targets=
                        gts_patch,

                    margin=
                        rank_margin,

                    max_pairs=
                        rank_pairs
                )
            )

            # =================================================
            # 4. Protect confident correct Semantic patches
            # =================================================

            confident_keep = (
                (
                    anomaly_mask
                    &
                    (
                        semantic_score
                        >
                        0.70
                    )
                )
                |
                (
                    normal_mask
                    &
                    (
                        semantic_score
                        <
                        0.30
                    )
                )
            )

            if confident_keep.any():

                loss_keep = (
                    correction[
                        confident_keep
                    ]
                    .abs()
                    .mean()
                )

            else:

                loss_keep = (
                    correction
                    .new_tensor(
                        0.0
                    )
                )

            # =================================================
            # 5. Boundary objective
            # =================================================

            loss_boundary = (
                boundary_gradient_loss(
                    final_score=
                        final_score,

                    targets=
                        gts_patch
                )
            )

            # =================================================
            # 6. Conservative magnitude regularizer
            # =================================================

            loss_magnitude = (
                correction
                .abs()
                .mean()
            )

            # =================================================
            # FINAL V3.1 LOSS
            #
            # Previous V3:
            #
            #   Rank dominated the entire objective.
            #
            # V3.1:
            #
            #   action discrimination receives substantially
            #   stronger supervision.
            # =================================================

            loss = (
                tri_class_weight
                *
                loss_tri

                +
                rank_weight
                *
                loss_rank

                +
                action_margin_weight
                *
                loss_action_margin

                +
                keep_weight
                *
                loss_keep

                +
                boundary_weight
                *
                loss_boundary

                +
                magnitude_weight
                *
                loss_magnitude
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                clip_model
                .signed_fusion_head
                .parameters(),
                max_norm=5.0
            )

            optimizer.step()

            # =================================================
            # Diagnostics
            # =================================================

            with torch.no_grad():

                predicted_action = (
                    action_logits
                    .argmax(
                        dim=1,
                        keepdim=True
                    )
                )

                # ---------------------------------------------
                # FN
                # ---------------------------------------------

                if fn_mask.any():

                    delta_fn_values.append(
                        float(
                            correction[
                                fn_mask
                            ]
                            .mean()
                            .item()
                        )
                    )

                    fn_action_acc_values.append(
                        float(
                            (
                                predicted_action[
                                    fn_mask
                                ]
                                ==
                                2
                            )
                            .float()
                            .mean()
                            .item()
                        )
                    )

                    fn_gate_values.append(
                        float(
                            action_gate[
                                fn_mask
                            ]
                            .mean()
                            .item()
                        )
                    )

                # ---------------------------------------------
                # FP
                # ---------------------------------------------

                if fp_mask.any():

                    delta_fp_values.append(
                        float(
                            correction[
                                fp_mask
                            ]
                            .mean()
                            .item()
                        )
                    )

                    fp_action_acc_values.append(
                        float(
                            (
                                predicted_action[
                                    fp_mask
                                ]
                                ==
                                0
                            )
                            .float()
                            .mean()
                            .item()
                        )
                    )

                    fp_gate_values.append(
                        float(
                            action_gate[
                                fp_mask
                            ]
                            .mean()
                            .item()
                        )
                    )

                # ---------------------------------------------
                # KEEP
                # ---------------------------------------------

                if confident_keep.any():

                    delta_keep_values.append(
                        float(
                            correction[
                                confident_keep
                            ]
                            .abs()
                            .mean()
                            .item()
                        )
                    )

                    keep_gate_values.append(
                        float(
                            action_gate[
                                confident_keep
                            ]
                            .mean()
                            .item()
                        )
                    )

            # =================================================
            # Statistics
            # =================================================

            total_values.append(
                float(
                    loss.item()
                )
            )

            tri_values.append(
                float(
                    loss_tri.item()
                )
            )

            rank_values.append(
                float(
                    loss_rank.item()
                )
            )

            margin_values.append(
                float(
                    loss_action_margin.item()
                )
            )

            keep_values.append(
                float(
                    loss_keep.item()
                )
            )

            boundary_values.append(
                float(
                    loss_boundary.item()
                )
            )

            magnitude_values.append(
                float(
                    loss_magnitude.item()
                )
            )

        # ====================================================
        # Epoch summary
        # ====================================================

        def _mean_or_nan(
            values
        ):
            if len(values) == 0:
                return float(
                    "nan"
                )

            return float(
                np.mean(
                    values
                )
            )

        logger.info(
            "Tri-State V3.1 Epoch {}/{} | "
            "Loss={:.6f} | "
            "Tri={:.6f} | "
            "Rank={:.6f} | "
            "Margin={:.6f} | "
            "Keep={:.6f} | "
            "Boundary={:.6f} | "
            "Mag={:.6f} | "
            "Delta_FN={:.6f} | "
            "Delta_FP={:.6f} | "
            "|Delta_KEEP|={:.6f} | "
            "FN_ActionAcc={:.4f} | "
            "FP_ActionAcc={:.4f} | "
            "Gate_FN={:.4f} | "
            "Gate_FP={:.4f} | "
            "Gate_KEEP={:.4f}"
            .format(
                epoch,
                epochs,

                _mean_or_nan(
                    total_values
                ),

                _mean_or_nan(
                    tri_values
                ),

                _mean_or_nan(
                    rank_values
                ),

                _mean_or_nan(
                    margin_values
                ),

                _mean_or_nan(
                    keep_values
                ),

                _mean_or_nan(
                    boundary_values
                ),

                _mean_or_nan(
                    magnitude_values
                ),

                _mean_or_nan(
                    delta_fn_values
                ),

                _mean_or_nan(
                    delta_fp_values
                ),

                _mean_or_nan(
                    delta_keep_values
                ),

                _mean_or_nan(
                    fn_action_acc_values
                ),

                _mean_or_nan(
                    fp_action_acc_values
                ),

                _mean_or_nan(
                    fn_gate_values
                ),

                _mean_or_nan(
                    fp_gate_values
                ),

                _mean_or_nan(
                    keep_gate_values
                ),
            )
        )

    clip_model.signed_fusion_head.eval()

    logger.info(
        "========== Tri-State Error Corrector V3.1 Finished =========="
    )

@torch.no_grad()
def calibrate_safe_fusion_statistics(
    clip_model,
    train_dataloader,
    args,
    device,
    logger
):
    """
    Estimate trust thresholds ONLY from source-normal data.

    No target labels are used.

    Calibrates:
        safe_shift_q50
        safe_shift_q95
        safe_delta_disagree_q95
    """

    logger.info(
        "========== Calibrate Safe Fusion Trust =========="
    )

    clip_model.eval()
    clip_model.signed_fusion_head.eval()

    identity_weight = float(
        getattr(
            args,
            "tta_identity_weight",
            0.5
        )
    )

    identity_weight = min(
        max(
            identity_weight,
            0.0
        ),
        1.0
    )

    flip_weight = (
        1.0
        -
        identity_weight
    )

    shift_values = []
    disagreement_values = []

    for items in tqdm(
        train_dataloader,
        desc="Safe Fusion Calibration"
    ):

        imgs, labels, gts = (
            items[:3]
        )

        imgs = imgs.to(
            device,
            non_blocking=True
        )

        # ====================================================
        # Determine source-normal images
        # ====================================================

        if (
            torch.is_tensor(
                gts
            )
            and
            gts.ndim >= 3
        ):

            gt_for_mask = (
                gts
                .to(
                    device
                )
                .float()
                .reshape(
                    gts.size(0),
                    -1
                )
            )

            normal_image_mask = (
                gt_for_mask
                .max(
                    dim=1
                )
                .values
                <=
                0.5
            )

        else:

            labels_device = (
                labels
                .to(
                    device
                )
                .float()
                .reshape(-1)
            )

            normal_image_mask = (
                labels_device
                <
                0.5
            )

        if not normal_image_mask.any():

            continue

        # ====================================================
        # Identity
        # ====================================================

        out_id = (
            clip_model
            .detect_forward_diagnostic(
                imgs,
                args
            )
        )

        # ====================================================
        # HFlip
        # ====================================================

        imgs_flip = torch.flip(
            imgs,
            dims=[3]
        )

        out_flip = (
            clip_model
            .detect_forward_diagnostic(
                imgs_flip,
                args
            )
        )

        semantic_id = (
            out_id[
                "semantic_map"
            ]
            .float()
        )

        semantic_flip = torch.flip(
            out_flip[
                "semantic_map"
            ]
            .float(),
            dims=[3]
        )

        z_id = (
            _extract_world_source_z_map(
                clip_model,
                out_id
            )
        )

        z_flip = torch.flip(
            _extract_world_source_z_map(
                clip_model,
                out_flip
            ),
            dims=[3]
        )

        evidence_id = (
            out_id[
                "world_evidence_map"
            ]
            .float()
        )

        evidence_flip = torch.flip(
            out_flip[
                "world_evidence_map"
            ]
            .float(),
            dims=[3]
        )

        target_size = (
            semantic_id.shape[-2:]
        )

        def _resize(
            x
        ):
            if (
                x.shape[-2:]
                !=
                target_size
            ):

                x = F.interpolate(
                    x,
                    size=
                    target_size,
                    mode="bilinear",
                    align_corners=False
                )

            return x

        semantic_flip = _resize(
            semantic_flip
        )

        z_id = _resize(
            z_id
        )

        z_flip = _resize(
            z_flip
        )

        evidence_id = _resize(
            evidence_id
        )

        evidence_flip = _resize(
            evidence_flip
        )

        semantic_tta = (
            identity_weight
            *
            semantic_id
            +
            flip_weight
            *
            semantic_flip
        )

        z_tta = (
            identity_weight
            *
            z_id
            +
            flip_weight
            *
            z_flip
        )

        evidence_tta = (
            identity_weight
            *
            evidence_id
            +
            flip_weight
            *
            evidence_flip
        )

        semantic_disagreement = (
            identity_weight
            *
            torch.abs(
                semantic_id
                -
                semantic_tta
            )
            +
            flip_weight
            *
            torch.abs(
                semantic_flip
                -
                semantic_tta
            )
        )

        # ====================================================
        # Aggregate correction
        # ====================================================

        aggregate_result = (
            clip_model
            .apply_signed_residual_fusion(
                semantic_score=
                semantic_tta,

                world_z=
                z_tta,

                world_evidence=
                evidence_tta,

                tta_disagreement=
                semantic_disagreement
            )
        )

        aggregate_intervention = (
            clip_model
            .build_tri_state_intervention(
                tri_result=
                aggregate_result,

                action_threshold=
                float(
                    getattr(
                        args,
                        "safe_action_threshold",
                        0.55
                    )
                ),

                action_temperature=
                float(
                    getattr(
                        args,
                        "safe_action_temperature",
                        0.10
                    )
                ),

                correction_cap=
                float(
                    getattr(
                        args,
                        "safe_max_correction",
                        0.30
                    )
                ),

                consistency_trust=
                None
            )
        )

        delta_tta = (
            aggregate_intervention[
                "safe_delta"
            ]
        )
        # ====================================================
        # Identity-view correction
        # ====================================================

        result_id = (
            clip_model
            .apply_signed_residual_fusion(
                semantic_score=
                semantic_id,

                world_z=
                z_id,

                world_evidence=
                evidence_id,

                tta_disagreement=
                semantic_disagreement
            )
        )

        intervention_id = (
            clip_model
            .build_tri_state_intervention(
                tri_result=
                result_id,

                action_threshold=
                float(
                    getattr(
                        args,
                        "safe_action_threshold",
                        0.55
                    )
                ),

                action_temperature=
                float(
                    getattr(
                        args,
                        "safe_action_temperature",
                        0.10
                    )
                ),

                correction_cap=
                float(
                    getattr(
                        args,
                        "safe_max_correction",
                        0.30
                    )
                ),

                consistency_trust=
                None
            )
        )

        delta_id = (
            intervention_id[
                "safe_delta"
            ]
        )

        # ====================================================
        # Flip-view correction
        # ====================================================

        delta_flip = (
            clip_model
            .apply_signed_residual_fusion(
                semantic_score=
                semantic_flip,

                world_z=
                z_flip,

                world_evidence=
                evidence_flip,

                tta_disagreement=
                semantic_disagreement
            )[
                "correction"
            ]
        )

        delta_disagreement = (
            identity_weight
            *
            torch.abs(
                delta_id
                -
                delta_tta
            )
            +
            flip_weight
            *
            torch.abs(
                delta_flip
                -
                delta_tta
            )
        )

        normal_delta_disagreement = (
            delta_disagreement[
                normal_image_mask
            ]
            .reshape(-1)
            .detach()
            .cpu()
        )

        disagreement_values.append(
            normal_delta_disagreement
        )

        # ====================================================
        # Source residual-distribution shift
        # ====================================================

        if (
            "world_shift_score"
            not in
            out_id
            or
            "world_shift_score"
            not in
            out_flip
        ):

            raise RuntimeError(
                "Safe Fusion requires "
                "'world_shift_score' from "
                "detect_forward_diagnostic()."
            )

        shift_id = (
            out_id[
                "world_shift_score"
            ]
            .float()
            .reshape(-1)
        )

        shift_flip = (
            out_flip[
                "world_shift_score"
            ]
            .float()
            .reshape(-1)
        )

        shift_tta = (
            identity_weight
            *
            shift_id
            +
            flip_weight
            *
            shift_flip
        )

        shift_values.append(
            shift_tta[
                normal_image_mask
            ]
            .detach()
            .cpu()
        )

    if len(
        shift_values
    ) == 0:

        raise RuntimeError(
            "No source-normal shift scores were collected."
        )

    if len(
        disagreement_values
    ) == 0:

        raise RuntimeError(
            "No source-normal correction disagreements "
            "were collected."
        )

    shifts = torch.cat(
        shift_values,
        dim=0
    ).float()

    disagreements = torch.cat(
        disagreement_values,
        dim=0
    ).float()

    shift_q50 = float(
        torch.quantile(
            shifts,
            0.50
        ).item()
    )

    shift_q95 = float(
        torch.quantile(
            shifts,
            0.95
        ).item()
    )

    delta_q95 = float(
        torch.quantile(
            disagreements,
            0.95
        ).item()
    )

    delta_q95 = max(
        delta_q95,
        1e-4
    )

    clip_model.safe_shift_q50 = (
        shift_q50
    )

    clip_model.safe_shift_q95 = (
        shift_q95
    )

    clip_model.safe_delta_disagree_q95 = (
        delta_q95
    )

    logger.info(
        "Safe Fusion source calibration: "
        f"ShiftQ50={shift_q50:.6f}, "
        f"ShiftQ95={shift_q95:.6f}, "
        f"DeltaDisagreeQ95={delta_q95:.6f}"
    )

    logger.info(
        "========== Safe Fusion calibration finished =========="
    )

    return {
        "shift_q50":
            shift_q50,

        "shift_q95":
            shift_q95,

        "delta_disagree_q95":
            delta_q95,
    }
def save_signed_fusion_checkpoint(
    clip_model,
    path,
    args
):
    """
    Save Safe Signed Residual Fusion together with all
    source-normal trust calibration statistics.
    """

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # ========================================================
    # Safe calibration must exist before saving.
    # ========================================================

    required_attributes = [
        "safe_shift_q50",
        "safe_shift_q95",
        "safe_delta_disagree_q95",
    ]

    for name in required_attributes:

        if (
            not hasattr(
                clip_model,
                name
            )
            or
            getattr(
                clip_model,
                name
            )
            is None
        ):

            raise RuntimeError(
                f"Cannot save Safe Fusion checkpoint: "
                f"'{name}' is missing. "
                "Run calibrate_safe_fusion_statistics() first."
            )

    checkpoint = {
        # ====================================================
        # Version
        # ====================================================

        "fusion_type":
            "tri_state_safe_residual_v3",

        # ====================================================
        # Head parameters
        # ====================================================

        "state_dict":
            clip_model
            .signed_fusion_head
            .state_dict(),

        "hidden_dim":
            int(
                getattr(
                    args,
                    "signed_fusion_hidden",
                    16
                )
            ),

        "max_correction":
            float(
                getattr(
                    args,
                    "signed_fusion_max_correction",
                    0.5
                )
            ),

        # ====================================================
        # Source-normal Safe Fusion calibration
        # ====================================================

        "safe_shift_q50":
            float(
                clip_model.safe_shift_q50
            ),

        "safe_shift_q95":
            float(
                clip_model.safe_shift_q95
            ),

        "safe_delta_disagree_q95":
            float(
                clip_model.safe_delta_disagree_q95
            ),

        # ====================================================
        # Metadata
        # ====================================================

        "source_dataset":
            str(
                args.dataset
            ),

        "tta_mode":
            str(
                getattr(
                    args,
                    "tta_mode",
                    "none"
                )
            ),

        "tta_identity_weight":
            float(
                getattr(
                    args,
                    "tta_identity_weight",
                    0.5
                )
            ),
    }

    torch.save(
        checkpoint,
        path
    )

def load_signed_fusion_checkpoint(
    clip_model,
    path,
    device,
    args=None,
    logger=None
):
    """
    Load Safe Signed Residual Fusion checkpoint.

    The Safe-Fusion checkpoint must contain:

        state_dict
        fusion_type
        hidden_dim
        max_correction
        safe_shift_q50
        safe_shift_q95
        safe_delta_disagree_q95

    Old signed_residual_v1 checkpoints are intentionally
    rejected because they do not contain the source-normal
    trust calibration required by Safe Fusion.
    """

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Signed Fusion checkpoint does not exist: {path}"
        )

    # ========================================================
    # 1. Load checkpoint
    # ========================================================

    try:

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location=device
        )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Signed Fusion checkpoint must be a dictionary."
        )

    # ========================================================
    # 2. Check version
    # ========================================================

    fusion_type = checkpoint.get(
        "fusion_type",
        None
    )

    if (
        fusion_type
        !=
          "tri_state_safe_residual_v3"
    ):

        raise RuntimeError(
            "\n"
            "The selected Signed Fusion checkpoint is an OLD "
            "or incompatible checkpoint.\n"
            f"Checkpoint: {path}\n"
            f"fusion_type: {fusion_type}\n\n"
            "Safe Fusion requires:\n"
            "    fusion_type = safe_signed_residual_v2\n\n"
            "Do NOT reuse the old "
            "'mvtec_signed_residual_fusion.pt'.\n"
            "Please train a new Safe Fusion checkpoint with:\n"
            "    --signed_fusion_retrain 1\n"
        )

    # ========================================================
    # 3. Required fields
    # ========================================================

    required_keys = [
        "state_dict",
        "safe_shift_q50",
        "safe_shift_q95",
        "safe_delta_disagree_q95",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if len(
        missing_keys
    ) > 0:

        raise RuntimeError(
            "Safe Signed Fusion checkpoint is missing "
            f"required fields: {missing_keys}"
        )

    # ========================================================
    # 4. Optional architecture check
    # ========================================================

    if args is not None:

        expected_hidden = int(
            getattr(
                args,
                "signed_fusion_hidden",
                16
            )
        )

        ckpt_hidden = int(
            checkpoint.get(
                "hidden_dim",
                expected_hidden
            )
        )

        if (
            ckpt_hidden
            !=
            expected_hidden
        ):

            raise RuntimeError(
                "Signed Fusion hidden dimension mismatch: "
                f"checkpoint={ckpt_hidden}, "
                f"current={expected_hidden}."
            )

        expected_max_correction = float(
            getattr(
                args,
                "signed_fusion_max_correction",
                0.5
            )
        )

        ckpt_max_correction = float(
            checkpoint.get(
                "max_correction",
                expected_max_correction
            )
        )

        if abs(
            ckpt_max_correction
            -
            expected_max_correction
        ) > 1e-8:

            raise RuntimeError(
                "Signed Fusion max_correction mismatch: "
                f"checkpoint={ckpt_max_correction:.6f}, "
                f"current={expected_max_correction:.6f}."
            )

    # ========================================================
    # 5. Load Signed Head
    # ========================================================

    if (
        clip_model.signed_fusion_head
        is None
    ):

        raise RuntimeError(
            "signed_fusion_head is not initialized."
        )

    clip_model.signed_fusion_head.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True
    )

    # ========================================================
    # 6. Restore source-normal Safe Fusion calibration
    # ========================================================

    clip_model.safe_shift_q50 = float(
        checkpoint[
            "safe_shift_q50"
        ]
    )

    clip_model.safe_shift_q95 = float(
        checkpoint[
            "safe_shift_q95"
        ]
    )

    clip_model.safe_delta_disagree_q95 = float(
        checkpoint[
            "safe_delta_disagree_q95"
        ]
    )

    # ========================================================
    # 7. Evaluation mode
    # ========================================================

    clip_model.signed_fusion_head.eval()

    for parameter in (
        clip_model
        .signed_fusion_head
        .parameters()
    ):

        parameter.requires_grad_(
            False
        )

    # ========================================================
    # 8. Log
    # ========================================================

    if logger is not None:

        logger.info(
            "Loaded Safe Signed Residual Fusion: "
            f"ckpt={path}, "
            f"ShiftQ50={clip_model.safe_shift_q50:.6f}, "
            f"ShiftQ95={clip_model.safe_shift_q95:.6f}, "
            f"DeltaDisagreeQ95="
            f"{clip_model.safe_delta_disagree_q95:.6f}"
        )

    return checkpoint

# ============================================================
# NEW:
# Train spatial world predictor
# ============================================================

def train_world_branch(
    clip_model,
    train_dataloader,
    args,
    device,
    logger
):
    logger.info(
        "========== Train Spatial-DR Predictor =========="
    )

    # ========================================================
    # Verify implementation
    # ========================================================

    world_variant = getattr(
        clip_model.world_predictor,
        "variant",
        None
    )

    if world_variant != "spatial_dr_v1":

        raise RuntimeError(
            "train_world_branch expected spatial_dr_v1, "
            f"but got {world_variant}."
        )

    # ========================================================
    # Freeze complete model
    # ========================================================

    for p in clip_model.parameters():

        p.requires_grad_(
            False
        )

    # ========================================================
    # Train Spatial-DR predictor only
    # ========================================================

    for p in (
        clip_model
        .world_predictor
        .parameters()
    ):

        p.requires_grad_(
            True
        )

    # ========================================================
    # Print trainable parameters
    # ========================================================

    trainable_parameter_num = sum(
        p.numel()
        for p in (
            clip_model
            .world_predictor
            .parameters()
        )
        if p.requires_grad
    )

    logger.info(
        "Spatial-DR trainable parameters: "
        f"{trainable_parameter_num}"
    )

    optimizer = torch.optim.AdamW(
        clip_model.get_world_trainable_parameters(),
        lr=args.world_lr,
        weight_decay=1e-4
    )

    # ========================================================
    # Initial gates
    # ========================================================

    with torch.no_grad():

        local_weights, global_gate = (
            clip_model
            .world_predictor
            .get_context_weights()
        )

    logger.info(
        "Spatial-DR before training: "
        "w3={:.6f}, "
        "w5={:.6f}, "
        "global_gate={:.6f}".format(
            local_weights[0].item(),
            local_weights[1].item(),
            global_gate.item()
        )
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        args.world_epochs + 1
    ):

        clip_model.eval()

        clip_model.world_predictor.train()

        total_loss = []

        normal_count = 0

        for items in tqdm(
            train_dataloader,
            desc=f"Spatial-DR Epoch {epoch}"
        ):

            imgs, labels, _ = items[:3]

            labels = labels.reshape(
                -1
            )

            # =================================================
            # Source NORMAL images only
            # =================================================

            normal_idx = (
                labels == 0
            )

            if (
                normal_idx.sum().item()
                ==
                0
            ):
                continue

            imgs = imgs[
                normal_idx
            ].to(
                device,
                non_blocking=True
            )

            normal_count += (
                imgs.size(0)
            )

            # =================================================
            # Frozen CLIP context / target
            # =================================================

            with torch.no_grad():

                stem, target = (
                    clip_model
                    .encode_world_pair(
                        imgs,
                        args
                    )
                )

            # =================================================
            # Spatial-DR expectation
            # =================================================

            predicted = (
                clip_model
                .world_predictor(
                    stem
                )
            )

            if predicted.shape != target.shape:

                raise RuntimeError(
                    "Spatial-DR prediction/target mismatch: "
                    f"predicted={tuple(predicted.shape)}, "
                    f"target={tuple(target.shape)}"
                )

            # =================================================
            # Normal latent prediction loss
            # =================================================

            loss = world_prediction_loss(
                predicted=predicted,
                target=target,
                l1_weight=args.world_l1_weight
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                clip_model
                .world_predictor
                .parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss.append(
                loss.item()
            )

        mean_loss = (
            float(
                np.mean(
                    total_loss
                )
            )
            if len(total_loss) > 0
            else float("nan")
        )

        # ====================================================
        # Spatial-DR diagnostics
        # ====================================================

        with torch.no_grad():

            local_weights, global_gate = (
                clip_model
                .world_predictor
                .get_context_weights()
            )

        logger.info(
            "Spatial-DR Epoch: {}/{}, "
            "Normal images: {}, "
            "Loss: {:.6f}, "
            "w3={:.6f}, "
            "w5={:.6f}, "
            "global_gate={:.6f}".format(
                epoch,
                args.world_epochs,
                normal_count,
                mean_loss,
                local_weights[0].item(),
                local_weights[1].item(),
                global_gate.item()
            )
        )

    # ========================================================
    # Finish
    # ========================================================

    clip_model.world_predictor.eval()

    with torch.no_grad():

        local_weights, global_gate = (
            clip_model
            .world_predictor
            .get_context_weights()
        )

    logger.info(
        "Spatial-DR final context weights: "
        "w3={:.6f}, "
        "w5={:.6f}, "
        "global_gate={:.6f}".format(
            local_weights[0].item(),
            local_weights[1].item(),
            global_gate.item()
        )
    )

    logger.info(
        "========== Spatial-DR Predictor Finished =========="
    )
# ============================================================
# NEW:
# Fit source-normal residual geometry
# ============================================================

@torch.no_grad()
def fit_world_geometry(
    clip_model,
    train_dataloader,
    args,
    device,
    logger
):
    logger.info(
        "========== Spatial-DR Residual Geometry Finished =========="
    )

    clip_model.eval()
    clip_model.world_predictor.eval()

    # ========================================================
    # Stage 1:
    # Image-balanced source-normal residual sampling
    # ========================================================

    residual_bank = []

    total_normal_images = 0

    patches_per_image = getattr(
        args,
        "world_patches_per_image",
        64
    )

    for items in tqdm(
        train_dataloader,
        desc="Balanced world residual sampling"
    ):

        imgs, labels, _ = items[:3]

        labels = labels.reshape(
            -1
        )

        normal_idx = (
            labels == 0
        )

        if (
            normal_idx.sum().item()
            ==
            0
        ):
            continue

        imgs = imgs[
            normal_idx
        ].to(
            device,
            non_blocking=True
        )

        total_normal_images += (
            imgs.size(0)
        )

        # ----------------------------------------------------
        # Frozen CLIP context and prediction target
        # ----------------------------------------------------

        stem, target = (
            clip_model
            .encode_world_pair(
                imgs,
                args
            )
        )

        # ----------------------------------------------------
        # Context -> expected latent state
        # ----------------------------------------------------

        predicted = (
            clip_model
            .world_predictor(
                stem
            )
        )

        # ----------------------------------------------------
        # Predictive residual
        # ----------------------------------------------------

        residual = (
            clip_model
            .world_geometry
            .residual(
                predicted,
                target
            )
        )

        # ----------------------------------------------------
        # Same common-mode correction as inference
        # ----------------------------------------------------

        residual = (
            clip_model
            .world_geometry
            .remove_common_mode(
                residual
            )
        )

        # residual:
        # [B, N, D]

        B, N, D = residual.shape

        sample_num = min(
            patches_per_image,
            N
        )

        # ----------------------------------------------------
        # Every source-normal image contributes equally.
        # ----------------------------------------------------

        for b in range(B):

            idx = torch.randperm(
                N,
                device=residual.device
            )[:sample_num]

            sampled = residual[
                b,
                idx,
                :
            ]

            residual_bank.append(
                sampled.cpu()
            )

    if len(residual_bank) == 0:
        raise RuntimeError(
            "No source-normal residuals were collected."
        )

    residual_bank = torch.cat(
        residual_bank,
        dim=0
    )

    logger.info(
        "Balanced residual bank before global sampling: "
        f"{tuple(residual_bank.shape)}, "
        f"normal_images={total_normal_images}"
    )

    # ========================================================
    # Global cap AFTER all source-normal images are observed
    # ========================================================

    max_patches = getattr(
        args,
        "world_max_patches",
        20000
    )

    if (
        max_patches > 0
        and
        residual_bank.size(0)
        >
        max_patches
    ):

        idx = torch.randperm(
            residual_bank.size(0)
        )[:max_patches]

        residual_bank = (
            residual_bank[
                idx
            ]
        )

    logger.info(
        "Final residual bank: "
        f"{tuple(residual_bank.shape)}"
    )

    residual_bank = residual_bank.to(
        device
    )

    # ========================================================
    # Fit normal residual PCA
    # ========================================================

    clip_model.world_geometry.fit_basis(
        residual_bank
    )

    logger.info(
        "World geometry effective rank: "
        f"{clip_model.world_geometry.effective_rank.item()}"
    )

    del residual_bank

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # Stage 2:
    # Pixel-level and image-level calibration
    # ========================================================

    normal_patch_score_bank = []

    normal_image_score_bank = []

    total_calibration_images = 0

    topk_ratio = getattr(
        args,
        "world_topk",
        0.03
    )

    for items in tqdm(
        train_dataloader,
        desc="World pixel/image calibration"
    ):

        imgs, labels, _ = items[:3]

        labels = labels.reshape(
            -1
        )

        normal_idx = (
            labels == 0
        )

        if (
            normal_idx.sum().item()
            ==
            0
        ):
            continue

        imgs = imgs[
            normal_idx
        ].to(
            device,
            non_blocking=True
        )

        total_calibration_images += (
            imgs.size(0)
        )

        # ----------------------------------------------------
        # Frozen CLIP context and target
        # ----------------------------------------------------

        stem, target = (
            clip_model
            .encode_world_pair(
                imgs,
                args
            )
        )

        # ----------------------------------------------------
        # Expected state
        # ----------------------------------------------------

        predicted = (
            clip_model
            .world_predictor(
                stem
            )
        )

        # ----------------------------------------------------
        # Raw pixel-level residual score
        # ----------------------------------------------------

        raw_score = (
            clip_model
            .world_geometry
            .raw_score(
                predicted,
                target
            )
        )

        # [B, N]

        normal_patch_score_bank.append(
            raw_score
            .reshape(-1)
            .cpu()
        )

        # ====================================================
        # Image-level score:
        # Top-K aggregation BEFORE calibration
        # ====================================================

        k = max(
            1,
            int(
                raw_score.size(1)
                *
                topk_ratio
            )
        )

        image_score = (
            raw_score
            .topk(
                k,
                dim=1
            )
            .values
            .mean(
                dim=1
            )
        )

        normal_image_score_bank.append(
            image_score.cpu()
        )

    if len(
        normal_patch_score_bank
    ) == 0:
        raise RuntimeError(
            "No source-normal patch scores were collected."
        )

    if len(
        normal_image_score_bank
    ) == 0:
        raise RuntimeError(
            "No source-normal image scores were collected."
        )

    normal_patch_score_bank = (
        torch.cat(
            normal_patch_score_bank,
            dim=0
        )
        .to(device)
    )

    normal_image_score_bank = (
        torch.cat(
            normal_image_score_bank,
            dim=0
        )
        .to(device)
    )

    # ========================================================
    # Source-only calibration
    # ========================================================

    patch_quantile = getattr(
        args,
        "world_patch_quantile",
        0.99
    )

    image_quantile = getattr(
        args,
        "world_image_quantile",
        0.99
    )

    patch_robust_k = getattr(
        args,
        "world_patch_robust_k",
        2.0
    )

    # --------------------------------------------------------
    # NEW World-v2.1:
    #
    # patch threshold =
    #
    # max(
    #     Q99,
    #     median + k * 1.4826 * MAD
    # )
    # --------------------------------------------------------

    clip_model.world_geometry.calibrate_patch(
        normal_patch_score_bank,
        quantile=patch_quantile,
        robust_k=patch_robust_k
    )

    # --------------------------------------------------------
    # Image calibration:
    # keep World-v2 behavior
    # --------------------------------------------------------

    clip_model.world_geometry.calibrate_image(
        normal_image_score_bank,
        quantile=image_quantile
    )

    # ========================================================
    # Logs
    # ========================================================

    patch_median = (
        clip_model
        .world_geometry
        .score_median
        .item()
    )

    patch_mad = (
        clip_model
        .world_geometry
        .score_mad
        .item()
    )

    patch_q = (
        clip_model
        .world_geometry
        .patch_quantile_threshold
        .item()
    )

    patch_robust = (
        clip_model
        .world_geometry
        .patch_robust_threshold
        .item()
    )

    patch_final = (
        clip_model
        .world_geometry
        .patch_threshold
        .item()
    )

    image_median = (
        clip_model
        .world_geometry
        .image_score_median
        .item()
    )

    image_mad = (
        clip_model
        .world_geometry
        .image_score_mad
        .item()
    )

    image_threshold = (
        clip_model
        .world_geometry
        .image_threshold
        .item()
    )

    logger.info(
        "World-v2.1 calibration:"
    )

    logger.info(
        "Patch median={:.6f}, "
        "MAD={:.6f}, "
        "Q={:.6f}, "
        "RobustT={:.6f}, "
        "FinalT={:.6f}".format(
            patch_median,
            patch_mad,
            patch_q,
            patch_robust,
            patch_final
        )
    )

    logger.info(
        "Image median={:.6f}, "
        "MAD={:.6f}, "
        "Q={:.6f}".format(
            image_median,
            image_mad,
            image_threshold
        )
    )

    logger.info(
        "Calibration normal images: "
        f"{total_calibration_images}"
    )

    # Additional distribution logs
    logger.info(
        "Patch score: "
        "mean={:.6f}, std={:.6f}, "
        "min={:.6f}, max={:.6f}".format(
            normal_patch_score_bank.mean().item(),
            normal_patch_score_bank.std().item(),
            normal_patch_score_bank.min().item(),
            normal_patch_score_bank.max().item()
        )
    )

    logger.info(
        "Image score: "
        "mean={:.6f}, std={:.6f}, "
        "min={:.6f}, max={:.6f}".format(
            normal_image_score_bank.mean().item(),
            normal_image_score_bank.std().item(),
            normal_image_score_bank.min().item(),
            normal_image_score_bank.max().item()
        )
    )

    logger.info(
        "========== World-v2.1 Geometry Finished =========="
    )
@torch.no_grad()
def fit_world_shift_calibration(
    clip_model,
    train_dataloader,
    args,
    device,
    logger
):
    """
    Calibrate sample-wise residual shift using SOURCE-NORMAL
    images only.

    This function:

        - does NOT train Spatial-DR;
        - does NOT refit PCA;
        - does NOT use target data;
        - only estimates the normal range of residual
          distribution shift.
    """

    logger.info(
        "========== Fit Source-Normal Residual Shift =========="
    )

    if (
        clip_model.world_predictor
        is None
    ):

        raise RuntimeError(
            "World predictor is not initialized."
        )

    if (
        clip_model.world_geometry
        is None
        or
        not clip_model.world_geometry.ready
    ):

        raise RuntimeError(
            "World geometry must be calibrated before "
            "shift calibration."
        )

    clip_model.eval()

    clip_model.world_predictor.eval()

    shift_score_bank = []

    normal_image_count = 0

    relative_scale_floor = getattr(
        args,
        "world_relative_scale_floor",
        0.25
    )

    for items in tqdm(
        train_dataloader,
        desc="World shift calibration"
    ):

        imgs, labels, _ = (
            items[:3]
        )

        labels = (
            labels
            .reshape(-1)
        )

        normal_idx = (
            labels
            ==
            0
        )

        if (
            normal_idx.sum().item()
            ==
            0
        ):
            continue

        imgs = (
            imgs[
                normal_idx
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        normal_image_count += (
            imgs.size(0)
        )

        # ====================================================
        # Context / target
        # ====================================================

        stem, target = (
            clip_model
            .encode_world_pair(
                imgs,
                args
            )
        )

        # ====================================================
        # Expected latent
        # ====================================================

        predicted = (
            clip_model
            .world_predictor(
                stem
            )
        )

        # ====================================================
        # Residual geometry
        # ====================================================

        raw_score = (
            clip_model
            .world_geometry
            .raw_score(
                predicted,
                target
            )
        )

        # ====================================================
        # Sample-wise residual shift
        # ====================================================

        shift_score = (
            clip_model
            .world_geometry
            .image_shift_score(
                raw_score,
                scale_floor=
                relative_scale_floor
            )
        )

        shift_score_bank.append(
            shift_score
            .detach()
            .cpu()
        )

    if len(
        shift_score_bank
    ) == 0:

        raise RuntimeError(
            "No source-normal samples were collected "
            "for residual-shift calibration."
        )

    shift_score_bank = torch.cat(
        shift_score_bank,
        dim=0
    ).reshape(
        -1
    )

    shift_quantile = getattr(
        args,
        "world_shift_quantile",
        0.95
    )

    # ========================================================
    # Fit source-normal shift statistics
    # ========================================================

    clip_model.world_geometry.calibrate_shift(
        shift_score_bank.to(
            device
        ),
        quantile=
        shift_quantile
    )

    logger.info(
        "Source-normal residual shift: "
        "images={}, "
        "median={:.6f}, "
        "MAD={:.6f}, "
        "Q{:.3f}={:.6f}".format(
            normal_image_count,

            clip_model
            .world_geometry
            .shift_score_median
            .item(),

            clip_model
            .world_geometry
            .shift_score_mad
            .item(),

            shift_quantile,

            clip_model
            .world_geometry
            .shift_threshold
            .item()
        )
    )

    logger.info(
        "========== Residual Shift Calibration Finished =========="
    )
# ============================================================
# NEW:
# World checkpoint
# ============================================================

def save_world_checkpoint(
    clip_model,
    args,
    path
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    local_weights, global_gate = (
        clip_model
        .world_predictor
        .get_context_weights()
    )

    checkpoint = {

        "world_type":
            "spatial_dr_v1",

        "predictor":
            clip_model
            .world_predictor
            .state_dict(),

        "geometry":
            clip_model
            .world_geometry
            .state_dict(),

        "world_layer":
            args.world_layer,

        "world_hidden":
            args.world_hidden,

        "world_rank":
            args.world_rank,

        "local_weights":
            local_weights
            .detach()
            .cpu(),

        "global_gate":
            float(
                global_gate.item()
            )
    }
    torch.save(
        checkpoint,
        path
    )


def load_world_checkpoint(
    clip_model,
    path,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device
    )

    world_type = checkpoint.get(
        "world_type",
        None
    )

    if world_type != "spatial_dr_v1":

        raise RuntimeError(
            "Wrong world checkpoint. "
            f"Expected spatial_dr_v1, "
            f"but checkpoint type is {world_type}. "
            f"Path: {path}"
        )

    predictor_variant = getattr(
        clip_model.world_predictor,
        "variant",
        None
    )

    if predictor_variant != "spatial_dr_v1":

        raise RuntimeError(
            "Current model does not contain "
            "Spatial-DR v1 predictor."
        )

    clip_model.world_predictor.load_state_dict(
        checkpoint[
            "predictor"
        ],
        strict=True
    )

    geometry_load_result = (
        clip_model
        .world_geometry
        .load_state_dict(
            checkpoint[
                "geometry"
            ],
            strict=False
        )
    )

    # ========================================================
    # Only the newly introduced residual-shift buffers are
    # allowed to be absent in a legacy Spatial-DR checkpoint.
    # ========================================================

    allowed_missing_keys = {
        "shift_score_median",
        "shift_score_mad",
        "shift_threshold",
        "shift_ready_flag",
    }

    actual_missing_keys = set(
        geometry_load_result
        .missing_keys
    )

    unexpected_keys = set(
        geometry_load_result
        .unexpected_keys
    )

    invalid_missing_keys = (
            actual_missing_keys
            -
            allowed_missing_keys
    )

    if len(
            invalid_missing_keys
    ) > 0:
        raise RuntimeError(
            "Unexpected missing World geometry keys: "
            f"{sorted(invalid_missing_keys)}"
        )

    if len(
            unexpected_keys
    ) > 0:
        raise RuntimeError(
            "Unexpected World geometry checkpoint keys: "
            f"{sorted(unexpected_keys)}"
        )

    if len(
            actual_missing_keys
    ) > 0:
        # Legacy checkpoint:
        # shift calibration must be fitted separately.

        clip_model.world_geometry.shift_ready_flag.fill_(
            False
        )

    clip_model.world_predictor.eval()
# ============================================================
# Main train
# ============================================================

def train(args):

    os.makedirs(
        args.log_dir,
        exist_ok=True
    )

    logger = get_logger(
        os.path.join(
            args.log_dir,
            '{}_{}_s{}.txt'.format(
                args.dataset,
                args.fewshot,
                args.seed
            )
        )
    )

    print_args(
        logger,
        args
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Load CLIP
    # ========================================================

    clip_model, clip_transform = load(
        name=args.model,
        jit=(
            not args.model
            in available_models()
        ),
        device=device,
        download_root=args.clip_download_dir
    )

    # ========================================================
    # Input size
    # ========================================================

    clip_transform.transforms[0] = transforms.Resize(
        size=(
            args.img_size,
            args.img_size
        ),
        interpolation=
        transforms.InterpolationMode.BICUBIC
    )

    clip_transform.transforms[1] = transforms.CenterCrop(
        size=(
            args.img_size,
            args.img_size
        )
    )

    target_transform = transforms.Compose([
        transforms.Resize(
            size=
            clip_transform
            .transforms[0]
            .size,
            interpolation=
            transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor()
    ])

    # ========================================================
    # Freeze original CLIP
    # ========================================================

    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad_(
            False
        )

    clip_model = clip_model.to(
        device
    )

    # ========================================================
    # Insert AF-CLIP + optional world modules
    # ========================================================

    clip_model.insert(
        args=args,
        tokenizer=tokenize,
        device=device
    )
    # ========================================================
    # Verify Spatial-DR implementation
    # ========================================================

    if args.use_world:

        world_variant = getattr(
            clip_model.world_predictor,
            "variant",
            None
        )

        logger.info(
            "World predictor variant: "
            f"{world_variant}"
        )

        if world_variant != "spatial_dr_v1":
            raise RuntimeError(
                "Spatial-DR is not active. "
                f"Current predictor variant: {world_variant}"
            )

        local_weights, global_gate = (
            clip_model
            .world_predictor
            .get_context_weights()
        )

        logger.info(
            "Spatial-DR initial context weights: "
            "w3={:.6f}, "
            "w5={:.6f}, "
            "global_gate={:.6f}".format(
                local_weights[0].item(),
                local_weights[1].item(),
                global_gate.item()
            )
        )
    # ========================================================
    # Datasets
    # ========================================================

    test_dataset_mvtec = MVTecDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_isic = ISICDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_clinic = ClinicDBDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_colon = ColonDBDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_visa = VisaDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_btad = BTADDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_mpdd = MPDDDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_brainmri = BrainMRIDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_br35h = Br35HDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_dagm = DAGMDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    test_dataset_kvasir = KvasirDataset(
        root=args.data_dir,
        train=False,
        category=None,
        transform=clip_transform,
        gt_target_transform=target_transform
    )

    all_test_dataset_dict = {
        "mvtec":
            test_dataset_mvtec,

        "visa":
            test_dataset_visa,

        "btad":
            test_dataset_btad,

        "mpdd":
            test_dataset_mpdd,

        "dagm":
            test_dataset_dagm,

        "isic":
            test_dataset_isic,

        "clinic":
            test_dataset_clinic,

        "colon":
            test_dataset_colon,

        "brainmri":
            test_dataset_brainmri,

        "br35h":
            test_dataset_br35h,

        "kvasir":
            test_dataset_kvasir,
    }

    # ========================================================
    # Target datasets
    # ========================================================

    if len(
        args.test_dataset
    ) < 1:

        test_dataset_dict = (
            all_test_dataset_dict.copy()
        )

    else:

        test_dataset_dict = {}

        for ds_name in (
            args.test_dataset
        ):

            test_dataset_dict[
                ds_name
            ] = (
                all_test_dataset_dict[
                    ds_name
                ]
            )

    if (
        args.dataset
        in test_dataset_dict
    ):
        del test_dataset_dict[
            args.dataset
        ]

    # ========================================================
    # Auxiliary source
    # ========================================================

    if args.dataset == 'mvtec':

        train_dataset = (
            test_dataset_mvtec
        )

    else:

        train_dataset = (
            test_dataset_visa
        )

    train_dataloader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    )

    # ========================================================
    # Original AF-CLIP
    # ========================================================

    if args.weight is not None:

        logger.info(
            "Load existing AF-CLIP weights."
        )

        clip_model.state_prompt_embedding = (
            torch.load(
                os.path.join(
                    args.weight,
                    "{}_prompt.pt".format(
                        args.dataset
                    )
                ),
                map_location=device
            )
        )

        clip_model.adaptor = torch.load(
            os.path.join(
                args.weight,
                "{}_adaptor.pt".format(
                    args.dataset
                )
            ),
            map_location=device
        )

    else:

        logger.info(
            "Train original AF-CLIP semantic branch."
        )

        optimizer = torch.optim.Adam(
            clip_model.get_trainable_parameters(),
            lr=args.lr,
            betas=(
                0.5,
                0.999
            )
        )

        for epoch in range(
            1,
            args.epochs + 1
        ):

            total_loss = []

            for items in tqdm(
                train_dataloader
            ):

                imgs, labels, gts = (
                    items[:3]
                )

                labels = labels.to(
                    device
                )

                imgs = imgs.to(
                    device
                )

                gts = gts.to(
                    device
                )

                (
                    predict_labels,
                    predict_masks,
                    img_tokens
                ) = (
                    clip_model
                    .detect_forward_seg(
                        imgs,
                        args=args
                    )
                )

                gts = F.interpolate(
                    gts,
                    size=
                    predict_masks[
                        0
                    ].shape[-2:],
                    mode='bilinear'
                )

                gts[
                    gts < 0.5
                ] = 0

                gts[
                    gts > 0.5
                ] = 1

                loss = (
                    focal_loss(
                        predict_labels,
                        labels
                    )
                    +
                    args.lambda1
                    *
                    (
                        focal_loss(
                            predict_masks,
                            gts
                        )
                        +
                        l1_loss(
                            predict_masks,
                            gts
                        )
                    )
                    +
                    args.lambda2
                    *
                    patch_alignment_loss(
                        img_tokens,
                        labels,
                        gts
                    )
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss.append(
                    loss.item()
                )

            logger.info(
                "Epoch: {}/{}, "
                "Loss: {:.6f}".format(
                    epoch,
                    args.epochs,
                    np.mean(
                        total_loss
                    )
                )
            )

    # ========================================================
    # NEW:
    # Spatial World Expectation
    # ========================================================

    if args.use_world:

        if (
            args.world_ckpt
            is not None
        ):

            world_ckpt = (
                args.world_ckpt
            )

        else:

            world_ckpt = os.path.join(
                args.world_save_dir,
                "{}_spatial_dr_v1_L{}_R{}.pt".format(
                    args.dataset,
                    args.world_layer,
                    args.world_rank
                )
            )

        should_load = (
            os.path.exists(
                world_ckpt
            )
            and
            not args.world_retrain
        )

        if should_load:

            logger.info(
                "Load world checkpoint: "
                f"{world_ckpt}"
            )

            load_world_checkpoint(
                clip_model,
                world_ckpt,
                device
            )
            # ========================================================
            # Adaptive residual-shift calibration
            # ========================================================

            evidence_mode = getattr(
                args,
                "world_pixel_evidence_mode",
                "hard_max"
            )

            if (
                    evidence_mode
                    ==
                    "soft_adaptive"
            ):

                if not (
                        clip_model
                                .world_geometry
                                .shift_ready
                ):
                    logger.info(
                        "Residual-shift statistics are absent "
                        "from the current World checkpoint. "
                        "Fit them from source-normal images."
                    )

                    fit_world_shift_calibration(
                        clip_model=
                        clip_model,

                        train_dataloader=
                        train_dataloader,

                        args=
                        args,

                        device=
                        device,

                        logger=
                        logger
                    )
                save_world_checkpoint(
                    clip_model,
                    args,
                    world_ckpt
                )

                logger.info(
                    "Updated World checkpoint with "
                    "source-normal residual-shift calibration: "
                    f"{world_ckpt}"
                )
                logger.info(
                    "Adaptive residual shift is ready: "
                    "median={:.6f}, "
                    "MAD={:.6f}, "
                    "threshold={:.6f}".format(
                        clip_model
                        .world_geometry
                        .shift_score_median
                        .item(),

                        clip_model
                        .world_geometry
                        .shift_score_mad
                        .item(),

                        clip_model
                        .world_geometry
                        .shift_threshold
                        .item()
                    )
                )
            logger.info(
                "World pixel evidence configuration: "
                "mode={}, "
                "relative_weight={:.3f}, "
                "relative_tau={:.3f}, "
                "soft_temperature={:.3f}, "
                "reliability_tau={:.3f}, "
                "reliability_temperature={:.3f}"
                .format(
                    args.world_pixel_evidence_mode,
                    args.world_relative_weight,
                    args.world_relative_tau,
                    args.world_soft_temperature,
                    args.world_reliability_tau,
                    args.world_reliability_temperature
                )
            )

            logger.info(
                "Loaded source World calibration: "
                "median={:.6f}, "
                "MAD={:.6f}, "
                "Q={:.6f}, "
                "RobustT={:.6f}, "
                "LegacyT={:.6f}"
                .format(
                    clip_model
                    .world_geometry
                    .score_median
                    .item(),

                    clip_model
                    .world_geometry
                    .score_mad
                    .item(),

                    clip_model
                    .world_geometry
                    .patch_quantile_threshold
                    .item(),

                    clip_model
                    .world_geometry
                    .patch_robust_threshold
                    .item(),

                    clip_model
                    .world_geometry
                    .patch_threshold
                    .item()
                )
            )
        else:

            logger.info(
                "Train new spatial world branch."
            )

            train_world_branch(
                clip_model,
                train_dataloader,
                args,
                device,
                logger
            )

            fit_world_geometry(
                clip_model,
                train_dataloader,
                args,
                device,
                logger
            )

            save_world_checkpoint(
                clip_model,
                args,
                world_ckpt
            )

            logger.info(
                "Saved world checkpoint: "
                f"{world_ckpt}"
            )

    # ========================================================
    # Safe Signed Residual Fusion
    #
    # Spatial-DR + Soft-Source + HFlip consistency are frozen.
    # Only the tiny signed correction head is trained/loaded.
    #
    # Safe Fusion additionally requires source-normal trust
    # calibration:
    #
    #   safe_shift_q50
    #   safe_shift_q95
    #   safe_delta_disagree_q95
    #
    # No target labels are used.
    # ========================================================

    if bool(
            getattr(
                args,
                "use_signed_fusion",
                0
            )
    ):

        # ====================================================
        # 1. Basic requirements
        # ====================================================

        if not bool(
                getattr(
                    args,
                    "use_world",
                    0
                )
        ):
            raise RuntimeError(
                "Signed Residual Fusion requires --use_world 1."
            )

        # ----------------------------------------------------
        # Model-side implementation checks
        # ----------------------------------------------------

        if (
                not hasattr(
                    clip_model,
                    "signed_fusion_head"
                )
                or
                clip_model.signed_fusion_head
                is None
        ):
            raise RuntimeError(
                "use_signed_fusion=1, but clip/model.py did not "
                "initialize signed_fusion_head. "
                "Please apply the SignedResidualFusionHead "
                "modification first."
            )

        if (
                not hasattr(
                    clip_model,
                    "get_signed_fusion_parameters"
                )
                or
                not hasattr(
                    clip_model,
                    "apply_signed_residual_fusion"
                )
        ):
            raise RuntimeError(
                "clip/model.py is missing Signed Fusion helper "
                "methods. Expected get_signed_fusion_parameters() "
                "and apply_signed_residual_fusion()."
            )

        # ====================================================
        # 2. Recommended inference configuration
        # ====================================================

        if (
                str(
                    args.world_pixel_evidence_mode
                )
                !=
                "soft_source"
        ):
            logger.warning(
                "Safe Signed Fusion was designed for "
                "--world_pixel_evidence_mode soft_source, "
                "but current mode is '{}'."
                .format(
                    args.world_pixel_evidence_mode
                )
            )

        if (
                str(
                    args.tta_mode
                )
                !=
                "hflip"
        ):
            logger.warning(
                "Safe Signed Fusion source calibration uses "
                "Identity + HFlip consistency. "
                "Recommended setting is --tta_mode hflip, "
                "but current mode is '{}'."
                .format(
                    args.tta_mode
                )
            )

        # ====================================================
        # 3. Safe Fusion checkpoint path
        # ====================================================

        if (
                args.signed_fusion_ckpt
                is not None
        ):

            signed_fusion_ckpt = (
                args.signed_fusion_ckpt
            )

        else:

            signed_fusion_ckpt = os.path.join(
                args.world_save_dir,
                "{}_tri_state_safe_v3.pt"
                .format(
                    args.dataset
                )
            )

        # ====================================================
        # 4. Decide whether to load or retrain
        # ====================================================

        should_load_signed = (
                os.path.exists(
                    signed_fusion_ckpt
                )
                and
                not bool(
                    getattr(
                        args,
                        "signed_fusion_retrain",
                        0
                    )
                )
        )

        # ====================================================
        # 5. Load existing Safe Fusion checkpoint
        # ====================================================

        if should_load_signed:

            logger.info(
                "Load Safe Signed Residual Fusion checkpoint: "
                f"{signed_fusion_ckpt}"
            )

            load_signed_fusion_checkpoint(
                clip_model=
                clip_model,

                path=
                signed_fusion_ckpt,

                device=
                device,

                args=
                args,

                logger=
                logger
            )

        # ====================================================
        # 6. Train a new Safe Signed Fusion checkpoint
        # ====================================================

        else:

            logger.info(
                "Train new Safe Signed Residual Fusion Head."
            )

            # ------------------------------------------------
            # 6.1 Train error-targeted signed correction
            #
            # AF-CLIP:
            #     frozen
            #
            # Spatial-DR:
            #     frozen
            #
            # Geometry:
            #     frozen
            #
            # Signed Head:
            #     trainable
            # ------------------------------------------------

            train_signed_fusion_head(
                clip_model=
                clip_model,

                train_dataloader=
                train_dataloader,

                args=
                args,

                device=
                device,

                logger=
                logger
            )

            # ------------------------------------------------
            # 6.2 Source-normal trust calibration
            #
            # CRITICAL:
            #
            # This step MUST happen after Signed Head training
            # and before checkpoint saving.
            #
            # It estimates:
            #
            #     safe_shift_q50
            #     safe_shift_q95
            #     safe_delta_disagree_q95
            #
            # using SOURCE-NORMAL data only.
            # ------------------------------------------------

            calibration_result = (
                calibrate_safe_fusion_statistics(
                    clip_model=
                    clip_model,

                    train_dataloader=
                    train_dataloader,

                    args=
                    args,

                    device=
                    device,

                    logger=
                    logger
                )
            )

            logger.info(
                "Safe Fusion source calibration completed: "
                "ShiftQ50={:.6f}, "
                "ShiftQ95={:.6f}, "
                "DeltaDisagreeQ95={:.6f}"
                .format(
                    calibration_result[
                        "shift_q50"
                    ],

                    calibration_result[
                        "shift_q95"
                    ],

                    calibration_result[
                        "delta_disagree_q95"
                    ],
                )
            )

            # ------------------------------------------------
            # 6.3 Save the Signed Head together with all
            # source-normal trust statistics.
            # ------------------------------------------------

            save_signed_fusion_checkpoint(
                clip_model=
                clip_model,

                path=
                signed_fusion_ckpt,

                args=
                args
            )

            logger.info(
                "Saved Safe Signed Residual Fusion checkpoint: "
                f"{signed_fusion_ckpt}"
            )

        # ====================================================
        # 7. Final Safe Fusion calibration sanity check
        # ====================================================

        if bool(
                getattr(
                    args,
                    "use_safe_fusion",
                    0
                )
        ):

            required_safe_attributes = [
                "safe_shift_q50",
                "safe_shift_q95",
                "safe_delta_disagree_q95",
            ]

            missing_safe_attributes = []

            for name in required_safe_attributes:

                if (
                        not hasattr(
                            clip_model,
                            name
                        )
                        or
                        getattr(
                            clip_model,
                            name
                        )
                        is None
                ):
                    missing_safe_attributes.append(
                        name
                    )

            if len(
                    missing_safe_attributes
            ) > 0:
                raise RuntimeError(
                    "Safe Fusion is enabled, but source-normal "
                    "calibration statistics are missing: "
                    f"{missing_safe_attributes}. "
                    "Please retrain the Safe Signed Fusion "
                    "checkpoint with "
                    "--signed_fusion_retrain 1."
                )

        # ====================================================
        # 8. Freeze Signed Head for target evaluation
        # ====================================================

        clip_model.signed_fusion_head.eval()

        for parameter in (
                clip_model
                        .signed_fusion_head
                        .parameters()
        ):
            parameter.requires_grad_(
                False
            )

        # ====================================================
        # 9. Final status
        # ====================================================

        logger.info(
            "Safe Signed Residual Fusion is ready: "
            "ckpt={}, "
            "hidden={}, "
            "max_correction={:.3f}, "
            "safe_fusion={}, "
            "tta_mode={}, "
            "identity_weight={:.3f}"
            .format(
                signed_fusion_ckpt,

                args.signed_fusion_hidden,

                args.signed_fusion_max_correction,

                bool(
                    getattr(
                        args,
                        "use_safe_fusion",
                        0
                    )
                ),

                args.tta_mode,

                args.tta_identity_weight
            )
        )

        if bool(
                getattr(
                    args,
                    "use_safe_fusion",
                    0
                )
        ):
            logger.info(
                "Safe Fusion trust statistics: "
                "ShiftQ50={:.6f}, "
                "ShiftQ95={:.6f}, "
                "DeltaDisagreeQ95={:.6f}"
                .format(
                    float(
                        clip_model.safe_shift_q50
                    ),

                    float(
                        clip_model.safe_shift_q95
                    ),

                    float(
                        clip_model.safe_delta_disagree_q95
                    )
                )
            )

    # ========================================================
    # Evaluation
    # ========================================================

    clip_model.eval()

    for (
        dataset_name,
        test_ds
    ) in test_dataset_dict.items():

        logger.info(
            "---------------------------"
            "{}"
            "------------------------------"
            .format(
                dataset_name
            )
        )

        eval_all_class(
            clip_model,
            dataset_name,
            test_ds,
            args,
            logger,
            device
        )

        logger.info(
            "-------------------------------------------------------------"
        )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        'AF-CLIP + Spatial World Expectation'
    )

    # ========================================================
    # Original AF-CLIP arguments
    # ========================================================

    parser.add_argument(
        '--clip_download_dir',
        type=str,
        default='./download/clip/'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='mvtec',
        choices=[
            'mvtec',
            'visa'
        ]
    )

    parser.add_argument(
        '--model',
        type=str,
        default="ViT-L/14@336px"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=2
    )

    parser.add_argument(
        '--prompt_len',
        type=int,
        default=12
    )

    parser.add_argument(
        '--category',
        type=str,
        default=None
    )

    parser.add_argument(
        '--fewshot',
        type=int,
        default=0
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=122
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='./log/'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='defect'
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=518
    )

    parser.add_argument(
        '--feature_layers',
        nargs='+',
        type=int,
        default=[
            6,
            12,
            18,
            24
        ]
    )

    parser.add_argument(
        '--test_dataset',
        nargs='+',
        type=str,
        default=[]
    )

    parser.add_argument(
        '--weight',
        type=str,
        default=None
    )

    parser.add_argument(
        '--vis',
        type=int,
        default=0
    )

    parser.add_argument(
        '--vis_dir',
        type=str,
        default='./vis_results/'
    )

    parser.add_argument(
        '--memory_layers',
        nargs='+',
        type=int,
        default=[
            6,
            12,
            18,
            24
        ]
    )

    parser.add_argument(
        '--lambda1',
        type=float,
        default=1
    )

    parser.add_argument(
        '--lambda2',
        type=float,
        default=1.0,
        help=(
            'Patch-alignment loss weight. '
            'Use 1.0 to match the main AF-CLIP baseline setting.'
        )
    )

    # ========================================================
    # NEW:
    # Spatial World Expectation
    # ========================================================

    parser.add_argument(
        '--use_world',
        type=int,
        default=1,
        help=
        '1: enable spatial world branch'
    )

    parser.add_argument(
        '--world_layer',
        type=int,
        default=18,
        choices=[
            6,
            12,
            18,
            24
        ],
        help=
        'CLIP latent layer used as '
        'world prediction target'
    )

    parser.add_argument(
        '--world_hidden',
        type=int,
        default=512
    )

    parser.add_argument(
        '--world_epochs',
        type=int,
        default=5
    )

    parser.add_argument(
        '--world_lr',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--world_l1_weight',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--world_rank',
        type=int,
        default=32
    )

    parser.add_argument(
        '--world_max_patches',
        type=int,
        default=20000
    )

    parser.add_argument(
        '--world_weight',
        type=float,
        default=0.3
    )

    parser.add_argument(
        '--world_threshold',
        type=float,
        default=2.0
    )

    parser.add_argument(
        '--world_topk',
        type=float,
        default=0.03
    )

    parser.add_argument(
        '--world_save_dir',
        type=str,
        default='./world_weights/'
    )

    parser.add_argument(
        '--world_ckpt',
        type=str,
        default=None
    )

    parser.add_argument(
        '--world_retrain',
        type=int,
        default=0,
        choices=[
            0,
            1
        ],
        help=(
            '1: retrain Spatial-DR even if the checkpoint exists; '
            '0: load the existing checkpoint when available.'
        )
    )
    parser.add_argument(
        '--world_patches_per_image',
        type=int,
        default=64,
        help='balanced residual patches sampled from each source-normal image'
    )

    parser.add_argument(
        '--world_patch_quantile',
        type=float,
        default=0.99,
        help='source-normal quantile used to activate pixel-level world evidence'
    )

    parser.add_argument(
        '--world_image_quantile',
        type=float,
        default=0.99,
        help='source-normal quantile used to activate image-level world evidence'
    )

    parser.add_argument(
        '--world_uncertainty_floor',
        type=float,
        default=0.10,
        help='minimum world gate when AF-CLIP semantic prediction is very confident'
    )

    parser.add_argument(
        '--world_pixel_weight',
        type=float,
        default=0.30
    )
    # ============================================================
    # Semantic-World fusion
    # ============================================================

    parser.add_argument(
        '--world_fusion_mode',
        type=str,
        default='legacy',
        choices=[
            'legacy',
            'selective'
        ],
        help=(
            'Pixel-level Semantic-World fusion strategy. '
            'legacy reproduces the previous uncertainty-floor '
            'fusion. selective removes unconditional World '
            'injection and lets World evidence contribute mainly '
            'when the semantic expert is uncertain.'
        )
    )

    parser.add_argument(
        '--world_semantic_gate_power',
        type=float,
        default=1.0,
        help=(
            'Exponent gamma applied to semantic uncertainty '
            '4*p*(1-p) in selective fusion.'
        )
    )
    parser.add_argument(
        '--world_image_weight',
        type=float,
        default=0.20
    )
    # ========================================================
    # Test-Time Augmentation
    # ========================================================

    parser.add_argument(
        '--tta_mode',
        type=str,
        default='hflip',
        choices=[
            'none',
            'hflip',
            'flip4',
            'rot4'
        ],
        help=(
            'Test-time augmentation mode. '
            'Recommended first setting: hflip.'
        )
    )

    parser.add_argument(
        '--tta_identity_weight',
        type=float,
        default=0.5,
        help=(
            'Weight assigned to the original image prediction '
            'during TTA fusion.'
        )
    )
    # ========================================================
    # Spatial-DR diagnostics
    # ========================================================

    parser.add_argument(
        '--world_diag',
        type=int,
        default=1,
        choices=[
            0,
            1
        ],
        help=(
            '1: evaluate standalone Spatial-DR prediction, '
            'residual geometry, evidence activation, and fusion.'
        )
    )

    parser.add_argument(
        '--world_diag_sigma',
        type=float,
        default=4.0,
        help=(
            'Gaussian smoothing sigma used for standalone '
            'diagnostic pixel maps.'
        )
    )
    # ========================================================
    # Pixel-level World Evidence Calibration
    # ========================================================

    parser.add_argument(
        '--world_pixel_evidence_mode',
        type=str,
        default='soft_source',
        choices=[
            'hard_max',
            'hard_quantile',
            'hard_robust',
            'soft_source',
            'soft_relative',
            'soft_adaptive'
        ],
        help=(
            'Pixel World evidence calibration. '
            'hard_max reproduces legacy behavior; '
            'hard_quantile uses source Q only; '
            'soft_source keeps continuous source evidence; '
            'soft_relative enables cross-domain relative '
            'continuous calibration.'
        )
    )
    parser.add_argument(
        '--world_shift_quantile',
        type=float,
        default=0.95,
        help=(
            'Source-normal upper quantile used to calibrate '
            'the nominal residual-shift boundary.'
        )
    )

    parser.add_argument(
        '--world_shift_temperature',
        type=float,
        default=1.0,
        help=(
            'Temperature multiplier of the adaptive '
            'source-relative weighting function.'
        )
    )
    parser.add_argument(
        '--world_relative_weight',
        type=float,
        default=0.5,
        help=(
            'Weight of image-relative robust residual '
            'coordinate in soft_relative mode.'
        )
    )

    parser.add_argument(
        '--world_relative_tau',
        type=float,
        default=1.0,
        help=(
            'Reference robust-z level for the image-relative '
            'coordinate.'
        )
    )

    parser.add_argument(
        '--world_soft_temperature',
        type=float,
        default=0.5,
        help=(
            'Softplus temperature used for continuous '
            'World evidence.'
        )
    )

    parser.add_argument(
        '--world_relative_scale_floor',
        type=float,
        default=0.25,
        help=(
            'Minimum image-relative robust scale as a fraction '
            'of the source robust scale.'
        )
    )

    parser.add_argument(
        '--world_reliability_tau',
        type=float,
        default=0.5,
        help=(
            'Robust-z reference used by the cross-coordinate '
            'reliability gate.'
        )
    )

    parser.add_argument(
        '--world_reliability_temperature',
        type=float,
        default=0.5,
        help=(
            'Temperature of the World reliability gate.'
        )
    )

    parser.add_argument(
        '--world_reliability_floor',
        type=float,
        default=0.0,
        help=(
            'Minimum World reliability fusion gate.'
        )
    )

    # ========================================================
    # Signed Residual Fusion
    # ========================================================

    parser.add_argument(
        '--use_signed_fusion',
        type=int,
        default=1,
        choices=[
            0,
            1
        ],
        help=(
            '1: enable source-supervised signed residual '
            'pixel fusion.'
        )
    )

    parser.add_argument(
        '--signed_fusion_hidden',
        type=int,
        default=16,
        help=(
            'Hidden width of the 5->hidden->8->1 '
            'Signed Residual Fusion Head.'
        )
    )

    parser.add_argument(
        '--signed_fusion_max_correction',
        type=float,
        default=0.3,
        help=(
            'Maximum absolute signed logit correction.'
        )
    )
    # ============================================================
    # Safe Signed Fusion training
    # ============================================================
    # ============================================================
    # Tri-State Error Corrector
    # ============================================================

    parser.add_argument(
        '--tri_focal_gamma',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '--tri_class_weight',
        type=float,
        default=5.0
    )

    parser.add_argument(
        '--tri_action_margin_weight',
        type=float,
        default=0.50
    )

    parser.add_argument(
        '--tri_action_margin',
        type=float,
        default=0.50
    )
    parser.add_argument(
        '--tri_rank_weight',
        type=float,
        default=0.25
    )

    parser.add_argument(
        '--tri_keep_weight',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--tri_boundary_weight',
        type=float,
        default=0.10
    )

    parser.add_argument(
        '--tri_magnitude_weight',
        type=float,
        default=0.01
    )

    parser.add_argument(
        '--tri_rank_margin',
        type=float,
        default=0.50
    )

    parser.add_argument(
        '--tri_rank_pairs',
        type=int,
        default=256
    )

    # ============================================================
    # Safe intervention
    # ============================================================

    parser.add_argument(
        '--safe_action_threshold',
        type=float,
        default=0.55
    )

    parser.add_argument(
        '--safe_action_temperature',
        type=float,
        default=0.10
    )

    parser.add_argument(
        '--safe_min_correction',
        type=float,
        default=0.05
    )

    parser.add_argument(
        '--safe_max_correction',
        type=float,
        default=0.30
    )

    parser.add_argument(
        '--safe_consistency_floor',
        type=float,
        default=0.25
    )
    parser.add_argument(
        '--signed_fn_margin',
        type=float,
        default=0.30
    )

    parser.add_argument(
        '--signed_fp_margin',
        type=float,
        default=0.30
    )

    parser.add_argument(
        '--signed_fn_weight',
        type=float,
        default=0.50
    )

    parser.add_argument(
        '--signed_fp_weight',
        type=float,
        default=0.50
    )

    parser.add_argument(
        '--signed_keep_weight',
        type=float,
        default=1.00
    )

    parser.add_argument(
        '--signed_magnitude_weight',
        type=float,
        default=0.02
    )

    # ============================================================
    # Safe Fusion inference
    # ============================================================

    parser.add_argument(
        '--use_safe_fusion',
        type=int,
        default=1,
        choices=[
            0,
            1
        ]
    )

    parser.add_argument(
        '--safe_semantic_power',
        type=float,
        default=2.0
    )

    parser.add_argument(
        '--safe_world_support_temperature',
        type=float,
        default=0.25
    )

    parser.add_argument(
        '--safe_shift_scale_multiplier',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--safe_consistency_scale_multiplier',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--safe_delta_limit',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--signed_fusion_epochs',
        type=int,
        default=3
    )

    parser.add_argument(
        '--signed_fusion_lr',
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--signed_fusion_weight_decay',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--signed_fusion_dice_weight',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--signed_fusion_corr_weight',
        type=float,
        default=0.05,
        help=(
            'L1 regularization weight on the signed logit '
            'correction. This keeps AF-CLIP as the primary '
            'expert and makes correction conservative.'
        )
    )

    parser.add_argument(
        '--signed_fusion_max_pos_weight',
        type=float,
        default=10.0,
        help=(
            'Upper bound of batch-adaptive positive BCE weight.'
        )
    )

    parser.add_argument(
        '--signed_fusion_ckpt',
        type=str,
        default=None,
        help=(
            'Optional Signed Fusion checkpoint path. '
            'If omitted, use '
            '<world_save_dir>/<dataset>_signed_residual_fusion.pt.'
        )
    )

    parser.add_argument(
        '--signed_fusion_retrain',
        type=int,
        default=0,
        choices=[
            0,
            1
        ],
        help=(
            '1: retrain Signed Fusion even if checkpoint exists; '
            '0: load checkpoint when available.'
        )
    )


    args = parser.parse_args()

    args.seed = setup_seed(
        args.seed
    )

    train(
        args
    )