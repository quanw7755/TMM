"""
  @Author: 王权
  @FileName: liver.py
  @DateTime: 2026/6/22 16:39
  @SoftWare: PyCharm
"""
"""Liver dataset"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class LiverDataset(Dataset):
    def __init__(
        self,
        root,
        train=False,
        category=None,
        fewshot=0,
        seed=122,
        transform=None,
        gt_target_transform=None
    ):
        super().__init__()

        self.train = train
        self.fewshot = fewshot
        self.seed = seed

        self.transform = transform
        self.gt_target_transform = gt_target_transform

        # 如果传入的是 /data
        # 则最终目录为 /data/Med/Liver
        self.root = os.path.join(root, "Med", "Liver")

        self.categories = ["liver"]
        self.category = "liver"
        self.dataset_name = "liver"

        self.preprocess()
        self.update(category)

    @staticmethod
    def _find_images(folder):
        """
        同时支持 png、jpg、jpeg、bmp、tif、tiff。
        """
        extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.tif",
            "*.tiff"
        ]

        paths = []

        for extension in extensions:
            paths.extend(
                glob.glob(
                    os.path.join(folder, extension)
                )
            )

        return sorted(paths)

    @staticmethod
    def _normalize_stem(path):
        """
        用于匹配图像和 mask 文件名。

        例如：
            001.png
            001_mask.png
            001_anomaly_mask.png

        都可归一化到相近的名称。
        """
        stem = os.path.splitext(
            os.path.basename(path)
        )[0]

        suffixes = [
            "_mask",
            "_anomaly_mask",
            "-mask",
            "-anomaly_mask"
        ]

        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        return stem

    def _match_masks(self, img_paths, mask_folder, is_normal):
        """
        为每张图像查找对应 mask。

        正常图像找不到 mask 时允许返回 None，
        __getitem__ 中会生成全零 mask。

        异常图像找不到 mask 时直接报错。
        """
        mask_paths = self._find_images(mask_folder)

        mask_dict = {
            self._normalize_stem(path): path
            for path in mask_paths
        }

        matched_masks = []

        for img_path in img_paths:
            image_stem = self._normalize_stem(img_path)
            mask_path = mask_dict.get(image_stem, None)

            if mask_path is None and not is_normal:
                raise FileNotFoundError(
                    "Cannot find anomaly mask for abnormal image:\n"
                    f"image: {img_path}\n"
                    f"mask folder: {mask_folder}"
                )

            matched_masks.append(mask_path)

        return matched_masks

    def preprocess(self):
        good_img_dir = os.path.join(
            self.root,
            "test",
            "good",
            "img"
        )

        good_mask_dir = os.path.join(
            self.root,
            "test",
            "good",
            "anomaly_mask"
        )

        bad_img_dir = os.path.join(
            self.root,
            "test",
            "Ungood",
            "img"
        )

        bad_mask_dir = os.path.join(
            self.root,
            "test",
            "Ungood",
            "anomaly_mask"
        )

        for directory in [
            good_img_dir,
            bad_img_dir,
            bad_mask_dir
        ]:
            if not os.path.isdir(directory):
                raise FileNotFoundError(
                    f"Required Liver directory not found: {directory}"
                )

        self.good_img_paths = self._find_images(
            good_img_dir
        )

        self.bad_img_paths = self._find_images(
            bad_img_dir
        )

        if len(self.good_img_paths) == 0:
            raise RuntimeError(
                f"No normal images found in: {good_img_dir}"
            )

        if len(self.bad_img_paths) == 0:
            raise RuntimeError(
                f"No abnormal images found in: {bad_img_dir}"
            )

        # 正常 mask 目录允许不存在或者没有对应文件
        if os.path.isdir(good_mask_dir):
            self.good_gt_paths = self._match_masks(
                self.good_img_paths,
                good_mask_dir,
                is_normal=True
            )
        else:
            self.good_gt_paths = [
                None
            ] * len(self.good_img_paths)

        self.bad_gt_paths = self._match_masks(
            self.bad_img_paths,
            bad_mask_dir,
            is_normal=False
        )

        self.good_labels = [
            0
        ] * len(self.good_img_paths)

        self.bad_labels = [
            1
        ] * len(self.bad_img_paths)

    def _get_support_indices(self):
        """
        从正常图像中选择 K 张支持图像。
        """
        normal_num = len(self.good_img_paths)

        if self.fewshot <= 0:
            return np.array([], dtype=np.int64)

        if self.fewshot >= normal_num:
            raise ValueError(
                f"fewshot={self.fewshot}, but Liver only has "
                f"{normal_num} normal images. At least one normal "
                f"image should remain in the query set."
            )

        rng = np.random.RandomState(self.seed)

        permutation = rng.permutation(normal_num)

        support_indices = permutation[:self.fewshot]

        # 排序仅用于日志可读和结果稳定
        return np.sort(support_indices)

    def update(self, category=None):
        if category is not None and category != "liver":
            raise ValueError(
                "LiverDataset only supports category='liver', "
                f"but received category={category}"
            )

        self.category = "liver"

        support_indices = self._get_support_indices()

        if self.train and self.fewshot > 0:
            # ------------------------------------------------
            # Few-shot support：
            # 只使用选中的正常图像
            # ------------------------------------------------
            self.cur_img_paths = [
                self.good_img_paths[idx]
                for idx in support_indices
            ]

            self.cur_gt_paths = [
                self.good_gt_paths[idx]
                for idx in support_indices
            ]

            self.cur_img_labels = [
                0
            ] * len(self.cur_img_paths)

        else:
            # ------------------------------------------------
            # Query：
            # zero-shot 时：全部正常 + 全部异常
            # few-shot 时：剩余正常 + 全部异常
            # ------------------------------------------------
            all_good_indices = np.arange(
                len(self.good_img_paths)
            )

            if self.fewshot > 0:
                query_good_indices = np.setdiff1d(
                    all_good_indices,
                    support_indices
                )
            else:
                query_good_indices = all_good_indices

            query_good_img_paths = [
                self.good_img_paths[idx]
                for idx in query_good_indices
            ]

            query_good_gt_paths = [
                self.good_gt_paths[idx]
                for idx in query_good_indices
            ]

            self.cur_img_paths = (
                query_good_img_paths
                + self.bad_img_paths
            )

            self.cur_gt_paths = (
                query_good_gt_paths
                + self.bad_gt_paths
            )

            self.cur_img_labels = (
                [0] * len(query_good_img_paths)
                + [1] * len(self.bad_img_paths)
            )

        self.cur_img_categories = [
            "liver"
        ] * len(self.cur_img_paths)

        assert len(self.cur_img_paths) == len(
            self.cur_gt_paths
        )

        assert len(self.cur_img_paths) == len(
            self.cur_img_labels
        )

        assert len(self.cur_img_paths) == len(
            self.cur_img_categories
        )

    def __len__(self):
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        img_path = self.cur_img_paths[idx]
        gt_path = self.cur_gt_paths[idx]
        label = self.cur_img_labels[idx]
        category = self.cur_img_categories[idx]

        img = Image.open(img_path).convert("RGB")

        if gt_path is not None:
            gt = Image.open(gt_path).convert("L")
        else:
            # 正常图像没有 mask 时生成全零掩码
            gt = Image.new(
                mode="L",
                size=img.size,
                color=0
            )

        if self.transform is not None:
            img = self.transform(img)

        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        return img, label, gt, category, img_path