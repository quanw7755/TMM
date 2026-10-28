"""
  @Author: 王权
  @FileName: brain.py
  @DateTime: 2026/6/22 21:26
  @SoftWare: PyCharm
"""  

"""Brain anomaly dataset."""

import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class BrainDataset(Dataset):
    """
    Brain anomaly detection dataset.

    Expected directory structure:

        data/
        └── Med/
            └── Brain/
                └── test/
                    ├── good/
                    │   ├── img/
                    │   └── anomaly_mask/
                    └── Ungood/
                        ├── img/
                        └── anomaly_mask/

    Usage:

        train=False, fewshot=0:
            Use all good and Ungood images.

        train=False, fewshot>0:
            Use remaining good images and all Ungood images as query set.

        train=True, fewshot>0:
            Use K selected good images as support set.
    """

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

        # When root is:
        # /home/wsc2025/AF-CLIP/data
        #
        # The final path becomes:
        # /home/wsc2025/AF-CLIP/data/Med/Brain
        self.root = os.path.join(
            root,
            "Med",
            "Brain"
        )

        self.categories = ["brain"]
        self.category = "brain"
        self.dataset_name = "brain"

        self.preprocess()
        self.update(category)

    @staticmethod
    def _find_images(folder):
        """
        Find images with common extensions.
        """
        if folder is None or not os.path.isdir(folder):
            return []

        extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.tif",
            "*.tiff",
            "*.PNG",
            "*.JPG",
            "*.JPEG",
            "*.BMP",
            "*.TIF",
            "*.TIFF",
        ]

        image_paths = []

        for extension in extensions:
            image_paths.extend(
                glob.glob(
                    os.path.join(folder, extension)
                )
            )

        # Remove possible duplicate paths and keep stable order
        return sorted(set(image_paths))

    @staticmethod
    def _find_existing_dir(parent, candidate_names):
        """
        Find the first existing directory among several candidate names.
        """
        for candidate_name in candidate_names:
            candidate_dir = os.path.join(
                parent,
                candidate_name
            )

            if os.path.isdir(candidate_dir):
                return candidate_dir

        return None

    @staticmethod
    def _normalize_stem(path):
        """
        Normalize image and mask filenames for matching.

        Examples:
            001.png
            001_mask.png
            001_anomaly_mask.png

        They are all normalized to:
            001
        """
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]

        suffixes = [
            "_anomaly_mask",
            "-anomaly-mask",
            "-anomaly_mask",
            "_segmentation",
            "-segmentation",
            "_label",
            "-label",
            "_mask",
            "-mask",
            "_gt",
            "-gt",
        ]

        stem_lower = stem.lower()

        for suffix in suffixes:
            if stem_lower.endswith(suffix.lower()):
                stem = stem[:-len(suffix)]
                break

        return stem.lower()

    def _match_abnormal_masks(
        self,
        abnormal_img_paths,
        abnormal_mask_dir
    ):
        """
        Match every abnormal image with its anomaly mask.
        """
        mask_paths = self._find_images(
            abnormal_mask_dir
        )

        if len(mask_paths) == 0:
            raise RuntimeError(
                "No abnormal masks were found in:\n"
                f"{abnormal_mask_dir}"
            )

        mask_dict = {}

        for mask_path in mask_paths:
            mask_key = self._normalize_stem(
                mask_path
            )

            if mask_key in mask_dict:
                raise RuntimeError(
                    "Duplicate normalized mask filename detected:\n"
                    f"key: {mask_key}\n"
                    f"mask 1: {mask_dict[mask_key]}\n"
                    f"mask 2: {mask_path}"
                )

            mask_dict[mask_key] = mask_path

        matched_mask_paths = []

        for img_path in abnormal_img_paths:
            image_key = self._normalize_stem(
                img_path
            )

            mask_path = mask_dict.get(
                image_key,
                None
            )

            if mask_path is None:
                raise FileNotFoundError(
                    "Cannot find the corresponding anomaly mask.\n"
                    f"Image: {img_path}\n"
                    f"Normalized key: {image_key}\n"
                    f"Mask directory: {abnormal_mask_dir}"
                )

            matched_mask_paths.append(
                mask_path
            )

        return matched_mask_paths

    def preprocess(self):
        """
        Read all normal and abnormal images.

        Normal images:
            label = 0
            mask = None, and a zero mask is generated in __getitem__.

        Abnormal images:
            label = 1
            mask = the corresponding anomaly mask.
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                "Brain dataset root directory was not found:\n"
                f"{self.root}"
            )

        test_root = os.path.join(
            self.root,
            "test"
        )

        if not os.path.isdir(test_root):
            raise FileNotFoundError(
                "Brain test directory was not found:\n"
                f"{test_root}"
            )

        # -------------------------------------------------
        # Locate the normal directory
        # -------------------------------------------------
        good_root = self._find_existing_dir(
            test_root,
            [
                "good",
                "Good",
                "normal",
                "Normal"
            ]
        )

        if good_root is None:
            raise FileNotFoundError(
                "Cannot find the normal class directory under:\n"
                f"{test_root}\n"
                "Expected one of: good, Good, normal, Normal"
            )

        good_img_dir = self._find_existing_dir(
            good_root,
            [
                "img",
                "image",
                "images"
            ]
        )

        if good_img_dir is None:
            raise FileNotFoundError(
                "Cannot find the normal image directory under:\n"
                f"{good_root}\n"
                "Expected one of: img, image, images"
            )

        # -------------------------------------------------
        # Locate the abnormal directory
        # -------------------------------------------------
        abnormal_root = self._find_existing_dir(
            test_root,
            [
                "Ungood",
                "ungood",
                "bad",
                "Bad",
                "abnormal",
                "Abnormal"
            ]
        )

        if abnormal_root is None:
            raise FileNotFoundError(
                "Cannot find the abnormal class directory under:\n"
                f"{test_root}\n"
                "Expected one of: Ungood, ungood, bad, abnormal"
            )

        abnormal_img_dir = self._find_existing_dir(
            abnormal_root,
            [
                "img",
                "image",
                "images"
            ]
        )

        if abnormal_img_dir is None:
            raise FileNotFoundError(
                "Cannot find the abnormal image directory under:\n"
                f"{abnormal_root}\n"
                "Expected one of: img, image, images"
            )

        abnormal_mask_dir = self._find_existing_dir(
            abnormal_root,
            [
                "anomaly_mask",
                "anomaly_masks",
                "mask",
                "masks",
                "gt",
                "ground_truth"
            ]
        )

        if abnormal_mask_dir is None:
            raise FileNotFoundError(
                "Cannot find the abnormal mask directory under:\n"
                f"{abnormal_root}\n"
                "Expected one of: anomaly_mask, masks, gt, ground_truth"
            )

        # -------------------------------------------------
        # Read image paths
        # -------------------------------------------------
        self.good_img_paths = self._find_images(
            good_img_dir
        )

        self.abnormal_img_paths = self._find_images(
            abnormal_img_dir
        )

        if len(self.good_img_paths) == 0:
            raise RuntimeError(
                "No normal Brain images were found in:\n"
                f"{good_img_dir}"
            )

        if len(self.abnormal_img_paths) == 0:
            raise RuntimeError(
                "No abnormal Brain images were found in:\n"
                f"{abnormal_img_dir}"
            )

        # Normal masks are always treated as zero masks.
        # Even if good/anomaly_mask exists, it is not necessary to load it.
        self.good_gt_paths = [
            None
        ] * len(self.good_img_paths)

        # Abnormal images must have corresponding anomaly masks.
        self.abnormal_gt_paths = self._match_abnormal_masks(
            self.abnormal_img_paths,
            abnormal_mask_dir
        )

        self.good_labels = [
            0
        ] * len(self.good_img_paths)

        self.abnormal_labels = [
            1
        ] * len(self.abnormal_img_paths)

        print(
            "[Brain] normal images:",
            len(self.good_img_paths)
        )

        print(
            "[Brain] abnormal images:",
            len(self.abnormal_img_paths)
        )

        print(
            "[Brain] abnormal masks:",
            len(self.abnormal_gt_paths)
        )

    def _get_support_indices(self):
        """
        Select K normal support images using a fixed random seed.

        Under the same seed:
            1-shot is a subset of 2-shot.
            2-shot is a subset of 4-shot.
        """
        normal_num = len(
            self.good_img_paths
        )

        if self.fewshot <= 0:
            return np.array(
                [],
                dtype=np.int64
            )

        # Keep at least one normal image in the query set
        if self.fewshot >= normal_num:
            raise ValueError(
                f"fewshot={self.fewshot}, but Brain only has "
                f"{normal_num} normal images. At least one normal "
                "image must remain in the query set."
            )

        random_state = np.random.RandomState(
            self.seed
        )

        permutation = random_state.permutation(
            normal_num
        )

        support_indices = permutation[
            :self.fewshot
        ]

        # Sort indices only for stable and readable logging
        return np.sort(
            support_indices
        )

    def update(self, category=None):
        """
        Update the currently used data.

        When train=True and fewshot>0:
            Return K normal support images.

        When train=False and fewshot=0:
            Return all normal and abnormal images.

        When train=False and fewshot>0:
            Return the remaining normal images and all abnormal images.
        """
        if (
            category is not None
            and category != "brain"
        ):
            raise ValueError(
                "BrainDataset only supports category='brain', "
                f"but received category={category}"
            )

        self.category = "brain"

        support_indices = self._get_support_indices()

        # =================================================
        # Few-shot support set
        # =================================================
        if self.train and self.fewshot > 0:
            self.cur_img_paths = [
                self.good_img_paths[index]
                for index in support_indices
            ]

            self.cur_gt_paths = [
                None
            ] * len(self.cur_img_paths)

            self.cur_img_labels = [
                0
            ] * len(self.cur_img_paths)

        # =================================================
        # Source training set or query test set
        # =================================================
        else:
            all_good_indices = np.arange(
                len(self.good_img_paths)
            )

            if self.fewshot > 0:
                # Remove support normal images from query set
                query_good_indices = np.setdiff1d(
                    all_good_indices,
                    support_indices
                )
            else:
                query_good_indices = all_good_indices

            query_good_img_paths = [
                self.good_img_paths[index]
                for index in query_good_indices
            ]

            query_good_gt_paths = [
                None
            ] * len(query_good_img_paths)

            query_good_labels = [
                0
            ] * len(query_good_img_paths)

            self.cur_img_paths = (
                query_good_img_paths
                + self.abnormal_img_paths
            )

            self.cur_gt_paths = (
                query_good_gt_paths
                + self.abnormal_gt_paths
            )

            self.cur_img_labels = (
                query_good_labels
                + self.abnormal_labels
            )

        self.cur_img_categories = [
            "brain"
        ] * len(self.cur_img_paths)

        assert (
            len(self.cur_img_paths)
            == len(self.cur_gt_paths)
        ), (
            "Image and mask path numbers are inconsistent: "
            f"{len(self.cur_img_paths)} vs "
            f"{len(self.cur_gt_paths)}"
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_labels)
        ), (
            "Image and label numbers are inconsistent: "
            f"{len(self.cur_img_paths)} vs "
            f"{len(self.cur_img_labels)}"
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_categories)
        )

    def __len__(self):
        return len(
            self.cur_img_paths
        )

    def __getitem__(self, idx):
        img_path = self.cur_img_paths[idx]
        gt_path = self.cur_gt_paths[idx]
        label = self.cur_img_labels[idx]
        category = self.cur_img_categories[idx]

        img = Image.open(
            img_path
        ).convert("RGB")

        if label == 0 or gt_path is None:
            # Normal images always use an all-zero anomaly mask
            gt = Image.new(
                mode="L",
                size=img.size,
                color=0
            )
        else:
            gt = Image.open(
                gt_path
            ).convert("L")

            # Convert any positive mask value to foreground
            gt_array = np.array(
                gt,
                dtype=np.uint8
            )

            gt_array = (
                gt_array > 0
            ).astype(np.uint8) * 255

            gt = Image.fromarray(
                gt_array,
                mode="L"
            )

        if self.transform is not None:
            img = self.transform(
                img
            )

        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(
                gt
            )

        return (
            img,
            label,
            gt,
            category,
            img_path
        )

