"""
  @Author: 王权
  @FileName: chest.py
  @DateTime: 2026/6/22 23:40
  @SoftWare: PyCharm
"""

"""Chest anomaly classification dataset."""

import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class ChestDataset(Dataset):
    """
    Chest anomaly detection dataset.

    Expected directory structure:

        data/
        └── Med/
            └── Chest/
                └── test/
                    ├── good/
                    │   └── img/
                    └── Ungood/
                        └── img/

    Dataset behavior:

        train=False, fewshot=0:
            Use all good and Ungood images for evaluation.

        train=False, fewshot>0:
            Use the remaining good images and all Ungood images
            as the query set.

        train=True, fewshot>0:
            Select K good images as the few-shot support set.

    Important:
        Chest has no pixel-level anomaly masks.
        Zero masks are returned only to keep the interface
        compatible with the existing training and evaluation code.
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

        # Input:
        #   /home/wsc2025/AF-CLIP/data
        #
        # Final directory:
        #   /home/wsc2025/AF-CLIP/data/Med/Chest
        self.root = os.path.join(
            root,
            "Med",
            "Chest"
        )

        self.categories = ["chest"]
        self.category = "chest"
        self.dataset_name = "chest"

        self.has_image_gt = True
        self.has_pixel_gt = False
        self.supervision_mode = "cls"

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

        # Remove duplicate paths and keep a stable order
        return sorted(set(image_paths))

    @staticmethod
    def _find_existing_dir(parent, candidate_names):
        """
        Find the first existing directory from candidate names.
        """
        for candidate_name in candidate_names:
            candidate_dir = os.path.join(
                parent,
                candidate_name
            )

            if os.path.isdir(candidate_dir):
                return candidate_dir

        return None

    def preprocess(self):
        """
        Read normal and abnormal Chest images.

        Normal image:
            label = 0

        Abnormal image:
            label = 1

        Since Chest has no segmentation masks:
            gt_path = None for every image
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                "Chest dataset root directory was not found:\n"
                f"{self.root}"
            )

        test_root = os.path.join(
            self.root,
            "test"
        )

        if not os.path.isdir(test_root):
            raise FileNotFoundError(
                "Chest test directory was not found:\n"
                f"{test_root}"
            )

        # =================================================
        # Locate normal directory
        # =================================================
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
                "Cannot find the normal Chest directory under:\n"
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

        # =================================================
        # Locate abnormal directory
        # =================================================
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
                "Cannot find the abnormal Chest directory under:\n"
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

        # =================================================
        # Read paths
        # =================================================
        self.good_img_paths = self._find_images(
            good_img_dir
        )

        self.abnormal_img_paths = self._find_images(
            abnormal_img_dir
        )

        if len(self.good_img_paths) == 0:
            raise RuntimeError(
                "No normal Chest images were found in:\n"
                f"{good_img_dir}"
            )

        if len(self.abnormal_img_paths) == 0:
            raise RuntimeError(
                "No abnormal Chest images were found in:\n"
                f"{abnormal_img_dir}"
            )

        # Chest has no pixel-level ground-truth masks
        self.good_gt_paths = [
            None
        ] * len(self.good_img_paths)

        self.abnormal_gt_paths = [
            None
        ] * len(self.abnormal_img_paths)

        self.good_labels = [
            0
        ] * len(self.good_img_paths)

        self.abnormal_labels = [
            1
        ] * len(self.abnormal_img_paths)

        print(
            "[Chest] normal images:",
            len(self.good_img_paths)
        )

        print(
            "[Chest] abnormal images:",
            len(self.abnormal_img_paths)
        )

        print(
            "[Chest] pixel masks: 0 "
            "(classification-only dataset)"
        )

    def _get_support_indices(self):
        """
        Select K normal support images using a fixed seed.

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

        if self.fewshot >= normal_num:
            raise ValueError(
                f"fewshot={self.fewshot}, but Chest only has "
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

        return np.sort(
            support_indices
        )

    def update(self, category=None):
        """
        Update the currently used subset.

        train=True and fewshot>0:
            Return K normal support images.

        train=False and fewshot=0:
            Return all normal and abnormal images.

        train=False and fewshot>0:
            Return remaining normal images and all abnormal images.
        """
        if (
            category is not None
            and category != "chest"
        ):
            raise ValueError(
                "ChestDataset only supports category='chest', "
                f"but received category={category}"
            )

        self.category = "chest"

        support_indices = self._get_support_indices()

        # =================================================
        # Few-shot normal support set
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
        # Full source set or query test set
        # =================================================
        else:
            all_good_indices = np.arange(
                len(self.good_img_paths)
            )

            if self.fewshot > 0:
                # Remove support images from the query set
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
            "chest"
        ] * len(self.cur_img_paths)

        assert (
            len(self.cur_img_paths)
            == len(self.cur_gt_paths)
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_labels)
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
        label = self.cur_img_labels[idx]
        category = self.cur_img_categories[idx]

        img = Image.open(
            img_path
        ).convert("RGB")

        # Chest has no pixel-level anomaly annotations.
        # Return an all-zero mask only for interface compatibility.
        gt = Image.new(
            mode="L",
            size=img.size,
            color=0
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
