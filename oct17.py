"""
  @Author: 王权
  @FileName: oct17.py
  @DateTime: 2026/6/23 10:33
  @SoftWare: PyCharm
"""

"""Retina OCT2017 anomaly classification dataset."""

import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class OCT17Dataset(Dataset):
    """
    Retina OCT2017 anomaly detection dataset.

    Expected directory structure:

        data/
        └── Med/
            └── Retina_OCT2017/
                └── test/
                    ├── good/
                    │   └── img/
                    └── Ungood/
                        └── img/

    Dataset behavior
    ----------------
    1. train=False, fewshot=0

       Return all normal and abnormal images.

       Used for:
           - source-domain classification training
           - zero-shot target-domain evaluation

    2. train=True, fewshot>0

       Select K normal images from test/good/img
       as the few-shot support set.

       Used for:
           - normal memory construction

    3. train=False, fewshot>0

       Return:
           remaining normal images
           +
           all abnormal images

       Support images are removed from the query set.

    Important
    ---------
    The current OCT17 directory does not contain pixel-level
    anomaly masks.

    Therefore:
        good   -> image-level label 0
        Ungood -> image-level label 1

    An all-zero mask is returned only for compatibility with
    the existing AF-CLIP dataset interface. It must not be used
    as a real segmentation label.
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
        #   /home/wsc2025/AF-CLIP/data
        #
        # Final path:
        #   /home/wsc2025/AF-CLIP/data/Med/Retina_OCT2017
        self.root = os.path.join(
            root,
            "Med",
            "Retina_OCT2017"
        )

        self.categories = ["oct17"]
        self.category = "oct17"
        self.dataset_name = "oct17"

        # Metadata used by the unified training/evaluation code
        self.has_pixel_gt = False
        self.supervision_mode = "cls"

        self.preprocess()
        self.update(category)

    @staticmethod
    def _find_images(folder):
        """
        Find images with common image extensions.
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
            "*.webp",
            "*.PNG",
            "*.JPG",
            "*.JPEG",
            "*.BMP",
            "*.TIF",
            "*.TIFF",
            "*.WEBP",
        ]

        image_paths = []

        for extension in extensions:
            image_paths.extend(
                glob.glob(
                    os.path.join(
                        folder,
                        extension
                    )
                )
            )

        # Remove duplicate paths and keep deterministic order
        return sorted(set(image_paths))

    @staticmethod
    def _find_existing_dir(
        parent,
        candidate_names
    ):
        """
        Find the first existing directory among candidate names.
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
        Read all normal and abnormal OCT17 images.
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                "OCT17 dataset root was not found:\n"
                f"{self.root}"
            )

        test_root = os.path.join(
            self.root,
            "test"
        )

        if not os.path.isdir(test_root):
            raise FileNotFoundError(
                "OCT17 test directory was not found:\n"
                f"{test_root}"
            )

        # =====================================================
        # Locate the normal directory
        # =====================================================
        good_root = self._find_existing_dir(
            test_root,
            [
                "good",
                "Good",
                "normal",
                "Normal",
            ]
        )

        if good_root is None:
            raise FileNotFoundError(
                "Cannot find the normal OCT17 directory under:\n"
                f"{test_root}\n"
                "Expected one of: good, Good, normal, Normal"
            )

        good_img_dir = self._find_existing_dir(
            good_root,
            [
                "img",
                "image",
                "images",
            ]
        )

        if good_img_dir is None:
            raise FileNotFoundError(
                "Cannot find the normal image directory under:\n"
                f"{good_root}\n"
                "Expected one of: img, image, images"
            )

        # =====================================================
        # Locate the abnormal directory
        # =====================================================
        abnormal_root = self._find_existing_dir(
            test_root,
            [
                "Ungood",
                "ungood",
                "bad",
                "Bad",
                "abnormal",
                "Abnormal",
                "anomaly",
                "Anomaly",
            ]
        )

        if abnormal_root is None:
            raise FileNotFoundError(
                "Cannot find the abnormal OCT17 directory under:\n"
                f"{test_root}\n"
                "Expected one of: Ungood, ungood, bad, abnormal"
            )

        abnormal_img_dir = self._find_existing_dir(
            abnormal_root,
            [
                "img",
                "image",
                "images",
            ]
        )

        if abnormal_img_dir is None:
            raise FileNotFoundError(
                "Cannot find the abnormal image directory under:\n"
                f"{abnormal_root}\n"
                "Expected one of: img, image, images"
            )

        # =====================================================
        # Read image paths
        # =====================================================
        self.good_img_paths = self._find_images(
            good_img_dir
        )

        self.abnormal_img_paths = self._find_images(
            abnormal_img_dir
        )

        if len(self.good_img_paths) == 0:
            raise RuntimeError(
                "No normal OCT17 images were found in:\n"
                f"{good_img_dir}"
            )

        if len(self.abnormal_img_paths) == 0:
            raise RuntimeError(
                "No abnormal OCT17 images were found in:\n"
                f"{abnormal_img_dir}"
            )

        # OCT17 has no real pixel-level masks
        self.good_gt_paths = [
            None
        ] * len(self.good_img_paths)

        self.abnormal_gt_paths = [
            None
        ] * len(self.abnormal_img_paths)

        # Image-level labels
        self.good_labels = [
            0
        ] * len(self.good_img_paths)

        self.abnormal_labels = [
            1
        ] * len(self.abnormal_img_paths)

        print(
            "[OCT17] normal images:",
            len(self.good_img_paths)
        )

        print(
            "[OCT17] abnormal images:",
            len(self.abnormal_img_paths)
        )

        print(
            "[OCT17] total images:",
            len(self.good_img_paths)
            + len(self.abnormal_img_paths)
        )

        print(
            "[OCT17] pixel masks: unavailable "
            "(classification-only dataset)"
        )

    def _get_support_indices(self):
        """
        Select K normal support images using a fixed seed.

        With the same seed:
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
                f"fewshot={self.fewshot}, but OCT17 only has "
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

        # Sorting makes the selected paths easier to inspect
        return np.sort(
            support_indices
        )

    def update(self, category=None):
        """
        Update the currently active subset.

        train=True and fewshot>0:
            return K normal support images.

        train=False and fewshot=0:
            return all normal and abnormal images.

        train=False and fewshot>0:
            return remaining normal images and all abnormal images.
        """
        if (
            category is not None
            and category not in ["oct17", "oct2017"]
        ):
            raise ValueError(
                "OCT17Dataset only supports category='oct17', "
                f"but received category={category}"
            )

        self.category = "oct17"

        support_indices = self._get_support_indices()

        # =====================================================
        # Few-shot normal support set
        # =====================================================
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

        # =====================================================
        # Source training set or target query set
        # =====================================================
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
            "oct17"
        ] * len(self.cur_img_paths)

        assert (
            len(self.cur_img_paths)
            == len(self.cur_gt_paths)
        ), (
            "The number of image paths and GT paths "
            "is inconsistent."
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_labels)
        ), (
            "The number of image paths and labels "
            "is inconsistent."
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_categories)
        ), (
            "The number of image paths and categories "
            "is inconsistent."
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

        # OCT17 has no real pixel-level anomaly mask.
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
