"""
  @Author: 王权
  @FileName: resc.py
  @DateTime: 2026/6/23 10:42
  @SoftWare: PyCharm
"""
"""RESC retinal anomaly dataset."""

import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class RESCDataset(Dataset):
    """
    RESC retinal anomaly detection dataset.

    Supported directory structure:

        data/
        └── Med/
            └── Retina_RESC_AD/
                ├── valid/                      # optional
                │   ├── good/
                │   │   └── img/
                │   └── Ungood/                # optional
                │       ├── img/
                │       └── anomaly_mask/      # optional
                │
                └── test/
                    ├── good/
                    │   └── img/
                    └── Ungood/
                        ├── img/
                        └── anomaly_mask/       # optional

    Modes
    -----
    train=False, fewshot=0:
        Use all test good and test abnormal images.

    train=False, fewshot>0:
        Use the target query set.

        If valid/good exists:
            query = complete test set.

        Otherwise:
            query = remaining test good + all test abnormal.

    train=True, fewshot>0:
        Return K normal support images.

        Priority:
            valid/good/img
            test/good/img

    train=True, fewshot=0:
        Used as a complete source training dataset.

        If valid contains both good and abnormal samples:
            use valid.

        Otherwise:
            use complete test data.
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

        # root:
        # /home/wsc2025/AF-CLIP/data
        #
        # final:
        # /home/wsc2025/AF-CLIP/data/Med/Retina_RESC_AD
        self.root = os.path.join(
            root,
            "Med",
            "Retina_RESC_AD"
        )

        self.categories = ["resc"]
        self.category = "resc"
        self.dataset_name = "resc"

        # These two attributes are refreshed in update().
        self.has_pixel_gt = False
        self.supervision_mode = "cls"

        self.preprocess()
        self.update(category)

    @staticmethod
    def _find_images(folder):
        """
        Find common image file formats.
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

        paths = []

        for extension in extensions:
            paths.extend(
                glob.glob(
                    os.path.join(folder, extension)
                )
            )

        return sorted(set(paths))

    @staticmethod
    def _find_existing_dir(parent, candidate_names):
        """
        Return the first existing candidate directory.
        """
        if parent is None or not os.path.isdir(parent):
            return None

        for candidate_name in candidate_names:
            directory = os.path.join(
                parent,
                candidate_name
            )

            if os.path.isdir(directory):
                return directory

        return None

    @staticmethod
    def _normalize_stem(path):
        """
        Normalize image and mask names for matching.

        Examples:
            001.png
            001_mask.png
            001_anomaly_mask.png

        All become:
            001
        """
        stem = os.path.splitext(
            os.path.basename(path)
        )[0]

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

        lower_stem = stem.lower()

        for suffix in suffixes:
            if lower_stem.endswith(suffix.lower()):
                stem = stem[:-len(suffix)]
                break

        return stem.lower()

    def _find_good_root(self, split_root):
        return self._find_existing_dir(
            split_root,
            [
                "good",
                "Good",
                "normal",
                "Normal",
            ]
        )

    def _find_abnormal_root(self, split_root):
        return self._find_existing_dir(
            split_root,
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

    def _find_image_dir(self, class_root):
        return self._find_existing_dir(
            class_root,
            [
                "img",
                "image",
                "images",
            ]
        )

    def _find_mask_dir(self, class_root):
        return self._find_existing_dir(
            class_root,
            [
                "anomaly_mask",
                "anomaly_masks",
                "mask",
                "masks",
                "gt",
                "ground_truth",
            ]
        )

    def _match_masks(
        self,
        image_paths,
        mask_dir,
        required=False
    ):
        """
        Match abnormal images with masks.

        If mask_dir does not exist and required=False:
            return None for every image.

        If mask_dir exists but masks are incomplete:
            raise an error to prevent silent image-mask mismatch.
        """
        if mask_dir is None:
            if required:
                raise FileNotFoundError(
                    "The anomaly mask directory was not found."
                )

            return [None] * len(image_paths)

        mask_paths = self._find_images(mask_dir)

        if len(mask_paths) == 0:
            if required:
                raise RuntimeError(
                    f"No mask images were found in: {mask_dir}"
                )

            return [None] * len(image_paths)

        mask_dict = {}

        for mask_path in mask_paths:
            key = self._normalize_stem(mask_path)

            if key in mask_dict:
                raise RuntimeError(
                    "Duplicate normalized mask name detected:\n"
                    f"key: {key}\n"
                    f"mask 1: {mask_dict[key]}\n"
                    f"mask 2: {mask_path}"
                )

            mask_dict[key] = mask_path

        matched_paths = []
        missing_images = []

        for image_path in image_paths:
            key = self._normalize_stem(image_path)
            mask_path = mask_dict.get(key, None)

            if mask_path is None:
                missing_images.append(image_path)

            matched_paths.append(mask_path)

        # A partially annotated dataset must not be silently accepted.
        if len(missing_images) > 0:
            examples = "\n".join(missing_images[:10])

            raise FileNotFoundError(
                "Some RESC abnormal images do not have matching masks.\n"
                f"Mask directory: {mask_dir}\n"
                f"Missing number: {len(missing_images)}\n"
                f"Examples:\n{examples}"
            )

        return matched_paths

    def _read_split(self, split_name, required=False):
        """
        Read one split such as test or valid.
        """
        split_root = os.path.join(
            self.root,
            split_name
        )

        empty_result = {
            "good_images": [],
            "good_masks": [],
            "good_labels": [],
            "abnormal_images": [],
            "abnormal_masks": [],
            "abnormal_labels": [],
            "has_pixel_gt": False,
        }

        if not os.path.isdir(split_root):
            if required:
                raise FileNotFoundError(
                    f"RESC split directory was not found: {split_root}"
                )

            return empty_result

        # =====================================================
        # Normal images
        # =====================================================
        good_root = self._find_good_root(split_root)
        good_img_dir = self._find_image_dir(good_root)

        if required and good_img_dir is None:
            raise FileNotFoundError(
                "Cannot find the RESC normal image directory.\n"
                f"Split: {split_root}\n"
                "Expected: good/img or normal/images"
            )

        good_images = self._find_images(good_img_dir)

        if required and len(good_images) == 0:
            raise RuntimeError(
                "No RESC normal images were found in:\n"
                f"{good_img_dir}"
            )

        # Normal images always use zero anomaly masks.
        good_masks = [None] * len(good_images)
        good_labels = [0] * len(good_images)

        # =====================================================
        # Abnormal images
        # =====================================================
        abnormal_root = self._find_abnormal_root(
            split_root
        )

        abnormal_img_dir = self._find_image_dir(
            abnormal_root
        )

        abnormal_images = self._find_images(
            abnormal_img_dir
        )

        if required and len(abnormal_images) == 0:
            raise RuntimeError(
                "No RESC abnormal images were found under:\n"
                f"{split_root}\n"
                "Expected: Ungood/img or abnormal/images"
            )

        abnormal_mask_dir = self._find_mask_dir(
            abnormal_root
        )

        abnormal_masks = self._match_masks(
            abnormal_images,
            abnormal_mask_dir,
            required=False
        )

        abnormal_labels = [
            1
        ] * len(abnormal_images)

        has_pixel_gt = (
            len(abnormal_images) > 0
            and len(abnormal_masks) == len(abnormal_images)
            and all(
                path is not None
                for path in abnormal_masks
            )
        )

        return {
            "good_images": good_images,
            "good_masks": good_masks,
            "good_labels": good_labels,

            "abnormal_images": abnormal_images,
            "abnormal_masks": abnormal_masks,
            "abnormal_labels": abnormal_labels,

            "has_pixel_gt": has_pixel_gt,
        }

    def preprocess(self):
        """
        Read test and optional valid splits.
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                "RESC dataset root was not found:\n"
                f"{self.root}"
            )

        # Test is required.
        self.test_data = self._read_split(
            split_name="test",
            required=True
        )

        # Valid is optional.
        self.valid_data = self._read_split(
            split_name="valid",
            required=False
        )

        # Prefer valid/good as the few-shot support pool.
        if len(self.valid_data["good_images"]) > 0:
            self.support_good_images = (
                self.valid_data["good_images"]
            )

            self.support_good_masks = (
                self.valid_data["good_masks"]
            )

            self.support_from_valid = True
        else:
            self.support_good_images = (
                self.test_data["good_images"]
            )

            self.support_good_masks = (
                self.test_data["good_masks"]
            )

            self.support_from_valid = False

        print(
            "[RESC] test normal images:",
            len(self.test_data["good_images"])
        )

        print(
            "[RESC] test abnormal images:",
            len(self.test_data["abnormal_images"])
        )

        print(
            "[RESC] test pixel masks:",
            (
                len(self.test_data["abnormal_masks"])
                if self.test_data["has_pixel_gt"]
                else 0
            )
        )

        print(
            "[RESC] valid normal images:",
            len(self.valid_data["good_images"])
        )

        print(
            "[RESC] few-shot support source:",
            (
                "valid/good/img"
                if self.support_from_valid
                else "test/good/img"
            )
        )

        print(
            "[RESC] supervision:",
            (
                "classification + segmentation"
                if self.test_data["has_pixel_gt"]
                else "classification only"
            )
        )

    def _get_support_indices(self):
        """
        Select K normal support images with a fixed seed.
        """
        support_num = len(
            self.support_good_images
        )

        if self.fewshot <= 0:
            return np.array(
                [],
                dtype=np.int64
            )

        # When using valid as support, all valid normal images
        # may theoretically be selected.
        #
        # When using test good images, at least one normal query
        # image must remain.
        maximum_fewshot = (
            support_num
            if self.support_from_valid
            else support_num - 1
        )

        if self.fewshot > maximum_fewshot:
            raise ValueError(
                f"fewshot={self.fewshot}, but the maximum allowed "
                f"for RESC is {maximum_fewshot}. "
                f"Support source: "
                f"{'valid' if self.support_from_valid else 'test'}"
            )

        random_state = np.random.RandomState(
            self.seed
        )

        permutation = random_state.permutation(
            support_num
        )

        support_indices = permutation[
            :self.fewshot
        ]

        return np.sort(
            support_indices
        )

    def _use_valid_as_full_source(self):
        """
        Use valid as a complete source dataset only when it
        contains both normal and abnormal samples.
        """
        return (
            len(self.valid_data["good_images"]) > 0
            and len(self.valid_data["abnormal_images"]) > 0
        )

    def update(self, category=None):
        """
        Update the active data subset.
        """
        if (
            category is not None
            and category != "resc"
        ):
            raise ValueError(
                "RESCDataset only supports category='resc', "
                f"but received category={category}"
            )

        self.category = "resc"

        support_indices = self._get_support_indices()

        # =====================================================
        # Few-shot support set
        # =====================================================
        if self.train and self.fewshot > 0:
            self.cur_img_paths = [
                self.support_good_images[index]
                for index in support_indices
            ]

            self.cur_gt_paths = [
                None
            ] * len(self.cur_img_paths)

            self.cur_img_labels = [
                0
            ] * len(self.cur_img_paths)

            self.has_pixel_gt = False
            self.supervision_mode = "cls"

        # =====================================================
        # Complete source training set
        # =====================================================
        elif self.train and self.fewshot == 0:
            if self._use_valid_as_full_source():
                source_data = self.valid_data
            else:
                source_data = self.test_data

            self.cur_img_paths = (
                source_data["good_images"]
                + source_data["abnormal_images"]
            )

            self.cur_gt_paths = (
                source_data["good_masks"]
                + source_data["abnormal_masks"]
            )

            self.cur_img_labels = (
                source_data["good_labels"]
                + source_data["abnormal_labels"]
            )

            self.has_pixel_gt = source_data[
                "has_pixel_gt"
            ]

            self.supervision_mode = (
                "joint"
                if self.has_pixel_gt
                else "cls"
            )

        # =====================================================
        # Test query set
        # =====================================================
        else:
            test_good_images = self.test_data[
                "good_images"
            ]

            test_good_masks = self.test_data[
                "good_masks"
            ]

            # If support comes from test/good, remove selected
            # support images from the test query.
            if (
                self.fewshot > 0
                and not self.support_from_valid
            ):
                all_good_indices = np.arange(
                    len(test_good_images)
                )

                query_good_indices = np.setdiff1d(
                    all_good_indices,
                    support_indices
                )

                query_good_images = [
                    test_good_images[index]
                    for index in query_good_indices
                ]

                query_good_masks = [
                    test_good_masks[index]
                    for index in query_good_indices
                ]
            else:
                query_good_images = test_good_images
                query_good_masks = test_good_masks

            self.cur_img_paths = (
                query_good_images
                + self.test_data["abnormal_images"]
            )

            self.cur_gt_paths = (
                query_good_masks
                + self.test_data["abnormal_masks"]
            )

            self.cur_img_labels = (
                [0] * len(query_good_images)
                + self.test_data["abnormal_labels"]
            )

            self.has_pixel_gt = self.test_data[
                "has_pixel_gt"
            ]

            self.supervision_mode = (
                "joint"
                if self.has_pixel_gt
                else "cls"
            )

        self.cur_img_categories = [
            "resc"
        ] * len(self.cur_img_paths)

        assert (
            len(self.cur_img_paths)
            == len(self.cur_gt_paths)
        ), (
            "RESC image and GT numbers are inconsistent: "
            f"{len(self.cur_img_paths)} vs "
            f"{len(self.cur_gt_paths)}"
        )

        assert (
            len(self.cur_img_paths)
            == len(self.cur_img_labels)
        ), (
            "RESC image and label numbers are inconsistent: "
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

        if gt_path is None:
            # Normal image or classification-only abnormal image.
            # Zero mask is returned only for interface compatibility.
            gt = Image.new(
                mode="L",
                size=img.size,
                color=0
            )
        else:
            gt = Image.open(
                gt_path
            ).convert("L")

            gt_array = np.array(
                gt,
                dtype=np.uint8
            )

            # Convert all positive values to binary foreground.
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