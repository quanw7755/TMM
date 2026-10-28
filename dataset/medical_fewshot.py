"""
  @Author: 王权
  @FileName: medical_fewshot.py
  @DateTime: 2026/6/21 17:18
  @SoftWare: PyCharm
"""
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    "Brain",
    "Liver",
    "Retina_RESC",
    "Retina_OCT2017",
    "Chest",
    "Histopathology",
]

CLASS_INDEX = {
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
}

AF_CATEGORY_NAMES = {
    "Brain": "brainmri",
    "Liver": "liver",
    "Retina_RESC": "resc",
    "Retina_OCT2017": "oct2017",
    "Chest": "chest",
    "Histopathology": "histopathology",
}

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


class AFMedicalFewShotDataset(Dataset):
    """
    AF-CLIP医学少样本数据集。

    train=True:
        从 valid/good 和 valid/Ungood 中分别选择K张样本。

    train=False:
        加载 test/good 和 test/Ungood 中的全部测试样本。

    K-shot的定义：
        K张正常 + K张异常，总共2K张支持图像。
    """

    def __init__(
        self,
        root: str,
        class_name: str = "Brain",
        train: bool = False,
        fewshot: int = 0,
        iterate: int = -1,
        seed: int = 122,
        transform=None,
        gt_target_transform=None,
    ):
        super().__init__()

        if class_name not in CLASS_NAMES:
            raise ValueError(
                f"class_name={class_name} is invalid, "
                f"available classes: {CLASS_NAMES}"
            )

        if train and fewshot <= 0:
            raise ValueError(
                "Medical few-shot training requires fewshot > 0."
            )

        self.root = root
        self.class_name = class_name
        self.dataset_path = os.path.join(
            root,
            f"{class_name}_AD",
        )

        self.train = train
        self.fewshot = fewshot
        self.iterate = iterate
        self.seed = seed
        self.transform = transform
        self.gt_target_transform = gt_target_transform

        self.seg_flag = CLASS_INDEX[class_name]

        # 与AF-CLIP其他数据集保持一致
        self.category = AF_CATEGORY_NAMES[class_name]
        self.categories = [self.category]
        self.dataset_name = self.category

        if self.train:
            self.samples = self._load_fewshot_samples()
        else:
            self.samples = self._load_test_samples()

        print(
            f"[AFMedicalFewShotDataset] "
            f"class={self.class_name}, "
            f"train={self.train}, "
            f"fewshot={self.fewshot}, "
            f"samples={len(self.samples)}"
        )

    @staticmethod
    def _list_images(folder: str) -> List[str]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {folder}"
            )

        paths = [
            os.path.join(folder, filename)
            for filename in sorted(os.listdir(folder))
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if len(paths) == 0:
            raise RuntimeError(
                f"No images found in: {folder}"
            )

        return paths

    def _read_fixed_selection(
        self,
        state_prefix: str,
    ) -> Optional[List[str]]:
        """
        state_prefix:
            n: normal
            a: abnormal
        """
        if self.iterate < 0:
            return None

        seed_file = os.path.join(
            "./dataset/fewshot_seed",
            self.class_name,
            f"{self.fewshot}-shot.txt",
        )

        if not os.path.isfile(seed_file):
            raise FileNotFoundError(
                f"Few-shot seed file not found: {seed_file}"
            )

        target_prefix = f"{state_prefix}-{self.iterate}:"

        with open(seed_file, "r", encoding="utf-8") as infile:
            for line in infile:
                values = line.strip().split()

                if len(values) > 0 and values[0] == target_prefix:
                    return values[1:]

        raise RuntimeError(
            f"Cannot find {target_prefix} in {seed_file}"
        )

    def _select_paths(
        self,
        all_paths: List[str],
        state_prefix: str,
        random_seed: int,
    ) -> List[str]:
        if len(all_paths) < self.fewshot:
            raise ValueError(
                f"Requested {self.fewshot}-shot, "
                f"but only {len(all_paths)} samples are available."
            )

        fixed_names = self._read_fixed_selection(
            state_prefix
        )

        if fixed_names is not None:
            path_dict = {
                os.path.basename(path): path
                for path in all_paths
            }

            selected_paths = []

            for filename in fixed_names:
                if filename not in path_dict:
                    raise FileNotFoundError(
                        f"{filename} is not present in the dataset."
                    )

                selected_paths.append(
                    path_dict[filename]
                )

            if len(selected_paths) != self.fewshot:
                raise RuntimeError(
                    f"Expected {self.fewshot} selected samples, "
                    f"but obtained {len(selected_paths)}."
                )

            return selected_paths

        rng = random.Random(random_seed)

        return rng.sample(
            all_paths,
            self.fewshot,
        )

    def _load_fewshot_samples(
        self,
    ) -> List[Tuple[str, float, Optional[str]]]:
        normal_dir = os.path.join(
            self.dataset_path,
            "valid",
            "good",
            "img",
        )

        abnormal_dir = os.path.join(
            self.dataset_path,
            "valid",
            "Ungood",
            "img",
        )

        abnormal_mask_dir = os.path.join(
            self.dataset_path,
            "valid",
            "Ungood",
            "anomaly_mask",
        )

        normal_paths = self._list_images(normal_dir)
        abnormal_paths = self._list_images(abnormal_dir)

        selected_normal = self._select_paths(
            normal_paths,
            state_prefix="n",
            random_seed=self.seed,
        )

        selected_abnormal = self._select_paths(
            abnormal_paths,
            state_prefix="a",
            random_seed=self.seed + 1,
        )

        samples = []

        # 正常图像，掩码为None，后续转换成全零掩码
        for image_path in selected_normal:
            samples.append(
                (image_path, 0.0, None)
            )

        for image_path in selected_abnormal:
            if self.seg_flag > 0:
                mask_path = os.path.join(
                    abnormal_mask_dir,
                    os.path.basename(image_path),
                )

                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(
                        f"Mask not found: {mask_path}"
                    )
            else:
                mask_path = None

            samples.append(
                (image_path, 1.0, mask_path)
            )

        rng = random.Random(self.seed + 2)
        rng.shuffle(samples)

        return samples

    def _load_test_samples(
        self,
    ) -> List[Tuple[str, float, Optional[str]]]:
        normal_dir = os.path.join(
            self.dataset_path,
            "test",
            "good",
            "img",
        )

        abnormal_dir = os.path.join(
            self.dataset_path,
            "test",
            "Ungood",
            "img",
        )

        abnormal_mask_dir = os.path.join(
            self.dataset_path,
            "test",
            "Ungood",
            "anomaly_mask",
        )

        normal_paths = self._list_images(normal_dir)
        abnormal_paths = self._list_images(abnormal_dir)

        samples = []

        for image_path in normal_paths:
            samples.append(
                (image_path, 0.0, None)
            )

        for image_path in abnormal_paths:
            if self.seg_flag > 0:
                mask_path = os.path.join(
                    abnormal_mask_dir,
                    os.path.basename(image_path),
                )

                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(
                        f"Mask not found: {mask_path}"
                    )
            else:
                mask_path = None

            samples.append(
                (image_path, 1.0, mask_path)
            )

        return samples

    def update(self, category=None):
        # 与AF-CLIP的eval_all_class接口兼容
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        if self.transform is not None:
            image = self.transform(image)

        if mask_path is None:
            mask = Image.fromarray(
                np.zeros(
                    (original_height, original_width),
                    dtype=np.uint8,
                )
            )
        else:
            mask = Image.open(mask_path).convert("L")

        if self.gt_target_transform is not None:
            mask = self.gt_target_transform(mask)
        else:
            mask = torch.from_numpy(
                np.asarray(mask, dtype=np.float32)
            ).unsqueeze(0) / 255.0

        mask = (mask > 0.5).float()

        return (
            image,
            torch.tensor(label, dtype=torch.float32),
            mask,
            self.category,
            image_path,
        )