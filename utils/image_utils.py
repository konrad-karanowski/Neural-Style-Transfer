from typing import Tuple

import cv2
import torch
import numpy as np
from torchvision.transforms.transforms import Compose, ToTensor, Lambda, Normalize

IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [1.0, 1.0, 1.0]


def transform_img(img: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """
    Transform image
    :param img: input img as np ndarray
    :param normalize: whether to normalize or not
    :return: image as tensor [1, 3, h, w]
    """
    transforms_list = [
        ToTensor(),
        Lambda(lambda x: x.mul(255)), # this works lol xD
    ]
    if normalize:
        transforms_list.extend([
            Normalize(IMG_MEAN, IMG_STD)
        ])
    transforms = Compose(transforms_list)
    img = transforms(img)
    return img.unsqueeze(0)


def read_img(
        img_path: str,
        target_shape: Tuple[int, int] = (256, 256),
        normalize: bool = False) -> torch.Tensor:
    """
    Reads image and returns it
    :param img_path: path to img
    :param target_shape: target to reshape img
    :param normalize: whether to normalize or not
    :return: prepared img
    """
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"Couldn't find {img_path}, checkout name")
    img = cv2.resize(img[:, :, ::-1], target_shape, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = transform_img(img / 255.0, normalize)
    return img
