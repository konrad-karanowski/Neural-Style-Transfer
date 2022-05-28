from typing import Iterable, Sequence, Tuple

import torch

from models import VGG19Model
from utils.gramm_matrix import gram_matrix
from utils.loss import NSTLoss
from utils.video_writer import VideoWriter


def initialize_img(content_img: torch.Tensor, method: str, device: torch.device) -> torch.Tensor:
    """
    Initialize image using given method:
    - noise: start from random noise (prefered for image reconstruction)
    - content_img: start from content_img (best method for style transfer)
    :param content_img: content image
    :param method: string of init method
    :param device: device (cuda or cpu)
    :return: trainable image
    """
    if method == 'noise':
        img = torch.randint(
            255,
            size=content_img.shape,
            requires_grad=True,
            dtype=content_img.dtype,
            device=device
        )
    elif method == 'content_img':
        img = torch.tensor(
            content_img,
            requires_grad=True,
            device=device
        )
    else:
        raise Exception(f'Invalid initialization method')
    return img


def get_image_features(
        img: torch.Tensor,
        model: VGG19Model,
        content_layers: Iterable[int],
        style_layers: Iterable[int],
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """
    Obtains feature maps and gram matrices from image using given model
    :param img: image
    :param model: model
    :param content_layers: indexes of layers for content reconstruction
    :param style_layers: indexes of layers for style transfer
    :return: feature maps and gram matrices for given image
    """
    feature_maps = model(img)
    gram_matrices = [gram_matrix(img) for i, img in enumerate(feature_maps) if i in style_layers]
    final_feature_maps = [fe for i, fe in enumerate(feature_maps) if i in content_layers]
    return final_feature_maps, gram_matrices


def train_model(
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        lr: float,
        max_epochs: int,
        content_weight: float,
        style_weight: float,
        variance_weight: float,
        content_layers: Iterable[int],
        style_layers: Iterable[int],
        initialization_method: str,
        optimizer_str: str,
        video_writer: VideoWriter,
) -> None:
    """
    Trains neural style transfer model
    :param content_img: content image
    :param style_img: style image
    :param lr: learning rate
    :param max_epochs: max number of iterations
    :param content_weight: weight of content loss
    :param style_weight: weight of style loss
    :param variance_weight: weight of total variance loss
    :param content_layers: indexes of layers for content reconstruction
    :param style_layers: indexes of layers for style transfer
    :param initialization_method: method of initializing trainable image
    :param optimizer_str: string of optimizer (optimizing method)
    :param video_writer: class of video writer
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected device: {device}')
    model = VGG19Model()
    model.to(device)

    trainable_img = initialize_img(content_img, method=initialization_method, device=device)
    if style_weight > 0 and content_weight > 0:
        video_writer.append_img(content_img[0].permute(1, 2, 0).numpy())
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    target_feature_maps, _ = get_image_features(content_img, model, content_layers, style_layers)
    _, target_gram_matrices = get_image_features(style_img, model, content_layers, style_layers)

    criterion = NSTLoss(content_weight=content_weight, style_weight=style_weight, variance_weight=variance_weight)
    if optimizer_str == 'lbfgs':
        optimizer = torch.optim.LBFGS([trainable_img, ], max_iter=max_epochs)

        def _closure():
            optimizer.zero_grad()
            img_features, img_grams = get_image_features(trainable_img, model, content_layers, style_layers)
            total_loss = sum(criterion(
                target_feature_maps=target_feature_maps,
                target_gram_matrices=target_gram_matrices,
                img_feature_maps=img_features,
                img_gram_matrices=img_grams,
                img=trainable_img
            ))
            total_loss.backward()
            video_writer.append_img(trainable_img.cpu().detach()[0].permute(1, 2, 0).numpy())
            return total_loss

        optimizer.step(_closure)
    else:
        raise Exception(f'Unsupported optimizer: {optimizer_str}')
