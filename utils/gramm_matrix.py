import torch


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Calculate gram matrix
    :param features: features of image
    :return: gram matrix
    """
    b, ch, h, w = features.size()
    features = features.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gm = features.bmm(features_t)
    gm /= ch * h * w
    return gm
