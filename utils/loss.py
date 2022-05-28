from typing import Tuple, Sequence

import torch
from torch import nn


class NSTLoss(nn.Module):

    def __init__(
            self,
            content_weight: float,
            style_weight: float,
            variance_weight: float
    ) -> None:
        """
        Class for neural style transfer loss
        :param content_weight: weight of content loss
        :param style_weight: weight of style loss
        :param variance_weight: weight of total variance loss
        """
        super(NSTLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variance_weight = variance_weight

        self.content_criterion = nn.MSELoss(reduction='mean')
        self.style_criterion = nn.MSELoss(reduction='sum')

    def forward(self,
                target_feature_maps: Sequence[torch.Tensor],
                target_gram_matrices: Sequence[torch.Tensor],
                img_feature_maps: Sequence[torch.Tensor],
                img_gram_matrices: Sequence[torch.Tensor],
                img: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate partial losses (needs to be sum)
        :param target_feature_maps: feature maps of content image
        :param target_gram_matrices: gram matrices of style image
        :param img_feature_maps: feature maps of trained image
        :param img_gram_matrices: gram matrices of trained image
        :param img: trained image
        :return: content loss, style loss and variance loss
        """
        content_loss = 0.0
        for tfe, cfe in zip(target_feature_maps, img_feature_maps):
            content_loss += self.content_criterion(tfe, cfe)
        content_loss *= (self.content_weight / len(target_feature_maps))

        style_loss = 0.0
        for tgm, cgm in zip(target_gram_matrices, img_gram_matrices):
            style_loss += self.style_criterion(tgm[0], cgm[0])
        style_loss *= (self.style_weight / len(target_gram_matrices))

        variance_loss = self.variance_weight * (
            torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        )

        return content_loss, style_loss, variance_loss
