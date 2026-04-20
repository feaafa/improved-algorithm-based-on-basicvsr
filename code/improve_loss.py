# mmedit/models/losses/improve_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class ImproveLoss(nn.Module):
    """Enhanced loss function for BasicVSR++."""

    def __init__(self,
                 pixel_weight=1.0,
                 perceptual_weight=1.0,
                 temporal_weight=0.2,
                 edge_weight=0.5,
                 multi_scale_weights=[1.0, 0.5, 0.25],
                 use_perceptual=True,
                 use_temporal=True,
                 use_edge=True,
                 reduction='mean',
                 eps=1e-12):
        super().__init__()

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.edge_weight = edge_weight
        self.multi_scale_weights = multi_scale_weights
        self.reduction = reduction
        self.eps = eps

        self.use_perceptual = use_perceptual
        self.use_temporal = use_temporal
        self.use_edge = use_edge

        # Perceptual loss
        if self.use_perceptual:
            self.perceptual_loss = _SimplePerceptualLoss()

        # Edge loss
        if self.use_edge:
            self.edge_loss = _EdgeLoss()

    def _charbonnier_loss(self, pred, target):
        """Charbonnier Loss implementation."""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, pred, gt):
        """Forward function.

        Args:
            pred (Tensor): Predicted frames with shape (n, t, c, h, w)
            gt (Tensor): Ground truth frames with shape (n, t, c, h, w)

        Returns:
            dict: Dictionary of losses
        """
        losses = {}

        n, t, c, h, w = pred.shape

        # Reshape for batch processing
        pred_reshape = pred.reshape(n * t, c, h, w)
        gt_reshape = gt.reshape(n * t, c, h, w)

        # 1. Multi-scale Charbonnier loss
        # 使用列表收集每个尺度的损失，然后求和
        scale_losses = []
        for i, weight in enumerate(self.multi_scale_weights):
            if i == 0:
                # Original resolution
                loss_i = self._charbonnier_loss(pred_reshape, gt_reshape)
                scale_losses.append(weight * loss_i)
            else:
                # Downsampled resolutions
                scale_factor = 1.0 / (2 ** i)
                pred_down = F.interpolate(
                    pred_reshape,
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=True
                )
                gt_down = F.interpolate(
                    gt_reshape,
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=True
                )
                loss_i = self._charbonnier_loss(pred_down, gt_down)
                scale_losses.append(weight * loss_i)

        # 求和所有尺度的损失
        total_scale_loss = sum(scale_losses)
        pixel_loss = total_scale_loss / sum(self.multi_scale_weights)

        # 确保是张量
        losses['loss_pix'] = pixel_loss * self.pixel_weight

        # 验证 loss_pix 是张量
        assert torch.is_tensor(losses['loss_pix']), f"loss_pix should be tensor, got {type(losses['loss_pix'])}"

        # 2. Perceptual loss
        if self.use_perceptual and self.perceptual_weight > 0:
            percep_loss = self.perceptual_loss(pred_reshape, gt_reshape)
            losses['loss_percep'] = percep_loss * self.perceptual_weight
            assert torch.is_tensor(
                losses['loss_percep']), f"loss_percep should be tensor, got {type(losses['loss_percep'])}"

        # 3. Temporal consistency loss
        if self.use_temporal and self.temporal_weight > 0 and t > 1:
            pred_diff = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
            gt_diff = gt[:, 1:, :, :, :] - gt[:, :-1, :, :, :]

            temporal_loss = self._charbonnier_loss(
                pred_diff.reshape(-1, c, h, w),
                gt_diff.reshape(-1, c, h, w)
            )
            losses['loss_temporal'] = temporal_loss * self.temporal_weight
            assert torch.is_tensor(
                losses['loss_temporal']), f"loss_temporal should be tensor, got {type(losses['loss_temporal'])}"

        # 4. Edge-enhanced loss
        if self.use_edge and self.edge_weight > 0:
            edge_loss = self.edge_loss(pred_reshape, gt_reshape)
            losses['loss_edge'] = edge_loss * self.edge_weight
            assert torch.is_tensor(losses['loss_edge']), f"loss_edge should be tensor, got {type(losses['loss_edge'])}"

        # 打印调试信息（首次迭代）
        if not hasattr(self, '_first_forward_done'):
            print("\n=== ImproveLoss Debug Info ===")
            for key, value in losses.items():
                print(
                    f"{key}: type={type(value)}, shape={value.shape if torch.is_tensor(value) else 'N/A'}, value={value.item() if torch.is_tensor(value) else value}")
            print("=" * 40 + "\n")
            self._first_forward_done = True

        return losses


class _SimplePerceptualLoss(nn.Module):
    """Simplified perceptual loss using VGG19 features."""

    def __init__(self):
        super().__init__()

        try:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True)
            print("Successfully loaded pretrained VGG19 for perceptual loss")
        except Exception as e:
            print(f"Warning: Failed to load pretrained VGG19: {e}")
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=False)

        self.slice1 = nn.Sequential(*list(vgg.features[:4]))
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))
        self.slice3 = nn.Sequential(*list(vgg.features[9:18]))
        self.slice4 = nn.Sequential(*list(vgg.features[18:27]))
        self.slice5 = nn.Sequential(*list(vgg.features[27:36]))

        for param in self.parameters():
            param.requires_grad = False

        self.eval()

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.weights = [0.1, 0.1, 1.0, 1.0, 1.0]

    def forward(self, pred, gt):
        """Compute perceptual loss."""
        # Normalize to [0, 1]
        if pred.max() > 1.0:
            pred = pred / 255.0
            gt = gt / 255.0

        pred = torch.clamp(pred, 0, 1)
        gt = torch.clamp(gt, 0, 1)

        # Apply ImageNet normalization
        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        # Extract features
        pred_feats = self._extract_features(pred)
        with torch.no_grad():
            gt_feats = self._extract_features(gt)

        # Compute loss - 使用列表收集，然后求和
        layer_losses = []
        for i, (pred_feat, gt_feat) in enumerate(zip(pred_feats, gt_feats)):
            layer_loss = F.l1_loss(pred_feat, gt_feat)
            layer_losses.append(self.weights[i] * layer_loss)

        # 确保返回的是张量
        total_loss = sum(layer_losses)
        return total_loss

    def _extract_features(self, x):
        """Extract VGG19 features."""
        feats = []
        h = self.slice1(x)
        feats.append(h)
        h = self.slice2(h)
        feats.append(h)
        h = self.slice3(h)
        feats.append(h)
        h = self.slice4(h)
        feats.append(h)
        h = self.slice5(h)
        feats.append(h)
        return feats


class _EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, gt):
        """Compute edge loss."""
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1, :, :] + 0.587 * pred[:, 1:2, :, :] + 0.114 * pred[:, 2:3, :, :]
        gt_gray = 0.299 * gt[:, 0:1, :, :] + 0.587 * gt[:, 1:2, :, :] + 0.114 * gt[:, 2:3, :, :]

        # Compute edge gradients
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)

        gt_edge_x = F.conv2d(gt_gray, self.sobel_x, padding=1)
        gt_edge_y = F.conv2d(gt_gray, self.sobel_y, padding=1)

        # Compute edge magnitude
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + self.eps)
        gt_edge = torch.sqrt(gt_edge_x ** 2 + gt_edge_y ** 2 + self.eps)

        # Compute loss
        edge_diff = pred_edge - gt_edge
        edge_loss = torch.mean(torch.sqrt(edge_diff ** 2 + self.eps))

        # 确保返回的是标量张量
        return edge_loss