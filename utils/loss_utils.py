import torch
import torch.fft


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def fourier_loss(pred, gt):
    pred_fft = torch.fft.fft2(pred, norm='ortho')
    gt_fft = torch.fft.fft2(gt, norm='ortho')

    return torch.mean(torch.abs(pred_fft - gt_fft) ** 2)
