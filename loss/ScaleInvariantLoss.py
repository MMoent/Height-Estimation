import torch


class ScaleInvariantLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(ScaleInvariantLoss, self).__init__()

    def forward(self, output, target):
        # di = output - target
        di = target - output
        n = (256 * 256)
        di2 = torch.pow(di, 2)
        fisrt_term = torch.sum(di2, (1, 2, 3)) / n
        second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
        loss = fisrt_term - second_term
        return loss.mean()