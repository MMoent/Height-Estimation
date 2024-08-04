import torch


class RankingLoss(torch.nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()

    def forward(self, output, ordinal):
        # output: [bs, 1, h, w], target: [bs, N, 5]
        xa = ordinal[:, :, 0]
        ya = ordinal[:, :, 1]
        xb = ordinal[:, :, 2]
        yb = ordinal[:, :, 3]
        r = ordinal[:, :, 4]     # [bs, N]

        batch_indices = torch.arange(output.shape[0])[:, None]
        a = output[batch_indices, 0, xa, ya]
        b = output[batch_indices, 0, xb, yb]

        loss = torch.where(r==0, torch.square(a-b), \
                    torch.where(r==1, torch.log(1+torch.exp(a-b)), torch.log(1+torch.exp(b-a))))

        return torch.mean(loss)
