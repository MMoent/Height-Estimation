import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Define Sobel filters as parameters
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], 
                                                      [-2, 0, 2], 
                                                      [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], 
                                                      [ 0,  0,  0], 
                                                      [ 1,  2,  1]], dtype=torch.float32).reshape(1, 1, 3, 3))

    def forward(self, pred, true):
        # Apply Sobel filter to compute x and y gradients for both predicted and true images
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_true_x = F.conv2d(true, self.sobel_x, padding=1)
        grad_true_y = F.conv2d(true, self.sobel_y, padding=1)

        # Calculate the L1 loss between the gradients of the predicted and true images
        grad_diff_x = torch.abs(grad_pred_x - grad_true_x)
        grad_diff_y = torch.abs(grad_pred_y - grad_true_y)

        # Calculate mean loss across all batches and both gradient directions
        loss = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)

        return loss
