import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class HeightRenderer(torch.nn.Module):
    def __init__(self, BS, W=256, H=256, total_sample_num=64, origin_height=64, max_height=64, plane_scale=128, total_sample_len=64, use_ground=True):
        super(HeightRenderer, self).__init__()
        self.W, self.H = W, H
        self.BS, self.use_ground = BS, use_ground
        self.total_sample_len, self.total_sample_num = total_sample_len, total_sample_num
        self.origin_height, self.max_height = origin_height, max_height
        self.plane_scale = plane_scale
        self.total_sample_len = total_sample_len
        
        _x = torch.linspace(-1, 1, self.W)
        _y = torch.linspace(-1, 1, self.H)
        _z = torch.linspace(1, -1, self.total_sample_num)
        xv, yv, zv = torch.meshgrid(_x, _y, _z, indexing='xy')
        grid = torch.cat([xv[..., None], yv[..., None], zv[..., None]], dim=-1)[None,...].expand(self.BS, -1, -1, -1, -1)        
        self.grid = grid.permute(0, 3, 1, 2, 4).to('cuda')  # B Z X Y 3
        sample_len = ((torch.arange(self.total_sample_num)+1)*(self.total_sample_len/self.total_sample_num)).to("cuda")
        self.depth = sample_len[None,None,None,None,:]
        
    def forward(self, density):
        if self.use_ground:
            density = torch.cat([torch.ones(density.size(0),1,density.size(2),density.size(3), device='cuda')*1e3,density],1).to('cuda')
        alpha_grid = torch.nn.functional.grid_sample(density.unsqueeze(1), self.grid, align_corners=True)
        depth_sample = self.depth.permute(0,1,2,4,3).view(1,-1,self.total_sample_num,1)
        alpha_grid = alpha_grid.permute(0,3,4,2,1).view(self.BS,-1,self.total_sample_num)
        intv = self.total_sample_len/self.total_sample_num
        sigma_delta = alpha_grid*intv
        alpha = 1-(-sigma_delta).exp_()
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_()
        prob = (T*alpha)[...,None]
        depth = (depth_sample*prob).sum(dim=2)
        height = self.max_height - depth.permute(0,2,1).view(depth.size(0),-1,self.W, self.H)
        return height


class PanoRenderer(nn.Module):
    def __init__(self, BS, W=512, H=256, camera_height=2, max_height=64, plane_scale=128, total_sample_len=64, total_sample_num=300, use_ground=True) -> None:
        super().__init__()
        self.B, self.H, self.W = BS, H, W
        self.camera_height, self.max_height = camera_height, max_height
        self.plane_scale = plane_scale
        self.total_sample_len, self.total_sample_num = total_sample_len, total_sample_num
        self.use_ground = use_ground
        
        pano_direction = self.__get_pano_direction()[None, None, ...].expand(self.B, -1, -1, -1, -1)
        origin_z = (-1 + (self.camera_height * 2 / self.plane_scale)) * torch.ones(self.B, 1)
        origin_x, origin_y = torch.zeros(self.B, 1), torch.zeros(self.B, 1)
        origin = torch.cat([origin_x, origin_y, origin_z], dim=-1)[:, None, None, None, :]

        sample_len = torch.arange(1, self.total_sample_num+1) / self.total_sample_num
        sample_len = sample_len.view(1, -1, 1, 1, 1)
        self.depth = (sample_len * self.total_sample_len).permute(0, 2, 3, 4, 1).cuda()
        
        sample_points = origin + sample_len * pano_direction
        grid_min, grid_max = -1, -1 + self.max_height * 2 / self.plane_scale
        self.grid = sample_points.cuda()
        self.grid[...,2]   = ((sample_points[...,2]-grid_min)/(grid_max - grid_min)) * 2 - 1
        
    def __get_pano_direction(self):
        _x, _y = torch.meshgrid(torch.linspace(0, 1, self.H), torch.linspace(0, 1, self.W), indexing='ij')
        lat = (0.5 - _x) * torch.pi
        lon = (-0.5 - 2 * _y) * torch.pi
        axis1 = torch.cos(lat) * torch.cos(lon)
        axis2 = -torch.cos(lat) * torch.sin(lon)
        axis3 = torch.sin(lat)
        pano_direction = torch.concatenate([axis1[..., None], axis2[...,None], axis3[..., None]], dim=-1)
        return pano_direction
    
    def forward(self, voxel):
        voxel = torch.cat([torch.ones(voxel.size(0),1,voxel.size(2),voxel.size(3), device='cuda')*1000,voxel], dim=1) if self.use_ground else voxel
        alpha_grid = torch.nn.functional.grid_sample(voxel.unsqueeze(1), self.grid, align_corners=False)
        depth_sample = self.depth.permute(0,1,2,4,3).view(1,-1,self.total_sample_num,1)
        alpha_grid = alpha_grid.permute(0,3,4,2,1).view(self.B,-1,self.total_sample_num)
        intv = self.total_sample_len / self.total_sample_num
        sigma_delta = alpha_grid*intv 
        alpha = 1-(-sigma_delta).exp_() 
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() 
        prob = (T*alpha)[...,None]
        depth = (depth_sample*prob).sum(dim=2)
        opacity = prob.sum(dim=2)
        depth = depth.permute(0,2,1).view(self.B,-1,self.H, self.W)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        opacity = opacity.view(self.B,-1,self.H, self.W)
        return depth, opacity
    
    
if __name__ == "__main__":
    p_render = PanoRenderer(4, 512, 256, 2, 64, 128, 64, 300, True)
    h_render = HeightRenderer(4)
    x = torch.zeros(4, 64, 256, 256)
    x[:, :10, 18:28, 118:128] = 1.0
    x[:, :50, 118:128, 18:28] = 1.0
    x[:, :, 57:67, 18:28] = 1.0
    depth, opa = p_render(x.cuda())
    height = h_render(x.cuda())
    print(height.shape)
    plt.subplot(2, 1, 1)
    plt.imshow(height[0, 0, ...].cpu())    
    plt.subplot(2, 1, 2)
    plt.imshow(depth[0, 0, ...].cpu())
    plt.show()
    