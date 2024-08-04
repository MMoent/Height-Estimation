import os
import argparse
import random
import math
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import *
from equilib import equi2pers

from model import DualResUNet
from model import HeightRenderer, PanoRenderer

from loss import SSIMLoss, ScaleInvariantLoss, RankingLoss, GradientLoss
from skimage.metrics import structural_similarity as compare_ssim
from AerialStreetDataset import AerialStreetDataset


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    mae = np.mean(np.abs(gt - pred))
    ssim = compare_ssim(gt, pred, data_range=gt.max()-gt.min())

    return mae, rmse, ssim

    
def train(args):
    restore_from = args.restore_from
    seed = args.seed
    eps = args.epochs
    bs = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    net_type = args.net_type.lower()
    lr_decay = args.lr_decay
    num_workers = args.num_workers
    aerial_weight, street_weight = args.aerial_weight, args.street_weight
    use_rank, use_ssim, use_grad, use_sky = args.use_rank, args.use_ssim, args.use_grad, args.use_sky
        
    checkpoint = None
    if restore_from is not None:
        checkpoint = torch.load(restore_from)
        eps = checkpoint['epochs'] if eps is None else eps
        seed, bs, lr, weight_decay, lr_decay, net_type = checkpoint['seed'], checkpoint['batch_size'], checkpoint['lr'], checkpoint['weight_decay'], checkpoint['lr_decay'], checkpoint['net_type']
        use_rank, use_ssim, use_sky, use_grad = checkpoint.get('use_rank'), checkpoint.get('use_ssim'), checkpoint.get('use_sky'), checkpoint.get('use_grad')
        
        aerial_weight, street_weight = checkpoint.get('aerial_weight', 1.0), checkpoint.get('street_weight', 1.0)
        save_dir_path = os.path.split(restore_from)[0]
        print(f'Training restored. Model will be saved in {save_dir_path}')
        best_mae, best_rmse, best_ep, current_ep = checkpoint['best_mae'], checkpoint['best_rmse'], checkpoint['best_ep'], checkpoint['current_ep']
        
        log_path = os.path.join(save_dir_path, 'log.npy')
        log = np.load(log_path).tolist() if os.path.exists(log_path) else []
    else:
        time_now = datetime.now().strftime('%y%m%d%H%M%S')
        save_dir_path = os.path.join('experiments', 'density_st_fusion_cons', 
                                        f'DENSITY_STFUSIONCONS_{time_now}_{net_type.upper()}_EP{str(eps).zfill(3)}_BS{str(bs).zfill(2)}_LR{lr:.6f}'+ \
                                        ('' if lr_decay is None else f'_LD{str(int(lr_decay[0])).zfill(3)}+{lr_decay[1]:.2f}') + \
                                        f'_AW{aerial_weight:.1f}_SW{street_weight:.1f}' + \
                                        ('' if not use_rank else '_RANK') + ('' if not use_ssim else '_SSIM') + ('' if not use_grad else '_GRAD') + ('' if not use_sky else '_SKY')
                                    )
        os.makedirs(save_dir_path, exist_ok=True)
        print(f'Model will be saved in {save_dir_path}')
        best_mae, best_rmse, best_ep, current_ep = 1e7, 1e7, 0, 0
        log = []
        
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

    # dataset
    train_dataset = AerialStreetDataset(root='./dataset', train=True, require_st_cons=True)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataset = AerialStreetDataset(root='./dataset', train=False, test=False)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=True)
    
    # define network
    model = DualResUNet(in_channels_aerial=3, in_channels_pano=3, out_channels=64)
    model = model.cuda()
    torch.compile(model)
    
    height_renderer = HeightRenderer(BS=bs).cuda()
    pano_renderer = PanoRenderer(BS=bs).cuda()
    
    # define optimizer & lr scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    step_size, gamma = (100, 0.1) if lr_decay is None else (int(lr_decay[0]), lr_decay[1])
    lr_scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    
    if restore_from is not None:
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_decay:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
    # parallelize
    parallel = False
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_ids is not None:
        gpu_ids = list(map(int, gpu_ids.split(',')))
    if gpu_ids is not None and len(gpu_ids) > 1:
        parallel = True 
        model = torch.nn.DataParallel(model)
        
    # define loss
    scale_invariant_loss = ScaleInvariantLoss().cuda()
    ssim_loss = SSIMLoss().cuda()
    ranking_loss = RankingLoss().cuda()
    gradient_loss = GradientLoss().cuda()
    mae = L1Loss().cuda()
    mse = MSELoss().cuda()

    for ep in range(current_ep+1, eps+1):
        """training"""
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        train_loss = []
        for step, (aerial, ndsm, pano, sky, depth, ordinal) in loop:
            aerial, ndsm, pano, sky, depth, ordinal = aerial.cuda(), ndsm.cuda(), pano.cuda(), sky.cuda(), depth.cuda(), ordinal.cuda()
            # clear grad
            optimizer.zero_grad()
            density = model(aerial, pano)
            
            # aerial loss
            height = height_renderer(density)
            loss_aerial = scale_invariant_loss(height, ndsm)
            
            # street
            pano_depth, pano_opacity = pano_renderer(density)
            # if step % 200 == 0:
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(height[0, 0, ...].detach().cpu())
            #     plt.axis('off')
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(pano_depth[0, 0, ...].detach().cpu())
            #     plt.axis('off')
            #     plt.show()
            #     plt.close('all')
            
            loss_sky = 10.0 * torch.mean(torch.where(sky == 1.0, torch.abs(pano_opacity), torch.abs(pano_opacity - 1)))   # sky mask loss
            loss_rank, loss_ssim, loss_grad = [], [], []
            for h in range(4):
                rots = [{'roll': 0., 'pitch': 0., 'yaw': h * 90 / 180 * np.pi}] * bs
                depth_pers = equi2pers(pano_depth, rots, 256, 256, 90, z_down=True)
                loss_ssim.append(ssim_loss(depth_pers, depth[:, h, ...]))
                loss_rank.append(ranking_loss(depth_pers, ordinal[:, h, ...]))
                loss_grad.append(gradient_loss(depth_pers, depth[:, h, ...]))
            loss_street = float(use_sky) * loss_sky + float(use_rank) * sum(loss_rank) + float(use_ssim) * sum(loss_ssim) + float(use_grad) * sum(loss_grad)

            total_loss = aerial_weight * loss_aerial + street_weight * loss_street
            train_loss.append(total_loss)

            total_loss.backward()
            optimizer.step()
                
            loop.set_description(f"DSFC -- train on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: epoch [{ep}/{eps}]")
            loop.set_postfix(train_loss=total_loss.item())

        """validation"""
        model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            val_loss, val_mae, val_rmse = [], [], []
            for step, (aerial, ndsm, pano) in loop:
                aerial, ndsm, pano = aerial.cuda(), ndsm.cuda(), pano.cuda()
                density = model(aerial, pano)
                output = height_renderer(density)
                val_loss.append(scale_invariant_loss(output, ndsm))
                val_mae.append(mae(output, ndsm))
                val_rmse.append(torch.sqrt(mse(output, ndsm)))

                loop.set_description(f"DSFC -- val on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: epoch [{ep}/{eps}]")
                loop.set_postfix(val_loss=val_loss[-1].item())

        ave_train_loss = sum(train_loss) / len(train_loss)
        ave_train_loss = ave_train_loss.item()
        ave_val_loss = sum(val_loss) / len(val_loss)
        ave_val_loss = ave_val_loss.item()
        ave_val_mae = sum(val_mae) / len(val_mae)
        ave_val_mae = ave_val_mae.item()
        ave_val_rmse = sum(val_rmse) / len(val_rmse)
        ave_val_rmse = ave_val_rmse.item()
        log.append([ave_train_loss, ave_val_loss, ave_val_mae, ave_val_rmse])
        
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        if lr_decay is not None:
            lr_scheduler.step() # lr decay

        """save log"""
        np.save(os.path.join(save_dir_path, 'log.npy'), np.array(log))
        """save best model"""
        save_best = False
        if ave_val_mae < best_mae and ave_val_rmse < best_rmse:
            best_mae = ave_val_mae
            best_rmse = ave_val_rmse
            best_ep = ep
            save_best = True
        """save model for restore"""
        checkpoint = {
            'net': model.state_dict(),
            'seed': seed,
            'epochs': eps,
            'current_ep': ep,
            'batch_size': bs,
            'lr': lr,
            'current_lr': current_lr,
            'optimizer': optimizer.state_dict(),
            'weight_decay': weight_decay,
            'lr_decay': lr_decay,
            'lr_scheduler': None if lr_decay is None else lr_scheduler.state_dict(),
            'net_type': net_type,
            'aerial_weight': aerial_weight,
            'street_weight': street_weight,
            'use_rank': use_rank,
            'use_ssim': use_ssim,
            'use_grad': use_grad,
            'use_sky': use_sky,
            'best_mae': best_mae,
            'best_rmse': best_rmse,
            'best_ep': best_ep
        }
        checkpoint_path = os.path.join(save_dir_path, 'model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'CURRENT EPOCH {ep}: model saved to {checkpoint_path}.')
        if save_best:
            best_checkpoint_path = os.path.join(save_dir_path, 'model_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'BEST EPOCH {best_ep}: model saved to {best_checkpoint_path}.')

        """log training info"""
        log_time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        log_verbose = f'{log_time} -- epoch {str(ep).zfill(3)} batch size {str(bs).zfill(2)} lr {current_lr:.6f} | '\
                    f'train loss:{ave_train_loss:10.6f} val loss:{ave_val_loss:10.6f} '\
                    f'val MAE:{ave_val_mae:10.6f} val RMSE:{ave_val_rmse:10.6f} | '\
                    f'best epoch {str(best_ep).zfill(3)} best MAE:{best_mae:10.6f} best RMSE:{best_rmse:10.6f}\n'
        print(log_verbose)
        with open(os.path.join(save_dir_path, 'log_verbose.txt'), 'a') as f:
            f.write(log_verbose)

    """test"""
    print('training completed')
    print('testing...')
    test(save_dir_path, net_type, parallel)
    print('test over')
    print(100*'-')


def test(save_dir_path, net_type, parallel):
    model = DualResUNet(3, 3, 64).cuda()
    torch.compile(model)
    height_renderer = HeightRenderer(BS=1).cuda()
    pano_renderer = PanoRenderer(BS=1).cuda()
    
    checkpoint = torch.load(os.path.join(save_dir_path, 'model_best.pth'))
    if parallel:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    else:
        model.load_state_dict(checkpoint['net'])
        
    pred_save_path = os.path.join(save_dir_path, 'prediction')
    os.makedirs(pred_save_path, exist_ok=True)

    fns = []
    with open('./dataset/test.txt', 'r') as f:
        for i in f:
            fns.append(i.strip())
    
    aerial_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4485537493150323, 0.47377605990113797, 0.4520951120606348), 
                                std=(0.1799316263889618, 0.1567772635735416, 0.14737897874657677)),
    ])
    pano_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5530911549660469, 0.5883250463786659, 0.6191158691144336), 
                                std=(0.2013322671333761, 0.20358427316458785, 0.259951481179659)),
    ])
    metrics = []
    model.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(fns), total=len(fns))
        for step, fn in loop:
            aerial = cv2.imread(os.path.join('./dataset/aerial', fn+'_aerial.tif')).astype(np.float32)
            aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB) / 255.0
            
            pano = cv2.cvtColor(cv2.imread(os.path.join('./dataset/pano', fn+'.png')), cv2.COLOR_BGR2RGB)
            
            transformed_aerial = aerial_transforms(aerial).unsqueeze(0).cuda()
            transformed_pano = pano_transforms(pano).unsqueeze(0).cuda()

            density = model(transformed_aerial, transformed_pano)
            
            pred_pano, _ = pano_renderer(density)
            
            pred_pano = pred_pano.squeeze().cpu().numpy()
            prediction = height_renderer(density).squeeze().cpu().numpy()
            prediction[prediction < 0.1] = 0.
            cv2.imwrite(os.path.join(pred_save_path, fn+'_pred.tif'), prediction)
            mask = cv2.imread(os.path.join('./dataset/ndsm', fn+'_ndsm.tif'), cv2.IMREAD_UNCHANGED)
            metrics.append(compute_errors(mask, prediction))
    
    test_results = np.mean(np.array(metrics), axis=0)
    print(test_results)
    with open(os.path.join(save_dir_path, 'test_results.txt'), 'w') as f:
        f.write(f'MAE: {test_results[0]:10.6f}\n')
        f.write(f'RMSE:{test_results[1]:10.6f}\n')
        f.write(f'SSIM:{test_results[2]:10.6f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train density with street fusion & cons")

    parser.add_argument('-n', '--net_type', choices=['resunet'], default='resunet')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-e', '--epochs', type=int, help='epochs', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', required=True)
    parser.add_argument('--lr_decay', nargs=2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--restore_from', help='path to checkpoint. e.g. DENSITY_XXX/model_best.pth')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')

    parser.add_argument('--aerial_weight', type=float, default=1.0, help="weight of aerial loss")
    parser.add_argument('--street_weight', type=float, default=1.0, help="weight of street loss")
    parser.add_argument('--use_rank', action='store_true')
    parser.add_argument('--use_ssim', action='store_true')
    parser.add_argument('--use_sky', action='store_true')
    parser.add_argument('--use_grad', action='store_true')
    parser.add_argument('--test_path')
    args = parser.parse_args()
    
    test_path = args.test_path
    train(args) if test_path is None else test(save_dir_path=test_path, parallel=False)
    