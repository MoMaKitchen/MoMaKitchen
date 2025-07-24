import sys
sys.path.append('/cpfs01/shared/optimal/zhangpingrui/NavAffordance')    # change this to your path

import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import hydra
import pdb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from model.NaviAff import NaviAff
from dataset import PointAfforDataset
from utils.cal_metric import cal_metric
from utils.logger import logger
from datetime import datetime

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train(cfg: DictConfig):
    save_path = os.path.join(cfg["save_dir"], cfg["name"])

    device = torch.device(cfg["gpu"])
    tb_writer = SummaryWriter(save_path)
    
    train_dataset = PointAfforDataset('train', cfg.data)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], num_workers=12, shuffle=True, drop_last=True)

    eval_unseen_dataset = PointAfforDataset('test', cfg.data)
    eval_unseen_loader = DataLoader(eval_unseen_dataset, batch_size=cfg["batch_size"], num_workers=12, shuffle=False, drop_last=False)
    model = NaviAff(N_p=cfg["N_p"], emb_dim=cfg["emb_dim"], proj_dim=cfg["proj_dim"], num_heads=cfg["num_heads"], N_raw=cfg["N_raw"],
                    droprate=cfg["droprate"], attention_dim=cfg["attention_dim"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["decay_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["total_epoch"], eta_min=1e-6)
    
    start_epoch = 0
    total_steps = 0
    
    for epoch in range(start_epoch, cfg["total_epoch"]):
        logger.info(f"Epoch {epoch}/{cfg['total_epoch']}")
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        tb_writer.add_scalar("train/lr", learning_rate, total_steps)
        model.train()

        loss_sum = 0
        for iteration, batch in enumerate(train_loader):
            '''
            global_pc: [B, 4, 20000] 4: xyz+feat
            floor_pc: [B, 3, 10701]
            floor_mask: [B, 10701]
            floor_gt: [B, 10701]
            robot_h_w: [B, 2]
            '''
            global_pc, floor_pc, floor_mask, floot_gt, robot_h_w = batch
            optimizer.zero_grad()
            global_pc = global_pc.to(device)
            floor_pc = floor_pc.to(device)
            floor_mask = floor_mask.to(device)
            floot_gt = floot_gt.to(device)
            robot_h_w = robot_h_w.to(device)
            pred = model(global_pc, floor_pc, robot_h_w, floor_mask).squeeze(-1) # [B, N]

            pred_sele = pred[floor_mask==1]
            gt_sele = floot_gt[floor_mask==1]
            mask_weight = cfg["weight"]
            random_mask = (torch.rand(len(gt_sele)) < 0.5).to(gt_sele.device)

            weights = torch.where((gt_sele == 0) & random_mask, mask_weight, 1.0)

            pred_loss = (F.mse_loss(pred_sele, gt_sele, reduction='none') * weights).mean()

            loss = pred_loss

            if iteration%50 == 0:
                logger.info(f"Epoch:{epoch} | Iteration:{iteration} | loss:{loss.item()}")
                tb_writer.add_scalar("train/loss", loss.item(), total_steps)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            total_steps += 1

        mean_loss = loss_sum / len(train_loader)
        logger.info(f'Epoch:{epoch} | mean_loss:{mean_loss}')
        
        model_save_path = os.path.join(save_path, f"{epoch}.pt")
        torch.save(model, model_save_path)

        # eval
        model.eval()
        logger.info("Eval start")
        with torch.no_grad():
            pred_res = []
            gt_res = []
            test_loss_sum = 0
            # unseen
            logger.info("UnSeen eval")
            for iter, batch in enumerate(eval_unseen_loader):
                global_pc, floor_pc, floor_mask, floot_gt, robot_h_w = batch
                global_pc = global_pc.to(device)
                floor_pc = floor_pc.to(device)
                floor_mask = floor_mask.to(device)
                floot_gt = floot_gt.to(device)
                robot_h_w = robot_h_w.to(device)

                pred = model(global_pc, floor_pc, robot_h_w, floor_mask) # [B, ]

                pred_sele = pred[floor_mask==1].squeeze()
                gt_sele = floot_gt[floor_mask==1]

                loss = F.mse_loss(pred_sele, gt_sele)
                test_loss_sum += loss.item()

                pred_res.append(pred_sele.cpu().numpy())
                gt_res.append(gt_sele.cpu().numpy())
            
            test_mean_loss = test_loss_sum / len(eval_unseen_loader)
            
            metrics = cal_metric(pred_res, gt_res)

            logger.info(f"Epoch:{epoch} | eval_seen_mean_loss:{test_mean_loss} | metrics:{metrics}")
            tb_writer.add_scalar("eval_seen/loss", test_mean_loss, total_steps)
            tb_writer.add_scalars("eval_seen/metrics", metrics, total_steps)

        scheduler.step()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(seed=42)
    # config_path = "conf/config.yaml"
    # cfg = OmegaConf.load(config_path)
    # train(cfg)
    train()