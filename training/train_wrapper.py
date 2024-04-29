import os
import logging
import functools
from typing import Optional

import cv2
import numpy as np
from pathlib import Path
from nets.soft_detect import SoftDetect
import torch

from nets.letnet import LETNetTrain
from nets.loss import *

from training.scheduler import WarmupConstantSchedule
from utils import keypoints_normal2pixel, mutual_argmin, \
    mutual_argmax, plot_keypoints, plot_matches, EmptyTensorError, \
    warp, compute_keypoints_distance
from training.val_hpatches_utils import load_precompute_errors, draw_MMA


class TrainWrapper(LETNetTrain):
    def __init__(self,
                 # model params
                 c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 64, gray: bool = False,
                 # detect params
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5, n_limit: int = 5000,  # in training
                 scores_th_eval: float = 0.2, n_limit_eval: int = 5000,  # in evaluation
                 # gt projection params
                 train_gt_th: int = 5, eval_gt_th: int = 3,
                 # loss weights
                 w_pk: float = 1.0, w_rp: float = 1.0, w_sp: float = 1.0, w_ds: float = 1.0, w_mds: float = 1.0,
                 sc_th: float = 0.1, norm: int = 1, temp_sp: float = 0.1, temp_ds: float = 0.02,
                 # training params
                 lr: float = 1e-3, log_freq_img: int = 1000,
                 pretrained_model: Optional[str] = None,
                 lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=10000),
                 debug: bool = False,
                 ):
        super().__init__(c1, c2, c3, c4, dim, gray)
        self.save_hyperparameters()

        self.lr = lr

        self.radius = radius
        self.w_pk = w_pk
        self.w_rp = w_rp
        self.w_sp = w_sp
        self.w_ds = w_ds
        self.w_mds = w_mds

        self.train_gt_th = train_gt_th
        self.eval_gt_th = eval_gt_th

        self.scores_th_eval = scores_th_eval
        self.n_limit_eval = n_limit_eval

        self.log_freq_img = log_freq_img

        self.pretrained_model = pretrained_model
        self.lr_scheduler = lr_scheduler
        self.debug = debug
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit

        self.softdetect = SoftDetect(radius=self.radius, top_k=top_k,
                                     scores_th=scores_th, n_limit=n_limit)

        #  ================ load pretrained model ================
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                logging.info(f'Load pretrained model from {pretrained_model}')
                if pretrained_model.endswith('ckpt'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))['state_dict']
                elif pretrained_model.endswith('pt'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))
                else:
                    state_dict = None
                    logging.error(f"Error model file: {pretrained_model}")
                self.load_state_dict(state_dict, strict=False)
            else:
                logging.error(f"Can not find pretrained model: {pretrained_model}")

        # ================ losses ================

        if self.w_pk > 0:
            # self.pk_loss = PeakyLoss(scores_th=sc_th, radius=self.radius)
            self.pk_loss = LinePeakyLoss(scores_th=sc_th, radius=self.radius)
        if self.w_rp > 0:
            self.rp_loss = ReprojectionLocLoss(norm=norm, scores_th=sc_th, train_gt_th=self.train_gt_th)
        if self.w_sp > 0:
            self.ScoreMapRepLoss = ScoreMapRepLoss(temperature=temp_sp)
        if self.w_ds > 0:
            self.DescReprojectionLoss = DescReprojectionLoss(temperature=temp_ds)
        if self.w_mds > 0:
            self.LocalDescLoss = LocalDescLoss(temperature=temp_ds, window_size=80)

        # ================ evaluation ================
        lim = [1, 15]
        self.rng = np.arange(lim[0], lim[1] + 1)
        self.i_err = {thr: 0 for thr in self.rng}
        self.v_err = {thr: 0 for thr in self.rng}
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        self.errors = load_precompute_errors(str(Path(__file__).parent / 'errors.pkl'))

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"train_acc": 0,
                                                   'val_kpt_num': 0,
                                                   'val_repeatability': 0,
                                                   'val_acc': 0,
                                                   'val_matching_score': 0,
                                                   'val_kpt_num_mean': 0,
                                                   'val_repeatability_mean': 0,
                                                   'val_acc/mean': 0,
                                                   'val_matching_score/mean': 0,
                                                   'val_metrics/mean': 0,
                                                   "val_mma_mean": 0, })

    def training_step(self, batch, batch_idx):
        b, c, h, w = batch[0]['image0'].shape

        pred0 = super().extract_dense_map(batch[0]['image0'], True)
        pred1 = super().extract_dense_map(batch[0]['image1'], True)

        # =================== detect keypoints ===================
        kps0, scores0 = self.softdetect.detect_keypoints(pred0['scores_map'])
        kps1, scores1 = self.softdetect.detect_keypoints(pred1['scores_map'])
        num_det0, num_det1 = len(kps0[0]), len(kps1[0])

        # add random points
        kps0, scores0, num_det0 = self.add_random_kps(kps0, pred0['scores_map'])
        kps1, scores1, num_det1 = self.add_random_kps(kps1, pred1['scores_map'])

        # warp parameters
        warp01_params = {}
        for k, v in batch[0]['warp01_params'].items():
            warp01_params[k] = v[0]
        warp10_params = {}
        for k, v in batch[0]['warp10_params'].items():
            warp10_params[k] = v[0]

        # =================== compute descriptors ===================
        desc0 = torch.nn.functional.grid_sample(pred0['descriptor_map'][0].unsqueeze(0), kps0[0].view(1, 1, -1, 2),
                                                mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc1 = torch.nn.functional.grid_sample(pred1['descriptor_map'][0].unsqueeze(0), kps1[0].view(1, 1, -1, 2),
                                                mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
        desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)

        similarity_map_01 = torch.einsum('nd,dhw->nhw', desc0, pred1['descriptor_map'][0])
        similarity_map_10 = torch.einsum('nd,dhw->nhw', desc1, pred0['descriptor_map'][0])

        l_desc0 = torch.nn.functional.grid_sample(pred0['local_desc'][0].unsqueeze(0), kps0[0].view(1, 1, -1, 2),
                                                  mode='bilinear', align_corners=True)[0, :, 0, :].t()
        l_desc1 = torch.nn.functional.grid_sample(pred1['local_desc'][0].unsqueeze(0), kps1[0].view(1, 1, -1, 2),
                                                  mode='bilinear', align_corners=True)[0, :, 0, :].t()
        l_desc0 = torch.nn.functional.normalize(l_desc0, p=2, dim=1)
        l_desc1 = torch.nn.functional.normalize(l_desc1, p=2, dim=1)

        m_similarity_map_01 = torch.einsum('nd,dhw->nhw', l_desc0, pred1['local_desc'][0])
        m_similarity_map_10 = torch.einsum('nd,dhw->nhw', l_desc1, pred0['local_desc'][0])

        # ================ compute loss ================

        loss = 0
        loss_package = {}

        if self.w_pk > 0:
            loss_peaky0 = self.pk_loss(kps0, scores0, pred0['scores_map'])
            loss_peaky1 = self.pk_loss(kps1, scores1, pred1['scores_map'])

            loss_peaky = (loss_peaky0 + loss_peaky1) / 2.

            loss += self.w_pk * loss_peaky
            loss_package['loss_peaky'] = loss_peaky

        if self.w_rp > 0:
            loss_reprojection = self.rp_loss(kps0, scores0, pred0['scores_map'],
                                             kps1, scores1, pred1['scores_map'],
                                             warp01_params, warp10_params)

            loss += self.w_rp * loss_reprojection
            loss_package['loss_reprojection'] = loss_reprojection

        if self.w_sp > 0:
            loss_score_map_rp = self.ScoreMapRepLoss(kps0, scores0, pred0['scores_map'], similarity_map_01,
                                                     kps1, scores1, pred1['scores_map'], similarity_map_10,
                                                     warp01_params, warp10_params)
            loss += self.w_sp * loss_score_map_rp
            loss_package['loss_score_map_rp'] = loss_score_map_rp

        if self.w_ds > 0:
            loss_des = self.DescReprojectionLoss(kps0, pred0['scores_map'], similarity_map_01,
                                                 kps1, pred1['scores_map'], similarity_map_10,
                                                 warp01_params, warp10_params)
            loss += self.w_ds * loss_des
            loss_package['loss_des'] = loss_des

        if self.w_mds > 0:
            loss_mdes = self.LocalDescLoss(kps0, pred0['scores_map'], m_similarity_map_01,
                                           kps1, pred1['scores_map'], m_similarity_map_10,
                                           warp01_params, warp10_params)
            loss += self.w_mds * loss_mdes
            loss_package['loss_mdes'] = loss_mdes

        self.log('train/loss', loss)
        for k, v in loss_package.items():
            if 'loss' in k:
                self.log('train/' + k, v)

        pred = {'scores_map0': pred0['scores_map'],
                'scores_map1': pred1['scores_map'],
                'kpts0': [], 'kpts1': [],
                'desc0': [], 'desc1': [],
                'local_desc': []}

        for idx in range(b):
            pred['kpts0'].append(
                (kps0[idx][:num_det0] + 1) / 2 * kps0[idx].new_tensor([[w - 1, h - 1]]))
            pred['kpts1'].append(
                (kps1[idx][:num_det1] + 1) / 2 * kps0[idx].new_tensor([[w - 1, h - 1]]))
            pred['desc0'].append(desc0[:num_det0])
            pred['desc1'].append(desc1[:num_det1])
            pred['local_desc'] = [pred0['local_desc'], pred1['local_desc']]

        accuracy = self.evaluate(pred, batch[0])
        self.log('train_acc', accuracy, prog_bar=True)

        if batch_idx % self.log_freq_img == 0:
            self.log_image_and_score(batch[0], pred, 'train_')

        assert not torch.isnan(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = self.lr_scheduler(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'scheduled_lr'}]

    def log_image_and_score(self, batch, pred, suffix):
        b, c, h, w = pred['scores_map0'].shape

        for idx in range(b):
            if idx > 1:
                break

            image0 = (batch['image0'][idx] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
            image1 = (batch['image1'][idx] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
            scores0 = (pred['scores_map0'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            scores1 = (pred['scores_map1'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            l_desc0 = (pred['local_desc'][0][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            l_desc1 = (pred['local_desc'][1][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()

            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()

            # =================== score map
            s = cv2.applyColorMap(scores0, cv2.COLORMAP_JET)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            self.logger.experiment.add_image(f'{suffix}score/{idx}_src', torch.tensor(s),
                                             global_step=self.global_step, dataformats='HWC')

            s = cv2.applyColorMap(scores1, cv2.COLORMAP_JET)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            self.logger.experiment.add_image(f'{suffix}score/{idx}_tgt', torch.tensor(s),
                                             global_step=self.global_step, dataformats='HWC')

            # =================== image with keypoints
            image0_kpts = plot_keypoints(image0, kpts0[:, [1, 0]], radius=1)
            image1_kpts = plot_keypoints(image1, kpts1[:, [1, 0]], radius=1)

            self.logger.experiment.add_image(f'{suffix}image/{idx}_src', torch.tensor(image0_kpts),
                                             global_step=self.global_step, dataformats='HWC')
            self.logger.experiment.add_image(f'{suffix}image/{idx}_tgt', torch.tensor(image1_kpts),
                                             global_step=self.global_step, dataformats='HWC')
            # =================== local descriptors
            self.logger.experiment.add_image(f'{suffix}local_desc/{idx}_src', torch.tensor(l_desc0),
                                             global_step=self.global_step, dataformats='CHW')
            self.logger.experiment.add_image(f'{suffix}local_desc/{idx}_tgt', torch.tensor(l_desc1),
                                             global_step=self.global_step, dataformats='CHW')
            # =================== matches
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()
            if desc0.shape[0] == 0 or desc1.shape[0] == 0:
                continue
            matches_est = mutual_argmax(desc0 @ desc1.t())
            mkpts0, mkpts1 = kpts0[matches_est[0]][:, [1, 0]], kpts1[matches_est[1]][:, [1, 0]]

            match_image = plot_matches(image0, image1, mkpts0, mkpts1)
            self.logger.experiment.add_image(f'{suffix}matches/{idx}', torch.tensor(match_image),
                                             global_step=self.global_step, dataformats='HWC')


    def add_random_kps(self, kps, score_map):
        b, c, h, w = score_map.shape
        wh = score_map[0].new_tensor([[w - 1, h - 1]])
        num = len(kps[0])
        # add random points
        rand = torch.rand(len(kps[0]), 2, device=kps[0].device) * 2 - 1  # -1~1
        kps_add = torch.cat([kps[0], rand])
        scores_kps = torch.nn.functional.grid_sample(score_map, kps_add.view(1, 1, -1, 2),
                                                     mode='bilinear', align_corners=True).squeeze()
        # nms for random points
        kps_wh_ = (kps_add / 2 + 0.5) * wh  # N0x2, (w,h)
        dist = compute_keypoints_distance(kps_wh_.detach(), kps_wh_.detach())
        local_mask = dist < self.radius
        valid_cnt = torch.sum(local_mask, dim=1)
        indices_need_nms = torch.where(valid_cnt > 1)[0]
        for i in indices_need_nms:
            if valid_cnt[i] > 0:
                kpt_indices = torch.where(local_mask[i])[0]
                scs_max_idx = scores_kps[kpt_indices].argmax()

                tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                tmp_mask[scs_max_idx] = False
                suppressed_indices = kpt_indices[tmp_mask]

                valid_cnt[suppressed_indices] = 0
        valid_mask = valid_cnt > 0
        kps_add = kps_add[valid_mask]
        scores_kps = scores_kps[valid_mask]
        valid_mask = valid_mask[:num]
        return [kps_add], [scores_kps], valid_mask.sum()

    def backward(self, loss, *args, **kwargs):
        if loss.requires_grad:
            loss.backward()
        else:
            logging.debug('loss is not backward')

    def on_before_optimizer_step(self, optimizer):
        if torch.isnan(self.let_net.block1.conv1.weight.grad).any():
            optimizer.zero_grad()
            logging.log(logging.ERROR, 'nan in grad')

    def compute_correspondence(self, pred0, pred1, batch, rand=False):
        # image size
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['scores_map'][0].new_tensor([[w - 1, h - 1]])

        if self.debug:
            from utils import display_image_in_actual_size
            image0 = batch[0]['image0'][0].permute(1, 2, 0).cpu().numpy()
            image1 = batch[0]['image1'][0].permute(1, 2, 0).cpu().numpy()
            display_image_in_actual_size(image0)
            display_image_in_actual_size(image1)

        pred0_with_rand = pred0
        pred1_with_rand = pred1
        pred0_with_rand['scores'] = []
        pred1_with_rand['scores'] = []
        pred0_with_rand['descriptors'] = []
        pred1_with_rand['descriptors'] = []
        pred0_with_rand['local_descriptors'] = []
        pred1_with_rand['local_descriptors'] = []
        pred0_with_rand['num_det'] = []
        pred1_with_rand['num_det'] = []
        # 1. detect keypoints of image0 and image1
        kps, score_dispersity, scores = self.softdetect.detect_keypoints_old(pred0['scores_map'])
        pred0_with_rand['keypoints'] = kps
        pred0_with_rand['score_dispersity'] = score_dispersity
        # 2. detect keypoints
        kps, score_dispersity, scores = self.softdetect.detect_keypoints_old(pred1['scores_map'])
        pred1_with_rand['keypoints'] = kps
        pred1_with_rand['score_dispersity'] = score_dispersity

        correspondences = []
        for idx in range(b):
            # =========================== prepare keypoints
            kpts0, kpts1 = pred0['keypoints'][idx], pred1['keypoints'][idx]  # (x,y), shape: Nx2

            # additional random keypoints
            if rand:
                rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1  # -1~1
                rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1  # -1~1
                kpts0 = torch.cat([kpts0, rand0])
                kpts1 = torch.cat([kpts1, rand1])

                pred0_with_rand['keypoints'][idx] = kpts0
                pred1_with_rand['keypoints'][idx] = kpts1

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]

            scores_kpts0 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()
            scores_kpts1 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()

            kpts0_wh_ = (kpts0 / 2 + 0.5) * wh  # N0x2, (w,h)
            kpts1_wh_ = (kpts1 / 2 + 0.5) * wh  # N1x2, (w,h)

            # ========================= nms
            dist = compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts0[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts0_wh = kpts0_wh_[valid_mask]
            kpts0 = kpts0[valid_mask]
            scores_kpts0 = scores_kpts0[valid_mask]
            pred0_with_rand['keypoints'][idx] = kpts0

            valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
            pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask]
            pred0_with_rand['num_det'].append(valid_mask.sum())

            dist = compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts1[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts1_wh = kpts1_wh_[valid_mask]
            kpts1 = kpts1[valid_mask]
            scores_kpts1 = scores_kpts1[valid_mask]
            pred1_with_rand['keypoints'][idx] = kpts1

            valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
            pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask]
            pred1_with_rand['num_det'].append(valid_mask.sum())

            # del dist, local_mask, valid_cnt, indices_need_nms, scs_max_idx, tmp_mask, suppressed_indices, valid_mask
            # torch.cuda.empty_cache()
            # ========================= nms

            pred0_with_rand['scores'].append(scores_kpts0)
            pred1_with_rand['scores'].append(scores_kpts1)

            descriptor_map0, descriptor_map1 = pred0['descriptor_map'][idx], pred1['descriptor_map'][idx]
            local_descriptor_map0, local_descriptor_map1 = pred0['local_desc'][idx], pred1['local_desc'][idx]

            desc0 = torch.nn.functional.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            desc1 = torch.nn.functional.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            local_desc0 = torch.nn.functional.grid_sample(local_descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                          mode='bilinear', align_corners=True)[0, :, 0, :].t()
            local_desc1 = torch.nn.functional.grid_sample(local_descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                          mode='bilinear', align_corners=True)[0, :, 0, :].t()

            desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)
            local_desc0 = torch.nn.functional.normalize(local_desc0, p=2, dim=1)
            local_desc1 = torch.nn.functional.normalize(local_desc1, p=2, dim=1)

            pred0_with_rand['descriptors'].append(desc0)
            pred1_with_rand['descriptors'].append(desc1)
            pred0_with_rand['local_descriptors'].append(local_desc0)
            pred1_with_rand['local_descriptors'].append(local_desc1)

            # =========================== prepare warp parameters
            warp01_params = {}
            for k, v in batch[0]['warp01_params'].items():
                warp01_params[k] = v[idx]
            warp10_params = {}
            for k, v in batch[0]['warp10_params'].items():
                warp10_params[k] = v[idx]

            # =========================== warp keypoints across images
            try:
                # valid keypoint, valid warped keypoint, valid indices
                kpts0_wh, kpts01_wh, ids0, ids0_out = warp(kpts0_wh, warp01_params)
                kpts1_wh, kpts10_wh, ids1, ids1_out = warp(kpts1_wh, warp10_params)
                if len(kpts0_wh) == 0 or len(kpts1_wh) == 0 or len(kpts0) == 0 or len(kpts1) == 0:
                    raise EmptyTensorError
            except EmptyTensorError:
                correspondences.append({'correspondence0': None, 'correspondence1': None,
                                        'dist': kpts0_wh.new_tensor(0),
                                        })
                continue

            if self.debug:
                from utils import display_image_in_actual_size
                image0 = batch[0]['image0'][0].permute(1, 2, 0).cpu().numpy()
                image1 = batch[0]['image1'][0].permute(1, 2, 0).cpu().numpy()

                p0 = kpts0_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts0 = plot_keypoints(image0, p0, radius=1, color=(255, 0, 0))
                # display_image_in_actual_size(img_kpts0)

                p1 = kpts1_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts1 = plot_keypoints(image1, p1, radius=1, color=(255, 0, 0))
                # display_image_in_actual_size(img_kpts1)

                p01 = kpts01_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts01 = plot_keypoints(img_kpts1, p01, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts01)

                p10 = kpts10_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts10 = plot_keypoints(img_kpts0, p10, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts10)

            # ============================= compute reprojection error
            dist01 = compute_keypoints_distance(kpts0_wh, kpts10_wh)
            dist10 = compute_keypoints_distance(kpts1_wh, kpts01_wh)

            dist_l2 = (dist01 + dist10.t()) / 2.
            # find mutual correspondences by calculating the distance
            # between keypoints (I1) and warpped keypoints (I2->I1)
            mutual_min_indices = mutual_argmin(dist_l2)

            dist_mutual_min = dist_l2[mutual_min_indices]
            valid_dist_mutual_min = dist_mutual_min.detach() < self.train_gt_th

            ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
            ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

            correspondence0 = ids0[ids0_d]
            correspondence1 = ids1[ids1_d]

            # L1 distance
            dist01_l1 = compute_keypoints_distance(kpts0_wh, kpts10_wh, p=1)
            dist10_l1 = compute_keypoints_distance(kpts1_wh, kpts01_wh, p=1)

            dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.

            # =========================== compute cross image descriptor similarity_map
            similarity_map_01 = torch.einsum('nd,dhw->nhw', desc0, descriptor_map1)
            similarity_map_10 = torch.einsum('nd,dhw->nhw', desc1, descriptor_map0)
            local_similarity_map_01 = torch.einsum('nd,dhw->nhw', local_desc0, local_descriptor_map1)
            local_similarity_map_10 = torch.einsum('nd,dhw->nhw', local_desc1, local_descriptor_map0)

            similarity_map_01_valid = similarity_map_01[ids0]  # valid descriptors
            similarity_map_10_valid = similarity_map_10[ids1]

            local_similarity_map_01_valid = local_similarity_map_01[ids0]  # valid descriptors
            local_similarity_map_10_valid = local_similarity_map_10[ids1]

            kpts01 = 2 * kpts01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kpts10 = 2 * kpts10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

            correspondences.append({'correspondence0': correspondence0,  # indices of matched kpts0 in all kpts
                                    'correspondence1': correspondence1,  # indices of matched kpts1 in all kpts
                                    'scores0': scores_kpts0[ids0],
                                    'scores1': scores_kpts1[ids1],
                                    'kpts01': kpts01, 'kpts10': kpts10,  # warped valid kpts
                                    'ids0': ids0, 'ids1': ids1,  # valid indices of kpts0 and kpts1
                                    'ids0_out': ids0_out, 'ids1_out': ids1_out,
                                    'ids0_d': ids0_d, 'ids1_d': ids1_d,  # match indices of valid kpts0 and kpts1
                                    'dist_l1': dist_l1,  # cross distance matrix of valid kpts using L1 norm
                                    'dist': dist_l2,  # cross distance matrix of valid kpts using L2 norm
                                    'similarity_map_01': similarity_map_01,  # all
                                    'similarity_map_10': similarity_map_10,  # all
                                    'similarity_map_01_valid': similarity_map_01_valid,  # valid
                                    'similarity_map_10_valid': similarity_map_10_valid,  # valid
                                    'local_similarity_map_01': local_similarity_map_01,  # all
                                    'local_similarity_map_10': local_similarity_map_10,  # all
                                    'local_similarity_map_01_valid': local_similarity_map_01_valid,  # valid
                                    'local_similarity_map_10_valid': local_similarity_map_10_valid,  # valid
                                    })

        return correspondences, pred0_with_rand, pred1_with_rand

    def evaluate(self, pred, batch):
        b = len(pred['kpts0'])

        accuracy = []
        for idx in range(b):
            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()
            if len(kpts0) == 0 or len(kpts1) == 0:
                return pred['kpts0'][0].new_tensor(0)
            matches_est = mutual_argmax(desc0 @ desc1.t())

            mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]

            # warp
            warp01_params = {}
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[idx]

            try:
                mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            except EmptyTensorError:
                continue

            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1))
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            correct = dist < self.eval_gt_th
            accuracy.append(correct.float().mean())

        accuracy = torch.stack(accuracy).mean() if len(accuracy) != 0 else pred['kpts0'][0].new_tensor(0)
        return accuracy

    def on_validation_epoch_start(self):
        # reset
        for thr in self.rng:
            self.i_err[thr] = 0
            self.v_err[thr] = 0
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_validation_epoch_end(self):
        # ============= compute average
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        accuracy_mean = np.mean(np.array(self.accuracy))
        matching_score_mean = np.mean(np.array(self.matching_score))

        self.log('val_kpt_num_mean', num_feat_mean)
        self.log('val_repeatability_mean', repeatability_mean)
        self.log('val_acc_mean', accuracy_mean)
        self.log('val_matching_score_mean', matching_score_mean)
        self.log('val_metrics/mean',
                 (repeatability_mean + accuracy_mean + matching_score_mean) / 3)

        # ============= compute and draw MMA
        self.errors['ours'] = (self.i_err, self.v_err, 0)
        n_i = 52
        n_v = 56
        MMA = 0
        for i in range(10):
            MMA += (self.i_err[i + 1] + self.v_err[i + 1]) / ((n_i + n_v) * 5)
        MMA = MMA / 10
        # MMA3 = (self.i_err[self.eval_gt_th] + self.v_err[self.eval_gt_th]) / ((n_i + n_v) * 5)
        self.log('val_mma_mean', MMA)

        MMA_image = draw_MMA(self.errors)

        self.logger.experiment.add_image(f'hpatches_MMA', torch.tensor(MMA_image),
                                         global_step=self.global_step, dataformats='HWC')

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dist, num_feat, repeatability, accuracy, matching_score, recall = self.val_match(batch)

        self.log('val_kpt_num', num_feat)
        self.log('val_repeatability', repeatability)
        self.log('val_acc', accuracy)
        self.log('val_matching_score', matching_score)
        self.log('val_recall', recall)

        self.num_feat.append(num_feat)
        self.repeatability.append(repeatability)
        self.accuracy.append(accuracy)
        self.matching_score.append(matching_score)

        # compute the MMA
        dist = dist.cpu().detach().numpy()
        if dataloader_idx == 0:
            for thr in self.rng:
                self.i_err[thr] += np.mean(dist <= thr)
        elif dataloader_idx == 1:
            for thr in self.rng:
                self.v_err[thr] += np.mean(dist <= thr)
        else:
            pass

        return {'num_feat': num_feat, 'repeatability': repeatability, 'accuracy': accuracy,
                'matching_score': matching_score}

    def val_match(self, batch):
        b, _, h0, w0 = batch['image0'].shape
        _, _, h1, w1 = batch['image1'].shape
        assert b == 1

        # ==================================== extract keypoints and descriptors
        detector = SoftDetect(radius=self.radius, top_k=self.top_k,
                              scores_th=self.scores_th, n_limit=self.n_limit_eval)

        desc_map_0, score_map_0, local_desc_0 = super().extract_dense_map(batch['image0'])
        kps_0, desc0, kps_scores_0, score_disp_0 = detector(score_map_0, desc_map_0)
        desc_map_1, score_map_1, local_desc_1 = super().extract_dense_map(batch['image1'])
        kps_1, desc1, kps_scores_1, score_disp_1 = detector(score_map_1, desc_map_1)

        kps_0 = keypoints_normal2pixel(kps_0, w0, h0)[0]
        kps_1 = keypoints_normal2pixel(kps_1, w1, h1)[0]
        desc0 = desc0[0]
        desc1 = desc1[0]

        num_feat = min(kps_0.shape[0], kps_1.shape[0])  # number of detected keypoints

        # ==================================== pack warp params
        warp01_params, warp10_params = {}, {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]

        try:
            # ==================================== covisible keypoints
            kpts0_cov, kpts01_cov, _, _ = warp(kps_0, warp01_params)
            kpts1_cov, kpts10_cov, _, _ = warp(kps_1, warp10_params)
            if len(kpts0_cov) == 0 or len(kpts1_cov) == 0:
                return (torch.tensor([float('inf')]),
                        num_feat,  # feature number
                        0,  # repeatability
                        0,  # accuracy
                        0,  # matching score
                        0,  # recall
                        )
            num_cov_feat = (len(kpts0_cov) + len(kpts1_cov)) / 2  # number of covisible keypoints

            # ==================================== get gt matching keypoints
            dist01 = compute_keypoints_distance(kpts0_cov, kpts10_cov)
            dist10 = compute_keypoints_distance(kpts1_cov, kpts01_cov)

            dist_mutual = (dist01 + dist10.t()) / 2.
            imutual = torch.arange(min(dist_mutual.shape), device=dist_mutual.device)
            dist_mutual[imutual, imutual] = 99999  # mask out diagonal

            mutual_min_indices = mutual_argmin(dist_mutual)
            dist = dist_mutual[mutual_min_indices]
            gt_num = (dist <= self.eval_gt_th).sum().cpu()  # number of gt matching keypoints

            # ==================================== putative matches
            matches_est = mutual_argmax(desc0 @ desc1.t())
            mkpts0, mkpts1 = kps_0[matches_est[0]], kps_1[matches_est[1]]

            num_putative = len(mkpts0)  # number of putative matches

            # ==================================== warp putative matches
            mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1)).cpu()
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            num_inlier = sum(dist <= self.eval_gt_th)

            return (dist,
                    num_feat,  # feature number
                    gt_num / max(num_cov_feat, 1),  # repeatability
                    num_inlier / max(num_putative, 1),  # accuracy
                    num_inlier / max(num_cov_feat, 1),  # matching score
                    num_inlier / max(gt_num, 1),  # recall
                    )
        except EmptyTensorError:
            return torch.tensor([[0]]), num_feat, 0, 0, 0, 0