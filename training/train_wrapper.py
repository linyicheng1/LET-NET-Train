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
                 w_pk: float = 1.0, w_rp: float = 1.0, w_sp: float = 1.0, w_ds: float = 1.0,
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
            self.pk_loss_old = PeakyLoss_old(scores_th=sc_th)
            self.pk_loss = PeakyLoss(scores_th=sc_th, radius=self.radius)
        if self.w_rp > 0:
            self.rp_loss = ReprojectionLocLoss(norm=norm, scores_th=sc_th, train_gt_th=self.train_gt_th)
            self.rp_loss_old = ReprojectionLocLoss_old(norm=norm, scores_th=sc_th)
        if self.w_sp > 0:
            self.ScoreMapRepLoss = ScoreMapRepLoss(temperature=temp_sp)
            self.ScoreMapRepLoss_old = ScoreMapRepLoss_old(temperature=temp_sp)
        if self.w_ds > 0:
            self.DescReprojectionLoss = DescReprojectionLoss(temperature=temp_ds)
            self.DescReprojectionLoss_old = DescReprojectionLoss_old(temperature=temp_ds)

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

        self.log('train/loss', loss)
        for k, v in loss_package.items():
            if 'loss' in k:
                self.log('train/' + k, v)

        pred = {'scores_map0': pred0['scores_map'],
                'scores_map1': pred1['scores_map'],
                'kpts0': [], 'kpts1': [],
                'desc0': [], 'desc1': []}

        for idx in range(b):
            pred['kpts0'].append(
                (kps0[idx][:num_det0] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['kpts1'].append(
                (kps1[idx][:num_det1] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['desc0'].append(desc0[:num_det0])
            pred['desc1'].append(desc1[:num_det1])

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

            # =================== matches
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()
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

    def evaluate(self, pred, batch):
        b = len(pred['kpts0'])

        accuracy = []
        for idx in range(b):
            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()

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