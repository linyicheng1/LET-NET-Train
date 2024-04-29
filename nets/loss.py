import torch
from utils import keypoints_normal2pixel, mutual_argmin, \
    mutual_argmax, plot_keypoints, plot_matches, EmptyTensorError, \
    warp, compute_keypoints_distance


class PeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1, radius: int = 2):
        super().__init__()
        self.scores_th = scores_th
        self.radius = radius
        self.temperature = 0.1
        # local xy grid
        self.kernel_size = 2 * self.radius + 1
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)

    def __call__(self, kps, scores, score_map):
        b, c, h, w = score_map.shape
        loss_mean = 0
        CNT = 0
        score_dispersitys = []

        self.hw_grid = self.hw_grid.to(score_map)  # to device
        patches = self.unfold(score_map)  # B x (kernel**2) x (H*W)

        for idx in range(b):
            xy_int = kps[idx].int()  # M x 2
            xy_residual = kps[idx] - xy_int  # M x 2
            hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                       dim=-1) ** 2

            indices_kpt = xy_int[:, 1] * w + xy_int[:, 0]  # M
            patch = patches[idx].t()  # (H*W) x (kernel**2)
            patch_scores = patch[indices_kpt]  # M x (kernel**2)
            # max is detached to prevent undesired backprop loops in the graph
            max_v = patch_scores.max(dim=1).values.detach()[:, None]
            x_exp = ((patch_scores - max_v) / self.temperature).exp()  # M * (kernel**2), in [0, 1]
            # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }

            dispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)
            score_dispersitys.append(dispersity)

        n_original = len(score_dispersitys[0])
        for idx in range(b):
            scores_kpts = scores[idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = score_dispersitys[idx][valid]
            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else score_map.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class LinePeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1, radius: int = 2):
        super().__init__()
        self.scores_th = scores_th
        self.radius = radius
        self.temperature = 0.1
        # local xy grid
        self.kernel_size = 2 * self.radius + 1
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)

    def __call__(self, kps, scores, score_map):
        b, c, h, w = score_map.shape
        loss_mean = 0
        CNT = 0
        score_dispersitys = []

        self.hw_grid = self.hw_grid.to(score_map)  # to device
        patches = self.unfold(score_map)  # B x (kernel**2) x (H*W)

        for idx in range(b):
            M = len(kps[idx])
            K = self.kernel_size
            K2 = K ** 2
            xy_int = kps[idx].int()  # M x 2
            xy_residual = kps[idx] - xy_int  # M x 2
            hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                       dim=-1) ** 2
            a = torch.linspace(-self.radius, self.radius, self.kernel_size).to(score_map.device)
            mask_x = torch.exp(-(a.view(1, K) - xy_residual[:, 0].view(M, 1)) * (a - xy_residual[:, 0].view(M, 1)))
            mask_x = torch.stack((mask_x, mask_x, mask_x, mask_x, mask_x), 1).view(M, K2)
            mask_y = torch.exp(-(a.view(1, K) - xy_residual[:, 1].view(M, 1)) * (a - xy_residual[:, 1].view(M, 1)))
            mask_y = torch.stack((mask_y, mask_y, mask_y, mask_y, mask_y), 2).view(M, K2)

            a = (self.hw_grid[None, :, :] - xy_residual[:, None, :])
            mask_xy = torch.exp(-torch.abs(a.sum(dim=2) * 0.525))
            mask_yx = torch.exp(-torch.abs((a[:, :, 0] - a[:, :, 1]) * 0.525))

            indices_kpt = xy_int[:, 1] * w + xy_int[:, 0]  # M
            patch = patches[idx].t()  # (H*W) x (kernel**2)
            patch_scores = patch[indices_kpt]  # M x (kernel**2)
            # max is detached to prevent undesired backprop loops in the graph
            max_v = patch_scores.max(dim=1).values.detach()[:, None]
            x_exp = ((patch_scores - max_v) / self.temperature).exp()  # M * (kernel**2), in [0, 1]

            # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
            scoredispersity_x = (mask_x * x_exp * hw_grid_dist2).sum(dim=1) / (mask_x*x_exp).sum(dim=1)
            scoredispersity_y = (mask_y * x_exp * hw_grid_dist2).sum(dim=1) / (mask_y*x_exp).sum(dim=1)
            scoredispersity_xy = (mask_xy * x_exp * hw_grid_dist2).sum(dim=1) / (mask_xy*x_exp).sum(dim=1)
            scoredispersity_yx = (mask_yx * x_exp * hw_grid_dist2).sum(dim=1) / (mask_yx*x_exp).sum(dim=1)

            dispersity = torch.stack([scoredispersity_x, scoredispersity_y, scoredispersity_xy, scoredispersity_yx], 1)
            dispersity, i = dispersity.max(dim=1)
            score_dispersitys.append(dispersity)

        n_original = len(score_dispersitys[0])
        for idx in range(b):
            scores_kpts = scores[idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = score_dispersitys[idx][valid]
            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else score_map.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class ReprojectionLocLoss(object):
    """
    Reprojection location errors of keypoints to train repeatable detector.
    """
    def __init__(self, norm: int = 1, scores_th: float = 0.1, train_gt_th: float = 2):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th
        self.train_gt_th = train_gt_th

    def __call__(self, kps0, scores0, score_map0, kps1, scores1, score_map1, warp01_params, warp10_params):
        b, c, h, w = score_map0.shape
        wh = score_map0.new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        # compute correspondence
        kps0_wh_ = (kps0[0] / 2 + 0.5) * wh  # N0x2, (w,h)
        kps1_wh_ = (kps1[0] / 2 + 0.5) * wh  # N1x2, (w,h)

        try:
            # valid keypoint, valid warped keypoint, valid indices
            kps0_wh_, kps01_wh, ids0, ids0_out = warp(kps0_wh_, warp01_params)
            kps1_wh_, kps10_wh, ids1, ids1_out = warp(kps1_wh_, warp10_params)
            if len(kps0_wh_) == 0 or len(kps1_wh_) == 0 or len(kps0[0]) == 0 or len(kps1[0]) == 0:
                raise EmptyTensorError
        except EmptyTensorError:
            return score_map0.new_tensor(0)

        dist01 = compute_keypoints_distance(kps0_wh_, kps10_wh)
        dist10 = compute_keypoints_distance(kps1_wh_, kps01_wh)
        dist_l2 = (dist01 + dist10.t()) / 2.
        dist01_l1 = compute_keypoints_distance(kps0_wh_, kps10_wh, p=1)
        dist10_l1 = compute_keypoints_distance(kps1_wh_, kps01_wh, p=1)
        dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.
        # min distance indices
        mutual_min_indices = mutual_argmin(dist_l2)

        dist_mutual_min = dist_l2[mutual_min_indices]
        valid_dist_mutual_min = dist_mutual_min.detach() < self.train_gt_th

        ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
        ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

        for idx in range(b):
            if self.norm == 2:
                dist = dist_l2
            elif self.norm == 1:
                dist = dist_l1
            else:
                raise TypeError('No such norm in correspondence.')
            scores0 = scores0[0].detach()[ids0_d]
            scores1 = scores1[0].detach()[ids1_d]
            valid = (scores0 > self.scores_th) * (scores1 > self.scores_th)
            reprojection_errors = dist[ids0_d, ids1_d][valid]

            loss_mean = loss_mean + reprojection_errors.sum()
            CNT = CNT + len(reprojection_errors)

        loss_mean = loss_mean / CNT if CNT != 0 else score_map0.new_tensor(0)

        assert not torch.isnan(loss_mean)
        return loss_mean


def local_similarity(descriptor_map, descriptors, kpts_wh, radius):
    """
    :param descriptor_map: CxHxW
    :param descriptors: NxC
    :param kpts_wh: Nx2 (W,H)
    :return:
    """
    _, h, w = descriptor_map.shape
    ksize = 2 * radius + 1

    descriptor_map_unflod = torch.nn.functional.unfold(descriptor_map.unsqueeze(0),
                                                       kernel_size=(ksize, ksize),
                                                       padding=(radius, radius))
    descriptor_map_unflod = descriptor_map_unflod[0].t().reshape(h * w, -1, ksize * ksize)
    # find the correspondence patch
    kpts_wh_long = kpts_wh.detach().long()
    patch_ids = kpts_wh_long[:, 0] + kpts_wh_long[:, 1] * h
    desc_patches = descriptor_map_unflod[patch_ids].permute(0, 2, 1).detach()  # N_kpts x s*s x 128

    local_sim = torch.einsum('nsd,nd->ns', desc_patches, descriptors)
    local_sim_sort = torch.sort(local_sim, dim=1, descending=True).values
    local_sim_sort_mean = local_sim_sort[:, 4:].mean(dim=1)  # 4 is safe radius for bilinear interplation

    return local_sim_sort_mean


class ScoreMapRepLoss(object):
    """ Scoremap repetability"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.radius = 2

    def __call__(self,
                 kps0, scores0, score_map0, similarity_map_01,
                 kps1, scores1, score_map1, similarity_map_10,
                 warp01_params, warp10_params):
        b, c, h, w = score_map0.shape
        wh = kps0[0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        # compute correspondence
        kps0_wh_ = (kps0[0] / 2 + 0.5) * wh  # N0x2, (w,h)
        kps1_wh_ = (kps1[0] / 2 + 0.5) * wh  # N1x2, (w,h)

        try:
            # valid keypoint, valid warped keypoint, valid indices
            kps0_wh_, kps01_wh, ids0, ids0_out = warp(kps0_wh_, warp01_params)
            kps1_wh_, kps10_wh, ids1, ids1_out = warp(kps1_wh_, warp10_params)
            if len(kps0_wh_) == 0 or len(kps1_wh_) == 0 or len(kps0[0]) == 0 or len(kps1[0]) == 0:
                raise EmptyTensorError
        except EmptyTensorError:
            return score_map0.new_tensor(0)

        for idx in range(b):
            kps01 = 2 * kps01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kps10 = 2 * kps10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

            scores_kps10 = torch.nn.functional.grid_sample(score_map0, kps10.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kps01 = torch.nn.functional.grid_sample(score_map1, kps01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]

            s0 = scores_kps01 * scores0[0][ids0]  # repeatability
            s1 = scores_kps10 * scores1[0][ids1]  # repeatability

            # ===================== repetability
            similarity_map_01 = similarity_map_01[ids0]
            similarity_map_10 = similarity_map_10[ids1]

            pmf01 = ((similarity_map_01.detach() - 1) / self.temperature).exp()
            pmf10 = ((similarity_map_10.detach() - 1) / self.temperature).exp()

            kps01 = kps01.detach()
            kps10 = kps10.detach()

            pmf01_kps = torch.nn.functional.grid_sample(pmf01.unsqueeze(0), kps01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kps = torch.nn.functional.grid_sample(pmf10.unsqueeze(0), kps10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            repetability01 = torch.diag(pmf01_kps)
            repetability10 = torch.diag(pmf10_kps)

            # ===================== reliability
            # ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            # descriptor_map0 = pred0['descriptor_map'][idx].detach()
            # descriptor_map1 = pred1['descriptor_map'][idx].detach()
            # descriptors0 = pred0['descriptors'][idx][ids0].detach()
            # descriptors1 = pred1['descriptors'][idx][ids1].detach()
            # kpts0 = pred0['keypoints'][idx][ids0].detach()
            # kpts1 = pred1['keypoints'][idx][ids1].detach()
            # kpts0_wh = (kpts0 / 2 + 0.5) * wh
            # kpts1_wh = (kpts1 / 2 + 0.5) * wh
            # ls0 = local_similarity(descriptor_map0, descriptors0, kpts0_wh, self.radius)
            # ls1 = local_similarity(descriptor_map1, descriptors1, kpts1_wh, self.radius)
            # reliability0 = 1 - ((ls0 - 1) / self.temperature).exp()
            # reliability1 = 1 - ((ls1 - 1) / self.temperature).exp()

            fs0 = repetability01  # * reliability0
            fs1 = repetability10  # * reliability1
            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                loss_mean = loss_mean + loss01.sum()
                CNT = CNT + len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                loss_mean = loss_mean + loss10.sum()
                CNT = CNT + len(loss10)

        loss_mean = loss_mean / CNT if CNT != 0 else kps0[0].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class DescReprojectionLoss(object):
    """ Reprojection loss between warp and descriptor matching """

    def __init__(self, temperature=0.02):
        super().__init__()
        self.inv_temp = 1. / temperature

    def __call__(self, kps0, score_map0, similarity_map_01,
                 kps1, score_map1, similarity_map_10,
                 warp01_params, warp10_params):
        b, c, h, w = score_map0.shape
        device = score_map0.device
        wh = kps0[0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        # compute correspondence
        kps0_wh_ = (kps0[0] / 2 + 0.5) * wh  # N0x2, (w,h)
        kps1_wh_ = (kps1[0] / 2 + 0.5) * wh  # N1x2, (w,h)

        try:
            # valid keypoint, valid warped keypoint, valid indices
            kps0_wh_, kps01_wh, ids0, ids0_out = warp(kps0_wh_, warp01_params)
            kps1_wh_, kps10_wh, ids1, ids1_out = warp(kps1_wh_, warp10_params)
            if len(kps0_wh_) == 0 or len(kps1_wh_) == 0 or len(kps0[0]) == 0 or len(kps1[0]) == 0:
                raise EmptyTensorError
        except EmptyTensorError:
            return score_map0.new_tensor(0)

        for idx in range(b):
            kps01 = 2 * kps01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kps10 = 2 * kps10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

            # ======================= valid

            similarity_map_01_valid, similarity_map_10_valid = similarity_map_01[ids0], similarity_map_10[ids1]
            similarity_map_01_valid = (similarity_map_01_valid - 1) * self.inv_temp
            similarity_map_10_valid = (similarity_map_10_valid - 1) * self.inv_temp

            # matching probability mass function
            pmf01_valid = torch.softmax(similarity_map_01_valid.view(-1, h * w), dim=1).view(-1, h, w)
            pmf10_valid = torch.softmax(similarity_map_10_valid.view(-1, h * w), dim=1).view(-1, h, w)

            pmf01_kpts_valid = torch.nn.functional.grid_sample(pmf01_valid.unsqueeze(0), kps01.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts_valid = torch.nn.functional.grid_sample(pmf10_valid.unsqueeze(0), kps10.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            # as we use the gt correspondence here, the outlier uniform pmf is ignored
            # C_{Q,N} in NRE
            C01 = torch.diag(pmf01_kpts_valid)
            C10 = torch.diag(pmf10_kpts_valid)

            # ======================= out
            similarity_map_01_out, similarity_map_10_out = similarity_map_01[ids0_out], similarity_map_10[ids1_out]
            out0 = torch.ones(len(similarity_map_01_out), device=device)
            out1 = torch.ones(len(similarity_map_10_out), device=device)
            # cat outside scores to similarity_map, thus similarity_map is (N, H*W +1)
            similarity_map_01_out = torch.cat([similarity_map_01_out.reshape(-1, h * w), out0[:, None]], dim=1)
            similarity_map_10_out = torch.cat([similarity_map_10_out.reshape(-1, h * w), out1[:, None]], dim=1)
            similarity_map_01_out = (similarity_map_01_out - 1) * self.inv_temp
            similarity_map_10_out = (similarity_map_10_out - 1) * self.inv_temp
            pmf01_out = torch.softmax(similarity_map_01_out, dim=1)
            pmf10_out = torch.softmax(similarity_map_10_out, dim=1)
            if len(pmf01_out) > 0:
                C01_out = pmf01_out[:, -1]
            else:
                C01_out = C01.new_tensor([])
            if len(pmf10_out) > 0:
                C10_out = pmf10_out[:, -1]
            else:
                C10_out = C10.new_tensor([])

            # ======================= out
            C = torch.cat([C01, C10, C01_out, C10_out])  # C
            C_widetilde = -C.log()  # \widetilde{C}

            loss_mean = loss_mean + C_widetilde.sum()
            CNT = CNT + len(C_widetilde)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class LocalDescLoss(object):
    def __init__(self, temperature=0.02, window_size=80):
        super().__init__()
        self.inv_temp = 1. / temperature
        self.window_size = window_size

    def __call__(self, kps0, score_map0, similarity_map_01,
                 kps1, score_map1, similarity_map_10,
                 warp01_params, warp10_params):
        b, c, h, w = score_map0.shape
        device = score_map0.device
        wh = kps0[0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        # compute correspondence
        kps0_wh_ = (kps0[0] / 2 + 0.5) * wh  # N0x2, (w,h)
        kps1_wh_ = (kps1[0] / 2 + 0.5) * wh  # N1x2, (w,h)

        try:
            # valid keypoint, valid warped keypoint, valid indices
            kps0_wh_, kps01_wh, ids0, ids0_out = warp(kps0_wh_, warp01_params)
            kps1_wh_, kps10_wh, ids1, ids1_out = warp(kps1_wh_, warp10_params)
            if len(kps0_wh_) == 0 or len(kps1_wh_) == 0 or len(kps0[0]) == 0 or len(kps1[0]) == 0:
                raise EmptyTensorError
        except EmptyTensorError:
            return score_map0.new_tensor(0)

        for idx in range(b):
            kps01 = 2 * kps01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kps10 = 2 * kps10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kps01_res = (kps01_wh - kps01_wh.int())
            kps10_res = (kps10_wh - kps10_wh.int())
            # ======================= valid

            similarity_map_01_valid, similarity_map_10_valid = similarity_map_01[ids0], similarity_map_10[ids1]
            similarity_map_01_valid = (similarity_map_01_valid - 1) * self.inv_temp
            similarity_map_10_valid = (similarity_map_10_valid - 1) * self.inv_temp

            # local mask
            kp0 = torch.maximum(torch.tensor([0, 0]).to(kps01_wh.device),
                                kps01_wh.detach().int() - self.window_size // 2)
            kp1 = torch.minimum(torch.tensor([w-1, h-1]).to(kps01_wh.device),
                                kps01_wh.detach().int() + self.window_size // 2 + 1)

            mask = torch.zeros(similarity_map_01_valid.shape, device=device, requires_grad=False)
            for i in range(len(kps01)):
                mask[i, 0:kp0[i, 1], :] = -100
                mask[i, kp1[i, 1]:h, :] = -100
                mask[i, :, 0:kp0[i, 0]] = -100
                mask[i, :, kp1[i, 0]:w] = -100
            similarity_map_01_valid = similarity_map_01_valid + mask

            kp0 = torch.maximum(torch.tensor([0, 0]).to(kps10_wh.device),
                                kps10_wh.detach().int() - self.window_size // 2)
            kp1 = torch.minimum(torch.tensor([w-1, h-1]).to(kps10_wh.device),
                                kps10_wh.detach().int() + self.window_size // 2 + 1)

            mask = torch.zeros(similarity_map_10_valid.shape, device=device, requires_grad=False)
            for i in range(len(kps10)):
                mask[i, 0:kp0[i, 1], :] = -100
                mask[i, kp1[i, 1]:h, :] = -100
                mask[i, :, 0:kp0[i, 0]] = -100
                mask[i, :, kp1[i, 0]:w] = -100
            similarity_map_10_valid = similarity_map_10_valid + mask

            pmf01_valid = torch.softmax(similarity_map_01_valid.view(-1, h * w), dim=1).view(-1, h, w)
            pmf10_valid = torch.softmax(similarity_map_10_valid.view(-1, h * w), dim=1).view(-1, h, w)

            pmf01_kpts_valid = torch.nn.functional.grid_sample(pmf01_valid.unsqueeze(0), kps01.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts_valid = torch.nn.functional.grid_sample(pmf10_valid.unsqueeze(0), kps10.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]

            # as we use the gt correspondence here, the outlier uniform pmf is ignored
            # C_{Q,N} in NRE
            C01 = torch.diag(pmf01_kpts_valid)
            C10 = torch.diag(pmf10_kpts_valid)

            C = torch.cat([C01, C10])  # C
            C_widetilde = -C.log()  # \widetilde{C}

            loss_mean = loss_mean + C_widetilde.sum()
            CNT = CNT + len(C_widetilde)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_tensor(0)
        # print("loss: ", loss_mean)
        assert not torch.isnan(loss_mean)
        return loss_mean


if __name__ == '__main__':
    pk = PeakyLoss()
    line_pk = LinePeakyLoss()
    kps = torch.tensor([[[4, 4]]], dtype=torch.float32)
    scores = torch.tensor([[0.5]], dtype=torch.float32)
    score_map = torch.zeros(1, 1, 8, 8) + 0.8
    score_map[:, :, 0, 0] = 1
    score_map[:, :, 1, 1] = 1
    score_map[:, :, 2, 2] = 1
    score_map[:, :, 3, 3] = 1
    score_map[:, :, 4, 4] = 1
    score_map[:, :, 5, 5] = 1
    score_map[:, :, 6, 6] = 1
    score_map[:, :, 7, 7] = 1
    score_map.requires_grad = True
    print("score_map", score_map)

    loss = pk(kps, scores, score_map)
    # loss.backward()
    # print("pk: ", loss, score_map.grad)

    loss = line_pk(kps, scores, score_map)
    loss.backward()
    print("lpk: ", loss, score_map.grad)

    # nre = DescReprojectionLoss()
    # mask_nre = LocalDescLoss(window_size=4)
    # kps0 = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    # kps1 = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    # score_map = torch.ones(1, 1, 9, 9)
    # similarity_map = torch.zeros(1, 9, 9)
    # similarity_map[:, 4, :] = 0.5
    # similarity_map[:, 4, 0] = 1.5
    # # similarity_map[:, 4, 4] = 0.5
    # similarity_map.requires_grad = True
    #
    # homography = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    # warp_params = {
    #     'mode': 'homo',
    #     'width': 9,
    #     'height': 9,
    #     'homography_matrix': homography,
    #     'k_w': 1,
    #     'k_h': 1
    # }
    #
    # loss = nre(kps0, score_map, similarity_map,
    #            kps1, score_map, similarity_map,
    #            warp_params, warp_params)
    # print("nre: ", loss)
    # loss.backward()
    # # print("grad: ", similarity_map.grad)







