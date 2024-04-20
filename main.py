import os
import sys
import time
import logging
import functools
from pathlib import Path
sys.path.append('./')

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.hpatches import HPatchesDataset
from datasets.megadepth import MegaDepthDataset
from datasets.cat_datasets import ConcatDatasets
from training.scheduler import WarmupConstantSchedule
from training.train_wrapper import TrainWrapper
from pytorch_lightning.callbacks import Callback



class RebuildDatasetCallback(Callback):
    def __init__(self):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        train_loader_dataset = trainer.train_dataloader[0].dataset
        train_loader_dataset.datasets[0].build_dataset()
        train_loader_dataset.datasets[1].build_dataset()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.autograd.set_detect_anomaly(True)

    pretrained_model = None
    debug = False

    c1 = 8
    c2 = 16
    c3 = 32
    c4 = 64
    dim = 64
    gray = False

    radius = 2
    top_k = 400
    scores_th_eval = 0.2
    n_limit_eval = 5000

    train_gt_th = 5
    eval_gt_th = 3

    w_pk = 0.5
    w_rp = 1
    w_sp = 1
    w_ds = 5
    w_mds = 1
    sc_th = 0.1
    norm = 1
    temp_sp = 0.1
    temp_ds = 0.02

    gpus = [0]
    warmup_steps = 500
    t_total = 10000
    image_size = 480
    log_freq_img = 400

    log_dir = 'log_' + Path(__file__).stem

    hpatch_dir = '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/WangShuo/datasets/HPatch'
    mega_dir = '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/disk-data/megadepth'
    imw2020val_dir = '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/disk-data/imw2020-val'

    batch_size = 1
    if debug:
        accumulate_grad_batches = 1
        num_workers = 0
        num_sanity_val_steps = 0

        reload_dataloaders_every_epoch = False
        limit_train_batches = 1
        limit_val_batches = 1.
        max_epochs = 100
    else:
        accumulate_grad_batches = 16
        num_workers = 8
        num_sanity_val_steps = 1

        reload_dataloaders_every_epoch = True
        limit_train_batches = 5000 // batch_size
        limit_val_batches = 1.
        max_epochs = 200

    lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=warmup_steps)

    # model

    # dataset
    mega_dataset1 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
                                     image_size=image_size, gray=False, colorjit=True, crop_or_scale='crop')
    mega_dataset2 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
                                     image_size=image_size, gray=False, colorjit=True, crop_or_scale='scale')
    train_datasets = ConcatDatasets(mega_dataset1, mega_dataset2)

    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, pin_memory=not debug,
                              num_workers=num_workers)

    hpatch_i_dataset = HPatchesDataset(root=hpatch_dir, alteration='i')
    hpatch_v_dataset = HPatchesDataset(root=hpatch_dir, alteration='v')
    hpatch_i_dataloader = DataLoader(hpatch_i_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers)
    hpatch_v_dataloader = DataLoader(hpatch_v_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers)

    imw2020val = MegaDepthDataset(root=imw2020val_dir, train=False, using_cache=True, colorjit=False, gray=False)
    imw2020val_dataloader = DataLoader(imw2020val, batch_size=1, pin_memory=not debug, num_workers=num_workers)

    log_name = 'debug' if debug else 'train'
    version = time.strftime("Version-%m%d-%H%M%S", time.localtime())

    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name, version=version, default_hp_metric=False)
    logging.info(f'>>>>>>>>>>>>>>>>> log dir: {logger.log_dir}')

    trainer = pl.Trainer(devices=gpus,
                         fast_dev_run=False,
                         accumulate_grad_batches=accumulate_grad_batches,
                         num_sanity_val_steps=num_sanity_val_steps,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         max_epochs=max_epochs,
                         logger=logger,
                         reload_dataloaders_every_n_epochs=reload_dataloaders_every_epoch,
                         callbacks=[
                             ModelCheckpoint(monitor='val_metrics/mean', save_top_k=3,
                                             mode='max', save_last=True,
                                             dirpath=logger.log_dir + '/checkpoints',
                                             auto_insert_metric_name=False,
                                             filename='epoch={epoch}-mean_metric={val_metrics/mean:.4f}'),
                             LearningRateMonitor(logging_interval='step'),
                             RebuildDatasetCallback()
                         ]
                         )

    model = TrainWrapper(
        c1=c1, c2=c2, c3=c3, c4=c4, dim=dim, gray=gray,
        radius=radius, top_k=top_k, scores_th=0, n_limit=0,
        scores_th_eval=scores_th_eval, n_limit_eval=n_limit_eval,
        train_gt_th=train_gt_th, eval_gt_th=eval_gt_th,
        w_pk=w_pk, w_rp=w_rp, w_sp=w_sp, w_ds=w_ds, w_mds=w_mds,
        sc_th=sc_th, norm=norm, temp_sp=temp_sp, temp_ds=temp_ds,
        lr=3e-4, log_freq_img=log_freq_img,
        pretrained_model=pretrained_model,
        lr_scheduler=lr_scheduler,
        debug=debug)

    trainer.fit(model, train_dataloaders=[train_loader],
                val_dataloaders=[hpatch_i_dataloader, hpatch_v_dataloader, imw2020val_dataloader])