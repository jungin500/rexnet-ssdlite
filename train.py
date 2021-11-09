import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import *
from loguru import logger

import torch
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from dataset import VOCDetectionDataset
from model import (
    ssdlite224_rexnet_v1_lightning,
    collate_batch
)


class Transform:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        tensor = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        tensor -= np.array(self.mean).reshape(3, 1, 1)
        tensor /= np.array(self.std).reshape(3, 1, 1)
        tensor = torch.from_numpy(tensor)
        return tensor


def main(args):

    # Arguments
    dataset_root_path, num_classes, backbone_weight_path, batch_size, num_workers, persistent_workers, resume_path, is_freeze_base_net, is_half_precision = \
        args.dataset, args.num_classes, args.backbone_weight, args.batch_size, args.num_workers, args.persistent_workers, args.resume, args.freeze_base_net, args.half

    if backbone_weight_path is None and resume_path is None:
        logger.error("Required either --bacbone-weight or --resume")
        return

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    logger.info("Training on %s" % (device_type, ))

    # Main network
    model = ssdlite224_rexnet_v1_lightning(
        backbone_weight_path=backbone_weight_path,
        is_freeze_base_net=is_freeze_base_net,
        num_classes=num_classes
    ).to(device)

    if resume_path is not None:
        # model.load_state_dict(torch.load(resume_path))
        model.load_from_checkpoint(checkpoint_path=resume_path)

    # Dataloader
    dataset = VOCDetectionDataset(
        data_root=dataset_root_path,
        folds='trainval',
        num_classes=num_classes,
        cache=True,
        class_map={0: 'Person', 1: 'Vehicle', 2: 'Corn'},
        transform=Transform(),
        half=is_half_precision,
        size=(416, 416)
    )

    trainset_items = int(len(dataset) * 0.9)
    validset_items = len(dataset) - trainset_items
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [trainset_items, validset_items])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )

    # Train model
    trainer = pl.Trainer(
        gpus=1, num_nodes=1,
        precision=16 if is_half_precision else 32,
        max_epochs=1000,
        callbacks=[
            ModelCheckpoint(
                monitor='valid_mean_ap',
                dirpath='checkpoints/',
                filename='rexnet-ssdlite-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='max'
            )
        ]
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True,
                        help='Dataset root path')
    parser.add_argument('--backbone-weight', '-bw', default=None,
                        help='Backbone weight RexNetV1 W=1.0 path')
    parser.add_argument('--batch-size', '-b', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=0,
                        help='Workers (Recommended to set equals core size)')
    parser.add_argument('--num-classes', '-c', type=int,
                        default=1000, help='Class count')
    parser.add_argument('--persistent-workers', '-p',
                        action='store_true', help='Persistent worker thread')
    parser.add_argument('--resume', '-r', default=None,
                        help='Resume weight path (RexNetSSDDetector)')
    parser.add_argument('--half', action='store_true',
                        help='(RECOMMENDED) Half precision (FP16)')
    parser.add_argument('--freeze-base-net', '-f', action='store_true',
                        help='(RECOMMENDED) Freeze backbone network')
    args = parser.parse_args()
    main(args)
