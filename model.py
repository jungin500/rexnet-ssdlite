import pytorch_lightning as pl

from typing import *

from rexnetv1 import ReXNetV1, LinearBottleneck, SiLU
from loguru import logger

import torch
from torch import nn

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead, SSD
import torchvision.models.detection._utils as det_utils
from functools import partial

import numpy as np

from mean_average_precision import MetricBuilder


def hack_inner2d_out(module, x):
    '''RexNetV1-related hacks
    
    '''
    modules_to_forward = module[:-1]
    last_module = module[-1]  # LinearBottleneck
    last_module_seq = last_module.out  # Sequential
    until_se_module_relu6_out = last_module_seq[:7]
    remainders = last_module_seq[7:]
    
    x = modules_to_forward(x)
    x = until_se_module_relu6_out(x)
    y = remainders(x)
    return x, y
    

class ReXNetSSDBackbone(nn.Module):
    def __init__(self, rexnetv1):
        super(ReXNetSSDBackbone, self).__init__()
        self.backbone_features = rexnetv1.features
        self.feature0_head = nn.Sequential(
            nn.Conv2d(in_channels=702, out_channels=672, kernel_size=(3, 3), padding=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=672),
            SiLU()
        )
        self.feature1_head = nn.Sequential(
            nn.Conv2d(in_channels=1044, out_channels=480, kernel_size=(3, 3), padding=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=480),
            SiLU()
        )
        
        self.backbone_tails = nn.ModuleList([
            LinearBottleneck(
                in_channels=in_c,
                channels=c,
                t=t,
                stride=s,
                use_se=se,
                se_ratio=se_ratio
            ) for in_c, c, t, s, se, se_ratio in [
                (185, 512, 1, 2, False, 12),
                (512, 256, 1, 2, False, 12),
                (256, 256, 1, 2, False, 12),
                (256, 128, 1, 1, False, 12),
            ]
        ])

    def forward(self, x):
        x14_inner_relu6_out, x = hack_inner2d_out(self.backbone_features[:-9], x)  # [1, 702, 14, 14] (ReLU6 applied), [1, 128, 14, 14]
        x7_inner_relu6_out, x = hack_inner2d_out(self.backbone_features[-9:-4], x)  # [1, 1044, 7, 7] (ReLU6 applied), [1, 185, 7, 7]
        
        # First two heads will use x14, x7 outputs
        result_dicts = {}
        
        feature0 = self.feature0_head(x14_inner_relu6_out)
        feature1 = self.feature1_head(x7_inner_relu6_out)
        result_dicts['0'] = feature0
        result_dicts['1'] = feature1
        
        for idx in range(len(self.backbone_tails)):
            x = self.backbone_tails[idx](x)
            result_dicts[str(idx + 2)] = x  # idx should begin from 2
        return result_dicts


class ReXNetSSD(pl.LightningModule):
    def __init__(self, batch_size, rexnet_ssd_model = None, num_classes = None):
        super(ReXNetSSD, self).__init__()
        if rexnet_ssd_model is None and num_classes is None:
            self.model = ssdlite224_rexnet_v1(
                backbone_weight_path = None,
                is_freeze_base_net = None,
                num_classes = 20
            )
        else:
            self.model = rexnet_ssd_model
        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=3)
        self.num_classes = num_classes
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, training_batch, batch_idx):
        '''
        loss_dict = model(
            torch.zeros(2, 3, 224, 224).cuda(),
            [{
                'boxes': torch.from_numpy(np.array([[0.2, 0.2, 0.6, 0.6], [0.2, 0.2, 0.6, 0.6]])).float().cuda(),
                'labels': torch.zeros(2).long().cuda(),
            } for _ in range(2)]
        )
        => (Training Mode)
        {
            'bbox_regression': tensor(36.4136, device='cuda:0', grad_fn=<DivBackward0>),
            'classification': tensor(0., device='cuda:0', grad_fn=<DivBackward0>)
        }
        '''
        image_batch, annotations_batch = training_batch
        loss_dict = self.model(image_batch, annotations_batch)
        self.log('train_loss_bbox_regression', loss_dict['bbox_regression'], batch_size=self.batch_size)
        self.log('train_loss_classification', loss_dict['classification'], batch_size=self.batch_size)

        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.metric_fn.reset()

    def validation_step(self, val_batch, batch_idx):
        image_batch, annotations_batch = val_batch
        # model.eval() is called before validation_step
        gt_boxes, gt_labels = [m['boxes'] for m in annotations_batch], [
            m['labels'] for m in annotations_batch]

        # Takes too much time
        # self.model.train()
        # loss_dict = self.model(image_batch, annotations_batch)
        # self.log('valid_loss_bbox_regression', loss_dict['bbox_regression'])
        # self.log('valid_loss_classification', loss_dict['classification'])
        # loss = sum(loss for loss in loss_dict.values())
        # self.log('valid_loss', loss)
        # self.model.eval()

        with torch.no_grad():
            batched_result = self.model(image_batch)

            for idx in range(len(batched_result)):
                single_result = batched_result[idx]
                gt_single_boxes, gt_single_labels = gt_boxes[idx], gt_labels[idx]
                gt_single_boxes, gt_single_labels = gt_single_boxes.cpu(), gt_single_labels.cpu()

                boxes, scores, labels = single_result['boxes'], single_result['scores'], single_result['labels']
                boxes, scores, labels = boxes.cpu().numpy(
                ), scores.cpu().numpy(), labels.cpu().numpy()
                boxes, scores, labels = boxes, np.expand_dims(
                    scores, 1), np.expand_dims(labels, 1)
                preds = np.concatenate([boxes, labels, scores], 1)

                # print("Filtered preds size:", filtered_preds.shape)

                gt_single_boxes, gt_single_labels = gt_single_boxes, np.expand_dims(
                    gt_single_labels, 1)
                gt = np.concatenate([gt_single_boxes, gt_single_labels, np.zeros(
                    (gt_single_boxes.shape[0], 2))], 1)
                # np.zeros(gt_single_boxes.shape[0], 2) -> difficult, crowd

                self.metric_fn.add(preds, gt)

            valid_mean_ap = self.metric_fn.value(iou_thresholds=0.5)['mAP']
            self.log('valid_mean_ap', valid_mean_ap, batch_size=self.batch_size)
            return valid_mean_ap

    def on_validation_epoch_end(self):
        logger.info("Validation mAP: %1.4f" %
                    self.metric_fn.value(iou_thresholds=0.5)['mAP'])


def ssdlite224_rexnet_v1_lightning(
    batch_size,
    **kwargs: Any
):
    model = ssdlite224_rexnet_v1(**kwargs)
    return ReXNetSSD(batch_size, model, kwargs['num_classes'])


def ssdlite224_rexnet_v1(
    backbone_weight_path: str,
    is_freeze_base_net: bool,
    num_classes: int = 80,
    **kwargs: Any,
):
    backbone = ReXNetV1()
    if is_freeze_base_net:
        logger.info("Disabling updates on backbone")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        logger.info("Enabling updates on backbone")

    # Load weight
    if backbone_weight_path is not None:
        backbone.load_state_dict(torch.load(backbone_weight_path))
    backbone = ReXNetSSDBackbone(backbone)

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    return model


def collate_batch(batch):
    return tuple(zip(*batch))
