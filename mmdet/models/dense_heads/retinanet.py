from retina_head import RetinaHead
import torch

self = RetinaHead(11, 7)
x = torch.rand(1, 7, 32, 32)
cls_score, bbox_pred = self.forward_single(x)
# Each anchor predicts a score for each class except background
cls_per_anchor = cls_score.shape[1] / self.num_anchors
box_per_anchor = bbox_pred.shape[1] / self.num_anchors
assert cls_per_anchor == (self.num_classes)
assert box_per_anchor == 4