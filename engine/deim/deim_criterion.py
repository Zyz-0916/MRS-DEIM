"""    
DEIM: DETR with Improved Matching for Fast Convergence     
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)  
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.   
""" 

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision 

import copy
 
from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register    

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   

@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """     
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,     
        weight_dict,
        losses,
        alpha=0.2,     
        gamma=2.0,     
        num_classes=80,    
        reg_max=32, 
        boxes_weight_format=None,
        share_matched_indices=False,   
        mal_alpha=None,
        use_uni_set=True,     
        no_weight_vfl_epoch=-1,
        adaptive_fgl=None,
        mal_gamma_schedule=None,
        dn_balance=None,
        small_object_boost=None     
        ):
        """Create the criterion.   
        Parameters:    
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.     
            num_classes: number of object categories, omitting the special no-object category.     
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """    
        super().__init__() 
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict  
        self.losses = losses    
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices     
        self.alpha = alpha  
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max     
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set     
        self.no_weight_vfl_epoch = no_weight_vfl_epoch     
        self.epoch = 0

        self.base_weight_dict = weight_dict.copy()
        self.dynamic_weights = {k: 1.0 for k in weight_dict}
        self.dynamic_weights.setdefault('loss_fgl', 1.0)

        self.adaptive_fgl_cfg = adaptive_fgl or {}
        self.adaptive_fgl_enabled = self.adaptive_fgl_cfg.get('enable', False) and self.base_weight_dict.get('loss_fgl', 0.0) > 0
        self.adaptive_fgl_state = {
            'ema_ratio': None,
            'weight': self.base_weight_dict.get('loss_fgl', 0.0)
        }

        self.mal_gamma_schedule_cfg = mal_gamma_schedule or {}
        self.mal_gamma_schedule_enabled = self.mal_gamma_schedule_cfg.get('enable', False)

        self.dn_balance_cfg = dn_balance or {}
        self.dn_balance_enabled = self.dn_balance_cfg.get('enable', False)

        self.small_object_cfg = small_object_boost or {}
        self.small_object_enabled = self.small_object_cfg.get('enable', False)

        self.current_epoch = 0
        self._eps = 1e-6

        if self.no_weight_vfl_epoch != -1:     
            print(RED + f"no_weight_vfl_epoch set {self.no_weight_vfl_epoch}" + RESET)

    @torch.no_grad()  
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes 
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients  
        """     
        pred_logits = outputs['pred_logits']  
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)    
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)  
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())    
        losses = {'cardinality_error': card_err}
        return losses   

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):   
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  
        idx = self._get_src_permutation_idx(indices)   
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]    
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes 
     
        return {'loss_focal': loss}   

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None): 
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) 
        if values is None:   
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) 
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))  
            ious = torch.diag(ious).detach() 
        else:    
            ious = values    
     
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        target_classes[idx] = target_classes_o 
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)     
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
  
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch: 
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        else:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}
    
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):   
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)  
        if values is None:  
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))  
            ious = torch.diag(ious).detach()     
        else:
            ious = values
     
        src_logits = outputs['pred_logits']     
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o 
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
     
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)    
        target_score_o[idx] = ious.to(target_score_o.dtype)    
        target_score_raw = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach() 
        target_score = target_score_raw.pow(self.gamma)

        dynamic_gamma = self._compute_mal_gamma(ious)
        if dynamic_gamma is not None and dynamic_gamma.numel() > 0:
            target_score = self._inject_dynamic_gamma(target_score, target_score_raw, target_classes_o, idx, dynamic_gamma)
        if self.mal_alpha != None:     
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target   

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))     
        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch:     
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none') 
        else:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')    
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes    
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """   
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)    
        src_boxes = outputs['pred_boxes'][idx]   
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)   
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.view(target_boxes.shape[0], -1)
        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))    

        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight 

        if self._small_object_active() and target_boxes.numel() > 0:
            small_obj_weights = self._compute_small_object_weights(target_boxes)
            if small_obj_weights is not None:
                loss_bbox = loss_bbox * small_obj_weights.unsqueeze(-1)
                loss_giou = loss_giou * small_obj_weights

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses    

    def loss_local(self, outputs, targets, indices, num_boxes, T=5): 
        """Compute Fine-Grained Localization (FGL) Loss   
            and Decoupled Distillation Focal (DDF) Loss. """
     
        losses = {}   
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)    
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) 

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1)) 
            ref_points = outputs['ref_points'][idx].detach()   
            with torch.no_grad():   
                if self.fgl_targets_dn is None and 'is_dn' in outputs:   
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])     
                if self.fgl_targets is None and 'is_dn' not in outputs: 
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
     
            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets 
    
            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)     

            if 'teacher_corners' in outputs:  
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))   
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]     
    
                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True    
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)   
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)     
                    if 'is_dn' not in outputs: 
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0     
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)    
    
        return losses
  
    def _get_src_permutation_idx(self, indices):     
        # permute predictions following indices     
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])   
        src_idx = torch.cat([src for (src, _) in indices])  
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):  
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx 

    def _get_go_indices(self, indices, indices_aux_list): 
        """Get a matching union set across all decoder layers. """  
        results = []    
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)     
            count_sort_indices = torch.argsort(counts, descending=True) 
            unique_sorted = unique[count_sort_indices]   
            column_to_row = {} 
            for idx in unique_sorted:    
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:  
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)   
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):    
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None     
   
    def _pre_forward_dynamic(self, epoch):
        if not self.adaptive_fgl_enabled:
            return
        if epoch is None:
            return
        if epoch < self.adaptive_fgl_cfg.get('start_epoch', 0):
            self.dynamic_weights['loss_fgl'] = 1.0
            self.adaptive_fgl_state['weight'] = self.base_weight_dict.get('loss_fgl', 0.0)
            self.adaptive_fgl_state['ema_ratio'] = None

    def _post_forward_dynamic(self, raw_losses):
        if self.adaptive_fgl_enabled:
            self._update_adaptive_fgl(raw_losses)

    def _scale_loss_dict(self, raw_dict, suffix="", dn_idx=None):
        scaled = {}
        for key, value in raw_dict.items():
            if key not in self.base_weight_dict:
                continue
            base_weight = self.base_weight_dict[key]
            if base_weight == 0:
                continue
            dynamic_mult = self.dynamic_weights.get(key, 1.0)
            if dn_idx is not None:
                dynamic_mult *= self._get_dn_multiplier(key, dn_idx)
            weight = base_weight * dynamic_mult
            if weight == 0:
                continue
            scaled[key + suffix] = value * weight
        return scaled

    def _update_adaptive_fgl(self, raw_losses):
        if not raw_losses or 'loss_fgl' not in raw_losses:
            return
        start_epoch = self.adaptive_fgl_cfg.get('start_epoch', 0)
        if self.current_epoch < start_epoch:
            return

        base_weight = self.base_weight_dict.get('loss_fgl', 0.0)
        if base_weight <= 0:
            return

        loss_fgl = raw_losses['loss_fgl']
        if not torch.is_tensor(loss_fgl):
            loss_fgl = torch.as_tensor(loss_fgl, dtype=torch.float32)
        else:
            loss_fgl = loss_fgl.detach()

        device = loss_fgl.device

        loss_bbox = raw_losses.get('loss_bbox', 0.0)
        loss_giou = raw_losses.get('loss_giou', 0.0)
        loss_bbox_value = torch.as_tensor(loss_bbox, device=device, dtype=loss_fgl.dtype)
        loss_giou_value = torch.as_tensor(loss_giou, device=device, dtype=loss_fgl.dtype)

        loc_weight = torch.zeros(1, device=device, dtype=loss_fgl.dtype)
        if 'loss_bbox' in self.base_weight_dict:
            loc_weight = loc_weight + loss_bbox_value * self.base_weight_dict['loss_bbox']
        if 'loss_giou' in self.base_weight_dict:
            loc_weight = loc_weight + loss_giou_value * self.base_weight_dict['loss_giou']

        if torch.abs(loc_weight).item() <= self._eps:
            return

        current_multiplier = self.dynamic_weights.get('loss_fgl', 1.0)
        current_weight = base_weight * current_multiplier
        fgl_weighted = loss_fgl * current_weight
        ratio = (fgl_weighted / (loc_weight + self._eps)).item()

        ema = self.adaptive_fgl_state.get('ema_ratio')
        momentum = self.adaptive_fgl_cfg.get('momentum', 0.9)
        if ema is None:
            ema = ratio
        else:
            ema = momentum * ema + (1 - momentum) * ratio
        self.adaptive_fgl_state['ema_ratio'] = ema

        target_ratio = self.adaptive_fgl_cfg.get('target_ratio', 0.6)
        adjust_power = self.adaptive_fgl_cfg.get('adjust_power', 0.5)
        desired = target_ratio / max(ema, self._eps)
        adjust = desired ** adjust_power
        new_weight = current_weight * adjust

        min_weight = self.adaptive_fgl_cfg.get('min_weight', None)
        max_weight = self.adaptive_fgl_cfg.get('max_weight', None)
        if min_weight is not None:
            new_weight = max(min_weight, new_weight)
        if max_weight is not None:
            new_weight = min(max_weight, new_weight)

        max_step = self.adaptive_fgl_cfg.get('max_step', 0.25)
        max_delta = current_weight * max_step
        new_weight = current_weight + max(-max_delta, min(max_delta, new_weight - current_weight))

        warmup_epochs = self.adaptive_fgl_cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0 and self.current_epoch < start_epoch + warmup_epochs:
            progress = max(0.0, min(1.0, (self.current_epoch - start_epoch + 1) / max(1, warmup_epochs)))
            new_weight = current_weight + (new_weight - current_weight) * progress

        self.dynamic_weights['loss_fgl'] = new_weight / base_weight
        self.adaptive_fgl_state['weight'] = new_weight

    def _get_dn_multiplier(self, key, dn_idx):
        if not self.dn_balance_enabled or key != 'loss_ddf':
            return 1.0

        start_epoch = self.dn_balance_cfg.get('start_epoch', 0)
        if self.current_epoch < start_epoch:
            return 1.0

        weights = self.dn_balance_cfg.get('weights', [])
        if not weights:
            return 1.0
        if dn_idx >= len(weights):
            dn_idx = len(weights) - 1
        target = weights[dn_idx]

        warmup = self.dn_balance_cfg.get('warmup_epochs', 0)
        if warmup > 0:
            progress = max(0.0, min(1.0, (self.current_epoch - start_epoch + 1) / max(1, warmup)))
            return 1.0 + (target - 1.0) * progress
        return target

    def _small_object_active(self):
        if not self.small_object_enabled:
            return False
        start = self.small_object_cfg.get('start_epoch', 0)
        end = self.small_object_cfg.get('end_epoch', None)
        if self.current_epoch < start:
            return False
        if end is not None and self.current_epoch > end:
            return False
        return True

    def _compute_small_object_weights(self, target_boxes):
        if target_boxes.numel() == 0:
            return None
        areas = (target_boxes[:, 2] * target_boxes[:, 3]).to(target_boxes.device)
        threshold = self.small_object_cfg.get('area_threshold', 0.02)
        boost = self.small_object_cfg.get('boost_factor', 1.2)
        if boost <= 1.0:
            return None
        mask = (areas <= threshold).float()
        if mask.sum() == 0:
            return None
        warmup = self.small_object_cfg.get('warmup_epochs', 0)
        progress = 1.0
        start_epoch = self.small_object_cfg.get('start_epoch', 0)
        if warmup > 0:
            progress = max(0.0, min(1.0, (self.current_epoch - start_epoch + 1) / max(1, warmup)))
        weights = torch.ones_like(areas, dtype=target_boxes.dtype, device=target_boxes.device)
        weights = weights + (boost - 1.0) * progress * mask
        return weights

    def _compute_mal_gamma(self, ious):
        if not self.mal_gamma_schedule_enabled:
            return None
        if ious is None or ious.numel() == 0:
            return None
        start_epoch = self.mal_gamma_schedule_cfg.get('start_epoch', 0)
        if self.current_epoch < start_epoch:
            return None

        ious = ious.clamp(0, 1)
        low = self.mal_gamma_schedule_cfg.get('low_iou', 0.3)
        high = self.mal_gamma_schedule_cfg.get('high_iou', 0.7)
        gamma_low = self.mal_gamma_schedule_cfg.get('gamma_low', self.gamma)
        gamma_high = self.mal_gamma_schedule_cfg.get('gamma_high', self.gamma)

        if high <= low:
            gamma_vals = torch.full_like(ious, gamma_high)
        else:
            denom = max(high - low, self._eps)
            t = (ious - low) / denom
            t = t.clamp(0, 1)
            gamma_vals = gamma_low + (gamma_high - gamma_low) * t

        warmup = self.mal_gamma_schedule_cfg.get('warmup_epochs', 0)
        if warmup > 0 and self.current_epoch < start_epoch + warmup:
            progress = max(0.0, min(1.0, (self.current_epoch - start_epoch + 1) / max(1, warmup)))
            base = torch.full_like(gamma_vals, self.gamma)
            gamma_vals = base + (gamma_vals - base) * progress

        gamma_min = self.mal_gamma_schedule_cfg.get('gamma_min', None)
        gamma_max = self.mal_gamma_schedule_cfg.get('gamma_max', None)
        if gamma_min is not None or gamma_max is not None:
            gamma_vals = gamma_vals.clamp(min=gamma_min if gamma_min is not None else -float('inf'),
                                          max=gamma_max if gamma_max is not None else float('inf'))

        return gamma_vals

    def _inject_dynamic_gamma(self, target_score, target_score_raw, target_classes_o, idx, gamma_vals):
        if gamma_vals is None or gamma_vals.numel() == 0:
            return target_score
        pos_raw = target_score_raw[idx]
        if pos_raw.numel() == 0:
            return target_score
        gather_idx = target_classes_o.unsqueeze(-1)
        gt_scores = pos_raw.gather(-1, gather_idx).clamp_min(self._eps)
        gamma_vals = gamma_vals.to(gt_scores.device)
        gt_scores = torch.pow(gt_scores, gamma_vals.unsqueeze(-1))
        pos_with_gamma = target_score[idx]
        pos_with_gamma = pos_with_gamma.scatter(-1, gather_idx, gt_scores)
        target_score[idx] = pos_with_gamma
        return target_score
   
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):     
        loss_map = { 
            'boxes': self.loss_boxes,     
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,  
            'local': self.loss_local,    
            'cardinality': self.loss_cardinality,
        }   
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)  
     
    def forward(self, outputs, targets, **kwargs):  
        """ This performs the loss computation.     
        Parameters:   
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.    
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        epoch = kwargs.get('epoch', 0)
        self.current_epoch = epoch
        if epoch is not None:
            self.epoch = epoch

        self._pre_forward_dynamic(epoch)

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}    

        # Retrieve the matching between the outputs of the last layer and the targets   
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)['indices']
        self._clear_cache()    
     
        # Get the matching union set across all decoder layers.    
        if 'aux_outputs' in outputs:     
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:     
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']] 
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)['indices']  
                cached_indices.append(indices_aux)    
                indices_aux_list.append(indices_aux)    
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):  
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)     
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)  
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)  
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, '' 

        # Compute the average number of target boxes accross all nodes, for normalization purposes 
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)  
        if is_dist_available_and_initialized():     
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item() 

        # Compute all the requested losses, main loss 
        losses = {}
        main_raw_losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2  
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])     
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes   
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            raw_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)     
            for k, v in raw_dict.items():
                if k not in main_raw_losses:
                    main_raw_losses[k] = v.detach() if torch.is_tensor(v) else v
                else:
                    main_raw_losses[k] = main_raw_losses[k] + (v.detach() if torch.is_tensor(v) else v)
            scaled = self._scale_loss_dict(raw_dict)
            losses.update(scaled)
     
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss    
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale'] 
                for loss in self.losses:     
                    # TODO, indices and num_box are different from RT-DETRv2   
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes  
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)   
                    raw_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                    scaled = self._scale_loss_dict(raw_dict, suffix=f'_aux_{i}')
                    losses.update(scaled)
  
        # In case of auxiliary traditional head output at first decoder layer. just for dfine    
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']    
            for loss in self.losses:  
                # TODO, indices and num_box are different from RT-DETRv2  
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)  
                raw_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)    
                scaled = self._scale_loss_dict(raw_dict, suffix='_pre')
                losses.update(scaled)

        # In case of encoder auxiliary losses.   
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic: 
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:     
                enc_targets = targets  
 
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']): 
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2 
                    use_uni_set = self.use_uni_set and (loss == 'boxes') 
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes  
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    raw_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)    
                    scaled = self._scale_loss_dict(raw_dict, suffix=f'_enc_{i}')
                    losses.update(scaled)
  
            if class_agnostic:  
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs: 
            assert 'dn_meta' in outputs, ''    
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
     
            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss    
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn) 
                    raw_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta) 
                    scaled = self._scale_loss_dict(raw_dict, suffix=f'_dn_{i}', dn_idx=i)
                    losses.update(scaled) 
   
            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:   
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    raw_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)  
                    scaled = self._scale_loss_dict(raw_dict, suffix='_dn_pre')
                    losses.update(scaled)
   
        # For debugging Objects365 pre-train.
        self._post_forward_dynamic(main_raw_losses)

        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}    
        return losses  

    def get_loss_meta_info(self, loss, outputs, targets, indices): 
        if self.boxes_weight_format is None:   
            return {}     
   
        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)  

        if self.boxes_weight_format == 'iou':   
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))   
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':     
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))    
        else:
            raise AttributeError()
     
        if loss in ('boxes', ): 
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}    

        return meta   
 
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices    
        """   
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]  
        num_gts = [len(t['labels']) for t in targets]  
        device = targets[0]['labels'].device   

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:     
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)  
                gt_idx = gt_idx.tile(dn_num_group)   
                assert len(dn_positive_idx[i]) == len(gt_idx)  
                dn_match_indices.append((dn_positive_idx[i], gt_idx))     
            else:    
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device))) 

        return dn_match_indices     
 
 
    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

  
    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):   
        dis_left = label.long()  
        dis_right = dis_left + 1
 
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)   

        if weight is not None:   
            weight = weight.float()
            loss = loss * weight   

        if avg_factor is not None:
            loss = loss.sum() / avg_factor   
        elif reduction == 'mean':   
            loss = loss.mean()    
        elif reduction == 'sum':   
            loss = loss.sum()   

        return loss   

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1   
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list 
     
    def set_epoch(self, epoch):
        self.epoch = epoch    
        self.current_epoch = epoch
        if self.adaptive_fgl_enabled and epoch < self.adaptive_fgl_cfg.get('start_epoch', 0):
            self.dynamic_weights['loss_fgl'] = 1.0
            self.adaptive_fgl_state['ema_ratio'] = None
            self.adaptive_fgl_state['weight'] = self.base_weight_dict.get('loss_fgl', 0.0)
        if self.no_weight_vfl_epoch != -1 and self.epoch == self.no_weight_vfl_epoch:     
            print(RED + f"Epoch:[{self.epoch}]>=[{self.no_weight_vfl_epoch}] using weight in vfl/mal." + RESET)  
