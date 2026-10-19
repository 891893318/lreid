"""最终无提示词持续行人重识别方法的单阶段训练器。
包含 CE + Triplet、原始关系知识迁移、GSCM、SA-RSTKT 和参考状态 SDAC；
SDAC 以固定 PSC 属性坐标组织同一身份的 occupancy、visual 和 decision states，
并使用同阶段 EMA 跟踪这些聚合状态，不做逐样本 feature/logit/anchor 三路蒸馏。
SA-LTKC 不在此处执行，而是在 ``continual_train.py`` 的每个训练阶段结束后执行。
"""

from __future__ import absolute_import, print_function

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.loss.gscm import (
    DEFAULT_ANCHOR_GROUPS,
    GLOBAL_ANCHOR_GROUPS,
    SHUFFLED_ANCHOR_GROUPS,
    grouped_semantic_cross_camera_margin_loss,
)
from reid.metric_learning.distance import cosine_similarity
from reid.loss.triplet_loss_transreid import TripletLoss
from reid.utils.make_loss import make_loss
from .utils.meters import AverageMeter


class Trainer(object):
    """封装前向传播、损失计算和反向传播。"""

    def __init__(self, cfg, args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        controller_mode = getattr(args, 'psc_controller_mode', 'joint')
        controller_transfer_scale = 1.0
        if (bool(getattr(args, 'psc_controller', False))
                and controller_mode in ('transfer', 'joint')):
            controller_transfer_scale = float(
                getattr(args, 'psc_controller_current_transfer_scale', 1.0))
        self.AF_weight = args.AF_weight * controller_transfer_scale
        self.psc_controller_transfer_scale = controller_transfer_scale

        # 同时兼容普通模型与 DataParallel/DistributedDataParallel 包装的模型。
        feat_dim = getattr(getattr(model, 'module', model), 'out_channel', 2048)
        self.loss_fn, _ = make_loss(cfg, num_classes=num_classes, feat_dim=feat_dim)
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        triplet_margin = None if bool(cfg.MODEL.NO_MARGIN) else cfg.SOLVER.MARGIN
        self.acssd_triplet = TripletLoss(triplet_margin)
        self.sdac_teacher = None
        self.sdac_teacher_phase = None
        # Cross-batch state targets are training-only CPU memories.  Keeping
        # them off GPU is important because decision prototypes grow with the
        # continual classifier.  A batch is queued during loss construction
        # and committed only after the optimizer/parameter-EMA updates.
        self.sdac_state_memory = {}
        self.sdac_state_memory_phase = None
        self._sdac_pending_state_update = None
        self._sdac_random_coordinate_anchors = None

    def train(self, epoch, data_loader_train, optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None):
        self.model.train()
        self._current_epoch = epoch

        # 已冻结的 BN 层保持推理模式，避免其 running mean/variance 继续变化。
        base = getattr(self.model.module, 'base', None)
        if base is not None:
            for m in base.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.requires_grad is False and m.bias.requires_grad is False:
                        m.eval()

        # 分别统计耗时及各损失项，便于日志输出和 TensorBoard 可视化。
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_af = AverageMeter()
        losses_gscm = AverageMeter()
        losses_sdac = AverageMeter()
        losses_sdac_occupancy = AverageMeter()
        losses_sdac_visual = AverageMeter()
        losses_sdac_decision = AverageMeter()
        losses_sdac_states = AverageMeter()
        losses_acssd_stable = AverageMeter()
        losses_acssd_plastic = AverageMeter()
        losses_srsr_stable = AverageMeter()
        losses_srsr_plastic = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, _, sample_keys = self._parse_data(train_inputs)
            # 不同阶段的身份标签占用互不重叠的全局编号区间。
            targets += add_num
            s_features, bn_feat, cls_outputs, _ = self.model(s_inputs)

            # 当前任务的基本监督：身份分类损失与三元组度量损失。
            loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
            loss = loss_ce + loss_tp
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            # GSCM：利用语义锚点约束跨摄像头样本的特征间隔。
            gscm_loss = self._gscm_loss(s_features, targets, cids, training_phase)
            if gscm_loss is not None:
                loss = loss + gscm_loss
                losses_gscm.update(gscm_loss.item())

            s_features_old = None
            if old_model is not None:
                # 旧模型仅作为教师提供历史特征，不参与梯度计算。
                with torch.no_grad():
                    s_features_old, _, _, _ = old_model(
                        s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old = s_features_old[0]

                # ACSSD replaces full-space relation transfer with stronger
                # preservation in a fixed pedestrian-attribute contrast
                # subspace.  The complementary component receives current-ID
                # metric supervision and remains comparatively plastic.
                srsr = self._srsr_components(
                    s_features, s_features_old, training_phase)
                acssd = (None if srsr is not None else self._acssd_components(
                    s_features, s_features_old, training_phase))
                decomposition = srsr if srsr is not None else acssd
                if decomposition is None:
                    relation_features = s_features
                    relation_features_old = s_features_old
                    sa_relation, sa_confidence = self._build_sa_rstkt_relation(
                        s_features.detach(), s_features_old.detach(),
                        old_model, training_phase)
                    stable_scale = 1.0
                else:
                    relation_features, relation_features_old, residual = decomposition
                    sa_relation = sa_confidence = None
                    if srsr is not None:
                        stable_scale, plastic_scale = self._srsr_effective_scales()
                    else:
                        stable_scale, plastic_scale = self._acssd_effective_scales()
                    plastic_loss = self.acssd_triplet(
                        F.normalize(residual.float(), p=2, dim=1), targets)[0]
                    plastic_term = plastic_loss * plastic_scale
                    loss = loss + plastic_term
                    if srsr is not None:
                        losses_srsr_plastic.update(plastic_term.item())
                    else:
                        losses_acssd_plastic.update(plastic_term.item())

                # 将批内两两余弦相似度转为关系概率分布。
                affinity_new = self.get_normal_affinity(relation_features)
                affinity_old = self.get_normal_affinity(relation_features_old)
                # 保持新旧模型的可靠样本关系；启用 SA-RSTKT 时再做语义校准。
                divergence = self.cal_KL(
                    affinity_new,
                    affinity_old,
                    targets,
                    sa_rstkt_relation=sa_relation,
                    sa_rstkt_confidence=sa_confidence,
                )
                stable_term = divergence * self.AF_weight * stable_scale
                loss = loss + stable_term
                losses_af.update(divergence.item())
                if srsr is not None:
                    losses_srsr_stable.update(stable_term.item())
                elif acssd is not None:
                    losses_acssd_stable.update(stable_term.item())

            # EMA 只在 PSC identity-state 聚合后跟踪时序参考，
            # 不对逐样本的多视角输出施加一致性。
            sdac_loss = self._sdac_loss(
                s_inputs,
                s_features,
                cls_outputs,
                targets,
                training_phase,
            )
            if sdac_loss is not None:
                sdac_total, sdac_occupancy, sdac_visual, sdac_decision, sdac_states = sdac_loss
                loss = loss + sdac_total
                losses_sdac.update(sdac_total.item())
                losses_sdac_occupancy.update(sdac_occupancy.item())
                losses_sdac_visual.update(sdac_visual.item())
                losses_sdac_decision.update(sdac_decision.item())
                losses_sdac_states.update(float(sdac_states))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._update_sdac_teacher(training_phase)
            self._commit_sdac_state_memory(training_phase)

            batch_time.update(time.time() - end)
            end = time.time()
            self._write_scalars(
                epoch, train_iters, i, training_phase, batch_time,
                losses_ce, losses_tr, losses_af, losses_gscm,
                losses_sdac, losses_sdac_occupancy, losses_sdac_visual,
                losses_sdac_decision, losses_sdac_states,
                losses_acssd_stable, losses_acssd_plastic,
                losses_srsr_stable, losses_srsr_plastic,
            )

            if (i + 1) == train_iters:
                msg = ('Epoch: [{}][{}/{}]\t'
                       'Time {:.3f} ({:.3f})\t'
                       'Loss_ce {:.3f} ({:.3f})\t'
                       'Loss_tp {:.3f} ({:.3f})\t'
                       .format(epoch, i + 1, train_iters,
                               batch_time.val, batch_time.avg,
                               losses_ce.val, losses_ce.avg,
                               losses_tr.val, losses_tr.avg))
                if losses_af.count > 0:
                    msg += 'Loss_af {:.3f} ({:.3f})\t'.format(losses_af.val, losses_af.avg)
                    if self._sa_rstkt_enabled(training_phase):
                        msg += 'SA-RSTKT_w {:.3f}\t'.format(self._sa_rstkt_effective_weight())
                    if bool(getattr(self.args, 'psc_controller', False)):
                        msg += 'PSC_q {:.3f} PSC_AF_w {:.3f}\t'.format(
                            float(getattr(
                                self.args, 'psc_controller_current_similarity', 0.0)),
                            float(self.AF_weight))
                if losses_gscm.count > 0:
                    msg += 'Loss_gscm {:.4f} ({:.4f})\t'.format(losses_gscm.val, losses_gscm.avg)
                    msg += 'GSCM_w {:.3f}\t'.format(self._gscm_effective_weight())
                if losses_sdac.count > 0:
                    msg += 'Loss_sdac {:.4f} ({:.4f})\t'.format(
                        losses_sdac.val, losses_sdac.avg)
                    msg += (
                        'SDAC_occ/vis/dec {:.4f}/{:.4f}/{:.4f} '
                        '(avg {:.4f}/{:.4f}/{:.4f})\t'
                    ).format(
                        losses_sdac_occupancy.val,
                        losses_sdac_visual.val,
                        losses_sdac_decision.val,
                        losses_sdac_occupancy.avg,
                        losses_sdac_visual.avg,
                        losses_sdac_decision.avg,
                    )
                    msg += 'SDAC_states {:.0f}\t'.format(losses_sdac_states.val)
                    msg += 'SDAC_w {:.3f}\t'.format(self._sdac_effective_weight())
                if losses_acssd_stable.count > 0:
                    msg += 'ACSSD_stable/plastic {:.4f}/{:.4f}\t'.format(
                        losses_acssd_stable.val, losses_acssd_plastic.val)
                if losses_srsr_stable.count > 0:
                    msg += 'SRSR_stable/plastic {:.4f}/{:.4f}\t'.format(
                        losses_srsr_stable.val, losses_srsr_plastic.val)
                print(msg)

    # ---------------------------- SRSR ------------------------------
    def _srsr_enabled(self, training_phase):
        return (
            bool(getattr(self.args, 'srsr', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and getattr(self.args, 'semantic_reference', 'semantic') != 'none'
            and int(training_phase) >= int(getattr(
                self.args, 'srsr_start_phase', 2))
        )

    def _srsr_effective_scales(self):
        stable = max(float(getattr(self.args, 'srsr_stable_weight', 1.0)), 0.0)
        plastic = max(float(getattr(self.args, 'srsr_plastic_weight', 0.2)), 0.0)
        start = max(int(getattr(self.args, 'srsr_start_epoch', 5)), 0)
        warmup = max(int(getattr(self.args, 'srsr_warmup_epochs', 10)), 1)
        epoch = int(getattr(self, '_current_epoch', 0))
        if epoch < start:
            return 0.0, 0.0
        ramp = min(float(epoch - start + 1) / warmup, 1.0)
        return stable * ramp, plastic * ramp

    def _srsr_components(self, features, old_features, training_phase):
        if not self._srsr_enabled(training_phase):
            return None
        projector = getattr(
            self._model_ref(), 'project_role_routed_subspaces', None)
        if not callable(projector):
            raise RuntimeError('SRSR requires role-routed projection support')
        stable, residual = projector(features)
        stable_old, _ = projector(old_features)
        return (
            F.normalize(stable.float(), p=2, dim=1),
            F.normalize(stable_old.float(), p=2, dim=1),
            residual,
        )

    # ---------------------------- ACSSD -----------------------------
    def _acssd_enabled(self, training_phase):
        return (
            bool(getattr(self.args, 'acssd', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and getattr(self.args, 'semantic_reference', 'semantic') != 'none'
            and int(training_phase) >= int(getattr(
                self.args, 'acssd_start_phase', 2))
        )

    def _acssd_effective_scales(self):
        stable = max(float(getattr(self.args, 'acssd_stable_weight', 1.0)), 0.0)
        plastic = max(float(getattr(self.args, 'acssd_plastic_weight', 0.2)), 0.0)
        start = max(int(getattr(self.args, 'acssd_start_epoch', 5)), 0)
        warmup = max(int(getattr(self.args, 'acssd_warmup_epochs', 10)), 1)
        epoch = int(getattr(self, '_current_epoch', 0))
        if epoch < start:
            return 0.0, 0.0
        ramp = min(float(epoch - start + 1) / warmup, 1.0)
        return stable * ramp, plastic * ramp

    def _acssd_components(self, features, old_features, training_phase):
        if not self._acssd_enabled(training_phase):
            return None
        model_ref = self._model_ref()
        projector = getattr(model_ref, 'project_semantic_subspaces', None)
        if not callable(projector):
            raise RuntimeError('ACSSD requires semantic subspace projection support')
        energy = float(getattr(self.args, 'acssd_subspace_energy', 0.95))
        rank = int(getattr(self.args, 'acssd_subspace_rank', 0))
        construction = getattr(
            self.args, 'acssd_subspace_construction', 'group_centered')
        semantic, residual = projector(
            features, energy=energy, construction=construction, rank=rank)
        # The basis is fixed and shared, so projecting old features with the
        # current branch is equivalent and avoids any cache/reference mismatch.
        semantic_old, _ = projector(
            old_features, energy=energy, construction=construction, rank=rank)
        return (
            F.normalize(semantic.float(), p=2, dim=1),
            F.normalize(semantic_old.float(), p=2, dim=1),
            residual,
        )

    # ------------------------------ GSCM ------------------------------
    def _gscm_enabled(self, training_phase):
        """判断当前阶段是否满足启用 GSCM 的全部条件。"""
        return (
            bool(getattr(self.args, 'gscm', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and getattr(self.args, 'semantic_reference', 'semantic') != 'none'
            and float(getattr(self.args, 'gscm_weight', 0.0)) > 0.0
            and int(training_phase) >= int(getattr(self.args, 'gscm_start_phase', 1))
        )

    def _gscm_effective_weight(self):
        """按 epoch 线性预热 GSCM 权重。"""
        base_weight = max(float(getattr(self.args, 'gscm_weight', 0.1)), 0.0)
        start_epoch = max(int(getattr(self.args, 'gscm_start_epoch', 0)), 0)
        current_epoch = int(getattr(self, '_current_epoch', 0))
        if current_epoch < start_epoch:
            return 0.0
        warmup_epochs = max(int(getattr(self.args, 'gscm_warmup_epochs', 5)), 1)
        progress = min(float(current_epoch - start_epoch + 1) / warmup_epochs, 1.0)
        return base_weight * progress

    def _gscm_loss(self, features, targets, cameras, training_phase):
        """计算分组语义跨摄像头间隔损失。"""
        if not self._gscm_enabled(training_phase):
            return None
        loss_weight = self._gscm_effective_weight()
        if loss_weight <= 0.0:
            return None

        model_ref = self._model_ref()
        if not hasattr(model_ref, 'semantic_anchor_logits'):
            return None
        # 锚点分配只用于构造监督信号，无需对该分支反向传播。
        with torch.no_grad():
            anchor_logits = model_ref.semantic_anchor_logits(
                features.detach(),
                temperature=getattr(self.args, 'gscm_text_temp', 0.07),
            )
        if anchor_logits is None:
            return None

        # semantic 使用当前锚集合的语义分组；global/shuffled 用于对照实验。
        group_mode = getattr(self.args, 'gscm_anchor_groups', 'semantic')
        if group_mode == 'global':
            anchor_groups = getattr(model_ref, 'semantic_anchor_groups_global',
                                    GLOBAL_ANCHOR_GROUPS)
        elif group_mode == 'shuffled':
            anchor_groups = getattr(model_ref, 'semantic_anchor_groups_shuffled',
                                    SHUFFLED_ANCHOR_GROUPS)
        else:
            anchor_groups = getattr(model_ref, 'semantic_anchor_groups',
                                    DEFAULT_ANCHOR_GROUPS)
        if not anchor_groups:
            return None

        loss = grouped_semantic_cross_camera_margin_loss(
            features,
            targets,
            cameras,
            anchor_logits,
            visual_temperature=getattr(self.args, 'gscm_visual_temp', 0.07),
            base_margin=getattr(self.args, 'gscm_base_margin', 0.05),
            semantic_margin=getattr(self.args, 'gscm_semantic_margin', 0.10),
            groups=anchor_groups,
            use_entropy_confidence=bool(getattr(self.args, 'gscm_entropy_confidence', True)),
            use_cross_camera=bool(getattr(self.args, 'gscm_cross_camera', True)),
        )
        # 异常批次不向总损失传播 NaN/Inf。
        if not bool(torch.isfinite(loss).item()):
            return None
        return loss_weight * loss

    # ------------------------------ SDAC ------------------------------
    # Identity-Conditioned Reference State Alignment. Fixed PSC groups provide
    # the persistent coordinates and EMA follows the state trajectory within a
    # stage. All outputs are aggregated first, so no per-image consistency is
    # present in the objective.

    def _sdac_enabled(self, training_phase):
        return (
            bool(getattr(self.args, 'sdac', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and getattr(self.args, 'semantic_reference', 'semantic') != 'none'
            and float(getattr(self.args, 'sdac_weight', 0.0)) > 0.0
            and int(training_phase) >= int(getattr(self.args, 'sdac_start_phase', 2))
        )

    def _init_sdac_teacher(self, training_phase):
        if self.sdac_teacher is not None and self.sdac_teacher_phase == int(training_phase):
            return
        self.sdac_teacher = copy.deepcopy(self._model_ref()).cuda()
        self.sdac_teacher.eval()
        for parameter in self.sdac_teacher.parameters():
            parameter.requires_grad_(False)
        self.sdac_teacher_phase = int(training_phase)
        self.sdac_state_memory = {}
        self.sdac_state_memory_phase = int(training_phase)
        self._sdac_pending_state_update = None

    @torch.no_grad()
    def _update_sdac_teacher(self, training_phase):
        if not self._sdac_enabled(training_phase) or self.sdac_teacher is None:
            return
        momentum = min(max(float(getattr(self.args, 'sdac_ema', 0.997)), 0.0), 0.9999)
        student = self._model_ref()
        student_parameters = dict(student.named_parameters())
        student_buffers = dict(student.named_buffers())
        for name, teacher_parameter in self.sdac_teacher.named_parameters():
            current = student_parameters.get(name)
            if current is None or current.shape != teacher_parameter.shape:
                continue
            value = current.detach().to(teacher_parameter)
            if current.requires_grad:
                teacher_parameter.mul_(momentum).add_(value, alpha=1.0 - momentum)
            else:
                teacher_parameter.copy_(value)
        for name, teacher_buffer in self.sdac_teacher.named_buffers():
            current = student_buffers.get(name)
            if current is None or current.shape != teacher_buffer.shape:
                continue
            value = current.detach().to(teacher_buffer)
            if 'text_anchor_' in name or 'semantic_reference_' in name:
                teacher_buffer.copy_(value)
            elif torch.is_floating_point(teacher_buffer):
                teacher_buffer.mul_(momentum).add_(value, alpha=1.0 - momentum)
            else:
                teacher_buffer.copy_(value)
        self.sdac_teacher.eval()

    @staticmethod
    def _js_distribution(student, teacher, eps=1e-8):
        mixture = 0.5 * (student + teacher)
        return 0.5 * (
            (student * (student.clamp_min(eps).log() - mixture.clamp_min(eps).log())).sum()
            + (teacher * (teacher.clamp_min(eps).log() - mixture.clamp_min(eps).log())).sum()
        )

    def _sdac_effective_weight(self):
        base = max(float(getattr(self.args, 'sdac_weight', 0.2)), 0.0)
        warmup = max(int(getattr(self.args, 'sdac_warmup_epochs', 0)), 0)
        if warmup <= 1:
            return base
        epoch = max(int(getattr(self, '_current_epoch', 0)), 0)
        return base * min(1.0, float(epoch + 1) / float(warmup))

    def _sdac_coordinate_logits(self, model_ref, features, temperature):
        """Return SDAC-only coordinates without changing other PSC modules."""
        control = getattr(self.args, 'sdac_coordinate_control', 'semantic')
        if control == 'semantic':
            return model_ref.semantic_anchor_logits(
                features, temperature=temperature)

        anchors = model_ref.get_text_anchors()
        if anchors is None or anchors.numel() == 0:
            return None
        count = int(anchors.size(0))
        if control == 'uniform':
            # Group-wise softmax of zero logits is exactly uniform.  Keeping
            # the original slot count/grouping makes this a matched control.
            return features.new_zeros(features.size(0), count, dtype=torch.float32)
        if control != 'random_fixed':
            raise ValueError('unknown SDAC coordinate control: {}'.format(control))

        cached = self._sdac_random_coordinate_anchors
        if cached is None or cached.shape != (count, features.size(1)):
            # A dedicated CPU generator avoids changing training RNG streams.
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(getattr(
                self.args, 'sdac_coordinate_seed', 1701)))
            cached = F.normalize(torch.randn(
                count, features.size(1), generator=generator,
                dtype=torch.float32), p=2, dim=1)
            self._sdac_random_coordinate_anchors = cached
            print('SDAC coordinate control: random_fixed count={} seed={}'.format(
                count, int(getattr(self.args, 'sdac_coordinate_seed', 1701))))
        random_anchors = cached.to(
            device=features.device, dtype=features.dtype)
        normalized_features = F.normalize(features, p=2, dim=1)
        return normalized_features.matmul(random_anchors.t()).float() / max(
            float(temperature), 1e-6)

    @staticmethod
    def _sdac_sparse_assignment(probabilities, topk):
        width = int(probabilities.size(1))
        topk = int(topk)
        if topk <= 0 or topk >= width:
            return probabilities.detach()
        values, indices = probabilities.detach().topk(topk, dim=1)
        sparse = torch.zeros_like(probabilities).scatter_(1, indices, values)
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _sdac_normalize_distribution(distribution, eps=1e-8):
        distribution = distribution.clamp_min(0.0)
        return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _sdac_memory_targets(self, records, device,
                             visual_enabled=True, decision_enabled=True):
        """Batch CPU state records into one device transfer and target computation."""
        mode = getattr(self.args, 'sdac_state_memory_mode', 'fixed')
        if mode != 'dual':
            visual = None
            decision = None
            if visual_enabled:
                visual = torch.stack(
                    [record['visual'] for record in records]).to(
                        device=device, dtype=torch.float32)
            if decision_enabled:
                decision = torch.stack(
                    [record['decision'] for record in records]).to(
                        device=device, dtype=torch.float32)
            return visual, decision

        fast_visual = torch.stack(
            [record['fast_visual'] for record in records]).to(
                device=device, dtype=torch.float32)
        slow_visual = torch.stack(
            [record['slow_visual'] for record in records]).to(
                device=device, dtype=torch.float32)
        fast_decision = torch.stack(
            [record['fast_decision'] for record in records]).to(
                device=device, dtype=torch.float32)
        slow_decision = torch.stack(
            [record['slow_decision'] for record in records]).to(
                device=device, dtype=torch.float32)
        visual_drift = 1.0 - F.cosine_similarity(
            fast_visual, slow_visual, dim=1)
        decision_drift = 0.5 * (fast_decision - slow_decision).abs().sum(dim=1)
        drift_scale = max(float(
            getattr(self.args, 'sdac_state_drift_scale', 0.10)), 1e-6)
        fast_weight = ((visual_drift + decision_drift) * 0.5 / drift_scale).clamp(0.0, 1.0)
        visual = F.normalize(
            slow_visual * (1.0 - fast_weight[:, None])
            + fast_visual * fast_weight[:, None], p=2, dim=1)
        decision = self._sdac_normalize_distribution(
            slow_decision * (1.0 - fast_weight[:, None])
            + fast_decision * fast_weight[:, None])
        return visual, decision

    def _sdac_cross_batch_loss(self, feature_student, feature_teacher,
                               decision_student, decision_teacher,
                               assignment, targets, group_slices):
        """Match current identity-semantic states to pre-batch EMA memories."""
        eps = 1e-8
        device = feature_student.device
        visual_loss_enabled = float(getattr(
            self.args, 'sdac_visual_state_weight', 0.25)) > 0.0
        decision_loss_enabled = float(getattr(
            self.args, 'sdac_decision_state_weight', 0.5)) > 0.0
        memory_mode = getattr(self.args, 'sdac_state_memory_mode', 'fixed')
        # A dual target uses visual and decision drift jointly. Fixed/adaptive
        # memories can omit a branch whose loss weight is exactly zero.
        memory_visual_enabled = (
            visual_loss_enabled
            or (memory_mode == 'dual' and decision_loss_enabled))
        memory_decision_enabled = (
            decision_loss_enabled
            or (memory_mode == 'dual' and visual_loss_enabled))
        if not visual_loss_enabled and not decision_loss_enabled:
            self._sdac_pending_state_update = None
            zero = feature_student.sum() * 0.0
            return zero, zero, 0
        scope = getattr(self.args, 'sdac_state_scope', 'identity')
        if scope == 'global':
            identity_values = torch.full(
                (1,), -1, device=device, dtype=targets.dtype)
            membership = torch.ones(
                1, targets.numel(), device=device, dtype=assignment.dtype)
        else:
            identity_values, inverse = targets.unique(sorted=False, return_inverse=True)
            membership = F.one_hot(
                inverse, num_classes=identity_values.numel()).t().to(
                    device=device, dtype=assignment.dtype)

        weights = membership.unsqueeze(2) * assignment.unsqueeze(0)
        weights_t = weights.transpose(1, 2)
        mass = weights.sum(dim=1)
        safe_mass = mass.unsqueeze(2).clamp_min(eps)
        update_mass = max(float(
            getattr(self.args, 'sdac_state_update_mass', 0.50)), eps)
        valid = mass >= update_mass

        student_visual = None
        teacher_visual = None
        student_decision = None
        teacher_decision = None
        if visual_loss_enabled:
            student_visual = F.normalize(
                weights_t.matmul(feature_student), p=2, dim=2)
        if memory_visual_enabled:
            teacher_visual = F.normalize(
                weights_t.matmul(feature_teacher), p=2, dim=2)
        if decision_loss_enabled:
            student_decision = self._sdac_normalize_distribution(
                weights_t.matmul(decision_student) / safe_mass)
        if memory_decision_enabled:
            teacher_decision = self._sdac_normalize_distribution(
                weights_t.matmul(decision_teacher) / safe_mass)

        # Queue detached teacher statistics now, but do not mutate the memory
        # until the current optimization step and parameter EMA are complete.
        valid_indices = valid.nonzero(as_tuple=False)
        valid_identity_index = valid_indices[:, 0]
        valid_slot_index = valid_indices[:, 1]

        # One batched device-to-host transfer replaces per-state .item()/.cpu()
        # calls, which otherwise serialize the accelerator roughly 130 times
        # per identity-conditioned batch.
        valid_pairs_cpu = valid_indices.detach().cpu()
        identity_values_cpu = identity_values.detach().cpu()
        valid_mass_cpu = mass[
            valid_identity_index, valid_slot_index].detach().cpu()
        teacher_visual_cpu = None
        teacher_decision_cpu = None
        if teacher_visual is not None:
            teacher_visual_cpu = teacher_visual[
                valid_identity_index, valid_slot_index].detach().cpu()
        if teacher_decision is not None:
            teacher_decision_cpu = teacher_decision[
                valid_identity_index, valid_slot_index].detach().cpu()
        valid_pairs = valid_pairs_cpu.tolist()
        keys = [
            (int(identity_values_cpu[identity_index]), int(slot_index))
            for identity_index, slot_index in valid_pairs
        ]
        pending = [
            (key, float(valid_mass_cpu[position]),
             (teacher_visual_cpu[position]
              if teacher_visual_cpu is not None else None),
             (teacher_decision_cpu[position]
              if teacher_decision_cpu is not None else None))
            for position, key in enumerate(keys)
        ]
        self._sdac_pending_state_update = pending

        memory_positions = []
        memory_records = []
        for position, key in enumerate(keys):
            record = self.sdac_state_memory.get(key)
            if record is not None:
                memory_positions.append(position)
                memory_records.append(record)

        zero = feature_student.sum() * 0.0
        if not memory_positions:
            return zero, zero, 0

        # Gather all historical targets on CPU, then perform one transfer for
        # each state type. This preserves the exact memory semantics without
        # launching one copy/kernel per dictionary entry.
        target_visual, target_decision = self._sdac_memory_targets(
            memory_records, device,
            visual_enabled=visual_loss_enabled,
            decision_enabled=decision_loss_enabled)
        position_index = torch.as_tensor(
            memory_positions, device=device, dtype=torch.long)
        identity_index = valid_identity_index.index_select(0, position_index)
        slot_index = valid_slot_index.index_select(0, position_index)
        term_masses = mass[identity_index, slot_index]

        if visual_loss_enabled:
            current_visual = student_visual[identity_index, slot_index]
            visual_terms = 1.0 - F.cosine_similarity(
                current_visual, target_visual, dim=1)
        else:
            visual_terms = term_masses * 0.0
        if decision_loss_enabled:
            current_decision = student_decision[identity_index, slot_index]
            decision_terms = (
                target_decision * (
                    target_decision.clamp_min(eps).log()
                    - current_decision.clamp_min(eps).log())
            ).sum(dim=1)
        else:
            decision_terms = term_masses * 0.0

        slot_to_group = torch.empty(
            assignment.size(1), device=device, dtype=torch.long)
        for group_index, (start, end) in enumerate(group_slices):
            slot_to_group[start:end] = group_index

        # Preserve equal weighting of active identity-attribute groups while
        # weighting slots inside a group by current assignment mass. Scatter
        # reduction removes the former nested Python list/set loops.
        group_ids = (
            identity_index * len(group_slices)
            + slot_to_group.index_select(0, slot_index))
        _, inverse_group = torch.unique(
            group_ids, sorted=False, return_inverse=True)
        group_count = int(inverse_group.max().detach().item()) + 1
        group_mass = term_masses.new_zeros(group_count)
        group_visual = visual_terms.new_zeros(group_count)
        group_decision = decision_terms.new_zeros(group_count)
        group_mass.scatter_add_(0, inverse_group, term_masses)
        group_visual.scatter_add_(
            0, inverse_group, visual_terms * term_masses)
        group_decision.scatter_add_(
            0, inverse_group, decision_terms * term_masses)
        visual_sum = (group_visual / group_mass.clamp_min(eps)).sum()
        decision_sum = (group_decision / group_mass.clamp_min(eps)).sum()
        temperature_l = max(float(
            getattr(self.args, 'sdac_logit_temp', 4.0)), 1e-6)
        return (
            visual_sum / max(group_count, 1),
            decision_sum / max(group_count, 1) * (temperature_l * temperature_l),
            int(position_index.numel()),
        )

    @torch.no_grad()
    def _commit_sdac_state_memory(self, training_phase):
        """Update CPU state memories after the current student step."""
        pending = self._sdac_pending_state_update
        self._sdac_pending_state_update = None
        if (not pending
                or self.sdac_state_memory_phase != int(training_phase)
                or getattr(self.args, 'sdac_objective', 'state')
                not in ('cross_batch_state', 'cross_batch_identity')):
            return
        mode = getattr(self.args, 'sdac_state_memory_mode', 'fixed')
        fixed_momentum = min(max(float(
            getattr(self.args, 'sdac_state_ema', 0.90)), 0.0), 0.9999)
        adaptive_min = min(max(float(
            getattr(self.args, 'sdac_state_ema_min', 0.80)), 0.0), 0.9999)
        adaptive_max = min(max(float(
            getattr(self.args, 'sdac_state_ema_max', 0.995)), adaptive_min), 0.9999)
        fast_momentum = min(max(float(
            getattr(self.args, 'sdac_state_fast_ema', 0.80)), 0.0), 0.9999)
        slow_momentum = min(max(float(
            getattr(self.args, 'sdac_state_slow_ema', 0.99)), 0.0), 0.9999)
        quality_mass = max(float(
            getattr(self.args, 'sdac_state_quality_mass', 2.0)), 1e-6)

        for key, mass, visual, decision in pending:
            if visual is not None:
                visual = F.normalize(visual.float(), p=2, dim=0)
            if decision is not None:
                decision = self._sdac_normalize_distribution(decision.float())
            record = self.sdac_state_memory.get(key)
            if record is None:
                if mode == 'dual':
                    if visual is None or decision is None:
                        raise RuntimeError(
                            'dual state memory requires visual and decision states')
                    self.sdac_state_memory[key] = {
                        'fast_visual': visual.clone(),
                        'slow_visual': visual.clone(),
                        'fast_decision': decision.clone(),
                        'slow_decision': decision.clone(),
                        'updates': 1,
                    }
                else:
                    new_record = {'updates': 1}
                    if visual is not None:
                        new_record['visual'] = visual.clone()
                    if decision is not None:
                        new_record['decision'] = decision.clone()
                    self.sdac_state_memory[key] = new_record
                continue

            reliability = min(max(float(mass) / quality_mass, 0.0), 1.0)
            if mode == 'adaptive':
                momentum = adaptive_max - reliability * (
                    adaptive_max - adaptive_min)
            else:
                momentum = fixed_momentum

            if mode == 'dual':
                for prefix, momentum_value in (
                        ('fast', fast_momentum), ('slow', slow_momentum)):
                    updated_visual = (
                        record[prefix + '_visual'] * momentum_value
                        + visual * (1.0 - momentum_value))
                    record[prefix + '_visual'] = F.normalize(
                        updated_visual, p=2, dim=0)
                    updated_decision = (
                        record[prefix + '_decision'] * momentum_value
                        + decision * (1.0 - momentum_value))
                    record[prefix + '_decision'] = self._sdac_normalize_distribution(
                        updated_decision)
            else:
                if visual is not None:
                    record['visual'] = F.normalize(
                        record['visual'] * momentum
                        + visual * (1.0 - momentum), p=2, dim=0)
                if decision is not None:
                    record['decision'] = self._sdac_normalize_distribution(
                        record['decision'] * momentum
                        + decision * (1.0 - momentum))
            record['updates'] += 1

    def _sdac_loss(self, inputs, features, logits, targets, training_phase):
        """Align EMA outputs with a selectable attribution-control objective."""
        if not self._sdac_enabled(training_phase):
            return None
        self._sdac_pending_state_update = None
        self._init_sdac_teacher(training_phase)
        with torch.no_grad():
            teacher_features, _, teacher_logits, _ = self.sdac_teacher(
                inputs, get_all_feat=True)
            if isinstance(teacher_features, tuple):
                teacher_features = teacher_features[0]

        model_ref = self._model_ref()
        temperature_a = getattr(self.args, 'sdac_text_temp', 0.07)
        student_anchor_logits = self._sdac_coordinate_logits(
            model_ref, features, temperature=temperature_a)
        with torch.no_grad():
            teacher_anchor_logits = self._sdac_coordinate_logits(
                model_ref, teacher_features, temperature=temperature_a)
        if student_anchor_logits is None or teacher_anchor_logits is None:
            raise RuntimeError('SDAC requires aligned PSC responses')

        feature_student = F.normalize(features.float(), p=2, dim=1)
        feature_teacher = F.normalize(
            teacher_features.detach().float(), p=2, dim=1)
        temperature_l = max(float(
            getattr(self.args, 'sdac_logit_temp', 4.0)), 1e-6)
        if logits is None or teacher_logits is None or logits.shape != teacher_logits.shape:
            raise RuntimeError('SDAC requires aligned student/EMA classifier outputs')
        decision_student = F.softmax(logits.float() / temperature_l, dim=1)
        decision_teacher = F.softmax(
            teacher_logits.detach().float() / temperature_l, dim=1)

        anchor_groups = getattr(
            model_ref, 'semantic_anchor_groups', DEFAULT_ANCHOR_GROUPS)
        if not anchor_groups:
            raise RuntimeError('SDAC requires nonempty semantic anchor groups')
        topk = int(getattr(self.args, 'sdac_assignment_topk', 2))
        min_mass = max(float(
            getattr(self.args, 'sdac_min_state_mass', 0.25)), 0.0)
        dispersion_weight = max(float(
            getattr(self.args, 'sdac_dispersion_weight', 0.5)), 0.0)
        objective = getattr(self.args, 'sdac_objective', 'state')
        state_scope = getattr(self.args, 'sdac_state_scope', 'identity')
        eps = 1e-8
        zero = features.new_zeros(())

        student_group_probs = []
        teacher_group_probs = []
        teacher_group_assignments = []
        group_slices = []
        offset = 0
        for group in anchor_groups:
            indices = torch.as_tensor(
                group, device=features.device, dtype=torch.long)
            student_prob = F.softmax(
                student_anchor_logits.index_select(1, indices).float(), dim=1)
            teacher_prob = F.softmax(
                teacher_anchor_logits.index_select(1, indices).detach().float(), dim=1)
            student_group_probs.append(student_prob)
            teacher_group_probs.append(teacher_prob)
            teacher_group_assignments.append(
                self._sdac_sparse_assignment(teacher_prob, topk=topk))
            group_slices.append((offset, offset + len(group)))
            offset += len(group)

        p_student = torch.cat(student_group_probs, dim=1)
        p_teacher = torch.cat(teacher_group_probs, dim=1)
        assignment = torch.cat(teacher_group_assignments, dim=1)

        if objective == 'cross_batch_identity':
            # Strict identity-only control: retain the same cross-batch EMA
            # decision memory, but collapse all semantic groups and slots into
            # one state per identity. PSC responses are still computed above
            # to keep the forward path matched, but do not affect addressing.
            assignment = torch.ones(
                features.size(0), 1, device=features.device,
                dtype=assignment.dtype)
            group_slices = [(0, 1)]

        if objective in ('cross_batch_state', 'cross_batch_identity'):
            loss_visual, loss_decision, valid_state_count = (
                self._sdac_cross_batch_loss(
                    feature_student, feature_teacher,
                    decision_student, decision_teacher,
                    assignment, targets, group_slices))
            # Cross-batch memory replaces the batch occupancy comparison;
            # retaining an identically-zero diagnostic makes logs/tables
            # compatible with existing SDAC experiments.
            loss_occupancy = zero

        elif objective == 'instance':
            # Same EMA teacher, semantic groups, output spaces and weights as
            # the full method, but no identity or state aggregation.
            occupancy_losses = []
            for student_group, teacher_group in zip(
                    student_group_probs, teacher_group_probs):
                mixture = 0.5 * (student_group + teacher_group)
                occupancy_losses.append(0.5 * (
                    student_group * (
                        student_group.clamp_min(eps).log()
                        - mixture.clamp_min(eps).log())
                    + teacher_group * (
                        teacher_group.clamp_min(eps).log()
                        - mixture.clamp_min(eps).log())
                ).sum(dim=1))
            loss_occupancy = torch.stack(occupancy_losses, dim=1).mean()
            loss_visual = (1.0 - F.cosine_similarity(
                feature_student, feature_teacher, dim=1)).mean()
            loss_decision = (
                decision_teacher * (
                    decision_teacher.clamp_min(eps).log()
                    - decision_student.clamp_min(eps).log())
            ).sum(dim=1).mean() * (temperature_l * temperature_l)
            valid_state_count = int(features.size(0))
        else:
            if objective == 'state' and state_scope == 'global':
                # Preserve semantic states but remove identity conditioning.
                identity_membership = torch.ones(
                    1, features.size(0), device=features.device,
                    dtype=p_student.dtype)
            else:
                identities, inverse_identity = targets.unique(
                    sorted=False, return_inverse=True)
                identity_membership = F.one_hot(
                    inverse_identity, num_classes=identities.numel()).t().to(
                        device=features.device, dtype=p_student.dtype)
            identity_count = identity_membership.sum(
                dim=1, keepdim=True).clamp_min(1.0)

            occupancy_student = identity_membership.matmul(
                p_student) / identity_count
            occupancy_teacher = identity_membership.matmul(
                p_teacher) / identity_count
            occupancy_losses = []
            for start, end in group_slices:
                student_group = occupancy_student[:, start:end]
                teacher_group = occupancy_teacher[:, start:end]
                mixture = 0.5 * (student_group + teacher_group)
                occupancy_losses.append(0.5 * (
                    student_group * (
                        student_group.clamp_min(eps).log()
                        - mixture.clamp_min(eps).log())
                    + teacher_group * (
                        teacher_group.clamp_min(eps).log()
                        - mixture.clamp_min(eps).log())
                ).sum(dim=1))
            loss_occupancy = torch.stack(occupancy_losses, dim=1).mean()

            if objective == 'identity_prototype':
                # Keep identity aggregation, but collapse semantic states.
                visual_student_raw = identity_membership.matmul(feature_student)
                visual_teacher_raw = identity_membership.matmul(feature_teacher)
                visual_student_proto = F.normalize(
                    visual_student_raw, p=2, dim=1)
                visual_teacher_proto = F.normalize(
                    visual_teacher_raw, p=2, dim=1)
                centroid_distance = 1.0 - F.cosine_similarity(
                    visual_student_proto, visual_teacher_proto, dim=1)
                count = identity_count.squeeze(1)
                spread_student = 1.0 - (
                    visual_student_raw.norm(p=2, dim=1) / count)
                spread_teacher = 1.0 - (
                    visual_teacher_raw.norm(p=2, dim=1) / count)
                loss_visual = (
                    centroid_distance + dispersion_weight * (
                        spread_student - spread_teacher).abs()).mean()

                prototype_student = (
                    identity_membership.matmul(decision_student) / identity_count)
                prototype_teacher = (
                    identity_membership.matmul(decision_teacher) / identity_count)
                prototype_student = prototype_student / prototype_student.sum(
                    dim=1, keepdim=True).clamp_min(eps)
                prototype_teacher = prototype_teacher / prototype_teacher.sum(
                    dim=1, keepdim=True).clamp_min(eps)
                loss_decision = (
                    prototype_teacher * (
                        prototype_teacher.clamp_min(eps).log()
                        - prototype_student.clamp_min(eps).log())
                ).sum(dim=1).mean() * (temperature_l * temperature_l)
                valid_state_count = int(identity_membership.size(0))
            else:
                state_weights = (
                    identity_membership.unsqueeze(2) * assignment.unsqueeze(0))
                state_weights_t = state_weights.transpose(1, 2)
                mass = state_weights.sum(dim=1)
                valid = mass >= max(min_mass, eps)
                safe_mass = mass.unsqueeze(2).clamp_min(eps)

                visual_student_raw = state_weights_t.matmul(feature_student)
                visual_teacher_raw = state_weights_t.matmul(feature_teacher)
                visual_student = F.normalize(visual_student_raw, p=2, dim=2)
                visual_teacher = F.normalize(visual_teacher_raw, p=2, dim=2)
                centroid_distance = 1.0 - F.cosine_similarity(
                    visual_student, visual_teacher, dim=2)
                spread_student = 1.0 - (
                    visual_student_raw.norm(p=2, dim=2)
                    / mass.clamp_min(eps))
                spread_teacher = 1.0 - (
                    visual_teacher_raw.norm(p=2, dim=2)
                    / mass.clamp_min(eps))
                visual_state_loss = centroid_distance + dispersion_weight * (
                    spread_student - spread_teacher).abs()

                state_student = (
                    state_weights_t.matmul(decision_student) / safe_mass)
                state_teacher = (
                    state_weights_t.matmul(decision_teacher) / safe_mass)
                state_student = state_student / state_student.sum(
                    dim=2, keepdim=True).clamp_min(eps)
                state_teacher = state_teacher / state_teacher.sum(
                    dim=2, keepdim=True).clamp_min(eps)
                decision_state_loss = (
                    state_teacher * (
                        state_teacher.clamp_min(eps).log()
                        - state_student.clamp_min(eps).log())
                ).sum(dim=2) * (temperature_l * temperature_l)

                visual_sum = zero
                decision_sum = zero
                active_group_count = zero
                for start, end in group_slices:
                    active_mass = (
                        mass[:, start:end]
                        * valid[:, start:end].to(mass.dtype))
                    active_total = active_mass.sum(dim=1)
                    active_group = active_total > eps
                    slot_weight = (
                        active_mass
                        / active_total.unsqueeze(1).clamp_min(eps))
                    active_float = active_group.to(mass.dtype)
                    visual_sum = visual_sum + (
                        (visual_state_loss[:, start:end] * slot_weight)
                        .sum(dim=1) * active_float).sum()
                    decision_sum = decision_sum + (
                        (decision_state_loss[:, start:end] * slot_weight)
                        .sum(dim=1) * active_float).sum()
                    active_group_count = (
                        active_group_count + active_float.sum())
                loss_visual = visual_sum / active_group_count.clamp_min(1.0)
                loss_decision = decision_sum / active_group_count.clamp_min(1.0)
                valid_state_count = int(valid.detach().sum().item())

        occupancy_weight = max(float(
            getattr(self.args, 'sdac_occupancy_weight', 1.0)), 0.0)
        visual_weight = max(float(
            getattr(self.args, 'sdac_visual_state_weight', 0.25)), 0.0)
        decision_weight = max(float(
            getattr(self.args, 'sdac_decision_state_weight', 0.5)), 0.0)
        total = self._sdac_effective_weight() * (
            occupancy_weight * loss_occupancy
            + visual_weight * loss_visual
            + decision_weight * loss_decision)
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(
                'non-finite SDAC loss: objective={} scope={} occupancy={} '
                'visual={} decision={}'.format(
                    objective, state_scope,
                    float(loss_occupancy.detach().item()),
                    float(loss_visual.detach().item()),
                    float(loss_decision.detach().item())))
        return total, loss_occupancy, loss_visual, loss_decision, valid_state_count

    # ---------------------------- SA-RSTKT ----------------------------
    def _sa_rstkt_enabled(self, training_phase):
        """判断当前阶段是否启用语义感知关系知识迁移。"""
        return (
            bool(getattr(self.args, 'sa_rstkt', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and getattr(self.args, 'semantic_reference', 'semantic') != 'none'
            and float(getattr(self.args, 'sa_rstkt_weight', 0.0)) > 0.0
            and int(training_phase) >= int(getattr(self.args, 'sa_rstkt_start_phase', 2))
        )

    def _sa_rstkt_effective_weight(self):
        """计算带延迟启动和线性预热的 SA-RSTKT 强度。"""
        if not bool(getattr(self.args, 'sa_rstkt', False)):
            return 0.0
        base_weight = float(getattr(self.args, 'sa_rstkt_weight', 0.08))
        if base_weight <= 0.0:
            return 0.0
        epoch = int(getattr(self, '_current_epoch', 0))
        start_epoch = int(getattr(self.args, 'sa_rstkt_start_epoch', 5))
        if epoch < start_epoch:
            return 0.0
        warmup_epochs = int(getattr(self.args, 'sa_rstkt_warmup_epochs', 10))
        ramp = 1.0 if warmup_epochs <= 1 else min(1.0, float(epoch - start_epoch + 1) / warmup_epochs)
        return min(base_weight * ramp, 0.95)

    def _semantic_anchor_distribution(self, model, features, temperature=0.07):
        """返回 L2 归一化的语义锚点分布及逐样本最大置信度。"""
        model_ref = getattr(model, 'module', model)
        if (getattr(model_ref, 'semantic_reference', 'semantic') == 'none'
                or not hasattr(model_ref, 'semantic_anchor_logits')):
            return None, None
        logits = model_ref.semantic_anchor_logits(features, temperature=temperature)
        if logits is None:
            return None, None
        probs = F.softmax(logits.float(), dim=1)
        confidence = probs.max(dim=1)[0]
        probs = F.normalize(probs, p=2, dim=1)
        return probs, confidence

    def _pair_confidence(self, confidence, relation):
        """把逐样本置信度转换为与关系矩阵同形状的成对置信度。"""
        if confidence is None:
            return torch.ones_like(relation)
        if confidence.dim() == 1 and confidence.numel() == relation.size(0):
            confidence = confidence[:, None] * confidence[None, :]
        if confidence.shape != relation.shape:
            return torch.ones_like(relation)
        return confidence.float().clamp(0.0, 1.0)

    def _set_relation_eye(self, relation, confidence=None):
        """将关系矩阵及置信度矩阵的对角线固定为 1。"""
        eye = torch.eye(relation.size(0), device=relation.device)
        relation = relation * (1.0 - eye) + eye
        if confidence is None:
            return relation.clamp(0.0, 1.0), None
        confidence = confidence * (1.0 - eye) + eye
        return relation.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0)

    def _build_sa_rstkt_relation(self, features_new, features_old, old_model, training_phase):
        """融合新旧模型的语义锚点分布，构造批内语义关系与置信度。"""
        with torch.no_grad():
            if not self._sa_rstkt_enabled(training_phase):
                return None, None
            temp = getattr(self.args, 'sa_rstkt_text_temp', getattr(self.args, 'sdac_text_temp', 0.07))
            probs_new, conf_new = self._semantic_anchor_distribution(self.model, features_new, temperature=temp)
            probs_old, conf_old = self._semantic_anchor_distribution(old_model, features_old, temperature=temp)
            probs = [p for p in (probs_new, probs_old) if p is not None]
            confs = [c for c in (conf_new, conf_old) if c is not None]
            if not probs or not confs:
                return None, None

            # 分布内积表示两样本的语义接近程度，再对新旧模型结果取平均。
            relation = torch.stack([p @ p.t() for p in probs], dim=0).mean(dim=0)
            confidence = torch.stack(confs, dim=0).mean(dim=0)
            confidence = self._pair_confidence(confidence, relation)

            floor = float(getattr(self.args, 'sa_rstkt_conf_floor', 0.05))
            confidence = confidence.clamp_min(max(floor, 0.0))
            return self._set_relation_eye(relation, confidence)

    def _sa_rstkt_reliability(self, affinity_new, affinity_old, gts,
                              semantic_relation, semantic_confidence):
        """综合旧模型置信度、新旧一致性和语义一致性，估计关系可靠度。"""
        eps = 1e-8
        eye = torch.eye(affinity_new.size(0), device=affinity_new.device)
        old_confidence = affinity_old.float() / affinity_old.float().max(dim=1, keepdim=True)[0].clamp_min(eps)
        old_confidence = old_confidence.clamp(0.0, 1.0)

        # 新旧关系越接近，consistency 越大。
        consistency = 1.0 - (affinity_new.float() - affinity_old.float()).abs() / (
            affinity_new.float() + affinity_old.float() + eps)
        consistency = consistency.clamp(0.0, 1.0)
        power = max(float(getattr(self.args, 'sa_rstkt_consistency_power', 1.0)), 0.0)
        if power != 1.0:
            consistency = consistency.pow(power)

        same_id = gts * (1.0 - eye)
        diff_id = (1.0 - gts) * (1.0 - eye)
        semantic_relation = semantic_relation.detach().to(affinity_new.device).float().clamp(0.0, 1.0)
        # 同身份期望语义相似，异身份期望语义不相似。
        semantic_agreement = same_id * semantic_relation + diff_id * (1.0 - semantic_relation) + eye

        if semantic_confidence is None or semantic_confidence.shape != affinity_new.shape:
            confidence = torch.ones_like(affinity_new)
        else:
            confidence = semantic_confidence.detach().to(affinity_new.device).float().clamp(0.0, 1.0)

        floor = max(float(getattr(self.args, 'sa_rstkt_conf_floor', 0.05)), 0.0)
        reliability = old_confidence * consistency * semantic_agreement * confidence
        reliability = reliability * (1.0 - eye) + eye
        return reliability.clamp_min(floor).clamp(0.0, 1.0)

    def _apply_sa_rstkt_calibration(self, target_relation, gts,
                                    affinity_new, affinity_old,
                                    semantic_relation, semantic_confidence):
        """用语义关系校准蒸馏目标，并返回每个样本对的可靠性权重。"""
        if semantic_relation is None or semantic_relation.shape != target_relation.shape:
            return target_relation, None
        if semantic_confidence is not None and semantic_confidence.shape != target_relation.shape:
            semantic_confidence = None
        strength = self._sa_rstkt_effective_weight()
        if strength <= 0.0:
            return target_relation, None

        relation = semantic_relation.detach().to(target_relation.device).float().clamp(0.0, 1.0)
        confidence = torch.ones_like(target_relation) if semantic_confidence is None else semantic_confidence.detach().to(target_relation.device).float().clamp(0.0, 1.0)
        eye = torch.eye(target_relation.size(0), device=target_relation.device)
        same_id = gts * (1.0 - eye)
        diff_id = (1.0 - gts) * (1.0 - eye)

        semantic_weight = min(max(float(getattr(self.args, 'sa_rstkt_semantic_weight', 0.5)), 0.0), 1.0)
        neg_weight = min(max(float(getattr(self.args, 'sa_rstkt_neg_weight', 0.2)), 0.0), 1.0)

        # 拉高语义一致的正样本关系，压低语义相似的异身份错误关系。
        calibrated = target_relation.float()
        calibrated = calibrated + same_id * strength * semantic_weight * confidence * relation * (1.0 - calibrated)
        calibrated = calibrated - diff_id * strength * neg_weight * confidence * relation * calibrated
        calibrated = calibrated.clamp_min(1e-8)
        calibrated = calibrated * (1.0 - eye) + target_relation * eye

        reliability = self._sa_rstkt_reliability(
            affinity_new, affinity_old, gts, relation, confidence)
        return calibrated, reliability

    def _weighted_relation_kl(self, affinity_new, target, reliability=None):
        """计算关系 KL；有可靠性矩阵时先对新分布和目标分布重加权。"""
        eps = 1e-12
        if reliability is None:
            return self.KLDivLoss(affinity_new.clamp_min(eps).log(), target.detach().clamp_min(eps))

        weights = reliability.detach().to(affinity_new.device).float().clamp_min(eps)
        weighted_target = target.detach().float() * weights
        weighted_new = affinity_new.float() * weights
        weighted_target = weighted_target / weighted_target.sum(1, keepdim=True).clamp_min(eps)
        weighted_new = weighted_new / weighted_new.sum(1, keepdim=True).clamp_min(eps)
        return F.kl_div(
            weighted_new.clamp_min(eps).log(),
            weighted_target.clamp_min(eps),
            reduction='batchmean',
        )

    def cal_KL(self, Affinity_matrix_new, Affinity_matrix_old, targets,
               sa_rstkt_relation=None, sa_rstkt_confidence=None):
        """构造关系知识迁移目标，并计算新模型相对该目标的 KL 损失。"""
        gts = targets.reshape(-1, 1).eq(targets.reshape(1, -1)).float().to(targets.device)
        attri_new = self.get_attri(gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(gts, Affinity_matrix_old, margin=0)

        # 旧模型判断正确的关系直接保留。
        old_keep = attri_old['TN'] + attri_old['TP']
        target_1 = Affinity_matrix_old * old_keep
        # 旧模型出错而新模型正确时，采用新模型当前的关系值。
        new_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        target_2 = Affinity_matrix_new * new_keep
        # 新旧模型都漏判的正样本，用更严格（更大）的正阈值修正。
        hard_pos = attri_new['FN'] * attri_old['FN']
        thres_p = torch.maximum(attri_new['Thres_P'], attri_old['Thres_P'])
        target_3 = hard_pos * thres_p
        # 新旧模型都误判的负样本，用更严格（更小）的负阈值修正。
        hard_neg = attri_new['FP'] * attri_old['FP']
        thres_n = torch.minimum(attri_new['Thres_N'], attri_old['Thres_N'])
        target_4 = hard_neg * thres_n

        target = target_1 + target_2 + target_3 + target_4
        target, reliability = self._apply_sa_rstkt_calibration(
            target,
            gts,
            Affinity_matrix_new,
            Affinity_matrix_old,
            sa_rstkt_relation,
            sa_rstkt_confidence,
        )
        target = target / target.sum(1, keepdim=True).clamp_min(1e-12)
        return self._weighted_relation_kl(Affinity_matrix_new, target, reliability)

    def get_attri(self, Gts, pre_affinity_matrix, margin=0):
        """按批内自适应阈值将关系划分为 TP、FN、FP 和 TN。"""
        # 每一行中最相似的负样本分数作为正样本判定阈值。
        thres_p = ((1 - Gts) * pre_affinity_matrix).max(dim=1, keepdim=True)[0]
        t_scores = pre_affinity_matrix * Gts
        tp = ((t_scores - thres_p) > margin).float()
        tp = torch.maximum(tp, torch.eye(tp.size(0), device=tp.device))
        fn = Gts - tp

        # 每一行中最不相似的正样本分数作为负样本判定阈值。
        mapped_affinity = (1 - Gts) + pre_affinity_matrix
        mapped_affinity = mapped_affinity + torch.eye(mapped_affinity.size(0), device=mapped_affinity.device)
        thres_n = mapped_affinity.min(dim=1, keepdim=True)[0]
        n_scores = pre_affinity_matrix * (1 - Gts)
        fp = (n_scores > thres_n).float()
        tn = (1 - Gts) - fp
        return {
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'Thres_P': thres_p,
            'Thres_N': thres_n,
        }

    # ---------------------------- 通用工具 ----------------------------
    def _write_scalars(self, epoch, train_iters, i, training_phase, batch_time,
                       losses_ce, losses_tr, losses_af, losses_gscm,
                       losses_sdac, losses_sdac_occupancy, losses_sdac_visual,
                       losses_sdac_decision, losses_sdac_states,
                       losses_acssd_stable, losses_acssd_plastic,
                       losses_srsr_stable, losses_srsr_plastic):
        """把当前迭代的损失和耗时写入 TensorBoard。"""
        if self.writer is None:
            return
        step = epoch * train_iters + i
        self.writer.add_scalar('loss/Loss_ce_{}'.format(training_phase), losses_ce.val, step)
        self.writer.add_scalar('loss/Loss_tr_{}'.format(training_phase), losses_tr.val, step)
        if losses_af.count > 0:
            self.writer.add_scalar('loss/Loss_af_{}'.format(training_phase), losses_af.val, step)
        if losses_gscm.count > 0:
            self.writer.add_scalar('loss/Loss_gscm_{}'.format(training_phase), losses_gscm.val, step)
        if losses_sdac.count > 0:
            self.writer.add_scalar('loss/Loss_sdac_{}'.format(training_phase), losses_sdac.val, step)
            self.writer.add_scalar(
                'sdac/Loss_occupancy_{}'.format(training_phase),
                losses_sdac_occupancy.val, step)
            self.writer.add_scalar(
                'sdac/Loss_visual_state_{}'.format(training_phase),
                losses_sdac_visual.val, step)
            self.writer.add_scalar(
                'sdac/Loss_decision_state_{}'.format(training_phase),
                losses_sdac_decision.val, step)
            self.writer.add_scalar(
                'sdac/Valid_states_{}'.format(training_phase),
                losses_sdac_states.val, step)
        if losses_acssd_stable.count > 0:
            self.writer.add_scalar(
                'acssd/Loss_stable_{}'.format(training_phase),
                losses_acssd_stable.val, step)
            self.writer.add_scalar(
                'acssd/Loss_plastic_{}'.format(training_phase),
                losses_acssd_plastic.val, step)
        if losses_srsr_stable.count > 0:
            self.writer.add_scalar(
                'srsr/Loss_stable_{}'.format(training_phase),
                losses_srsr_stable.val, step)
            self.writer.add_scalar(
                'srsr/Loss_plastic_{}'.format(training_phase),
                losses_srsr_plastic.val, step)
        self.writer.add_scalar('time/Time_{}'.format(training_phase), batch_time.val, step)

    def get_normal_affinity(self, x, Norm=0.1):
        """计算批内样本的归一化关系分布，Norm 为 softmax 温度。"""
        pre_matrix_origin = cosine_similarity(x, x)
        return F.softmax(pre_matrix_origin / Norm, dim=1)

    def _parse_data(self, inputs):
        """解析数据批次，并将图像和身份标签搬到 GPU。"""
        imgs, fnames, pids, cids, domains = inputs
        return imgs.cuda(), pids.cuda(), cids, domains, fnames

    def _model_ref(self):
        """返回去除并行包装后的实际模型。"""
        return getattr(self.model, 'module', self.model)
