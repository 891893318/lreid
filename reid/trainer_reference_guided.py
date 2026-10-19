"""最终无提示词持续行人重识别方法的单阶段训练器。

本文件包含 CE + Triplet、原始关系知识迁移、GSCM、SA-RSTKT 和 SDAC 损失；
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
from reid.utils.make_loss import make_loss
from .utils.meters import AverageMeter


class Trainer(object):
    """封装前向传播、损失计算、反向传播以及 EMA 教师模型更新。"""

    def __init__(self, cfg, args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        # 同时兼容普通模型与 DataParallel/DistributedDataParallel 包装的模型。
        feat_dim = getattr(getattr(model, 'module', model), 'out_channel', 2048)
        self.loss_fn, _ = make_loss(cfg, num_classes=num_classes, feat_dim=feat_dim)
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        # SDAC 教师模型在首次需要时创建，每个持续学习阶段重新初始化一次。
        self.sdac_teacher = None
        self.sdac_teacher_phase = None
        self._logged_comparison_permutations = set()
        # Runtime diagnostics for reference-conditioned SDAC.
        self._sdac_ref_stats = None

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
        losses_sdac_text = AverageMeter()
        losses_sdac_feat = AverageMeter()
        losses_sdac_logit = AverageMeter()
        losses_sdac_ref = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, _ = self._parse_data(train_inputs)
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

            if old_model is not None:
                # 旧模型仅作为教师提供历史特征，不参与梯度计算。
                with torch.no_grad():
                    s_features_old, _, _, _ = old_model(s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old = s_features_old[0]

                # 将批内两两余弦相似度转为关系概率分布。
                affinity_new = self.get_normal_affinity(s_features)
                affinity_old = self.get_normal_affinity(s_features_old)
                sa_relation, sa_confidence = self._build_sa_rstkt_relation(
                    s_features.detach(),
                    s_features_old.detach(),
                    old_model,
                    training_phase,
                )
                # 保持新旧模型的可靠样本关系；启用 SA-RSTKT 时再做语义校准。
                divergence = self.cal_KL(
                    affinity_new,
                    affinity_old,
                    targets,
                    sa_rstkt_relation=sa_relation,
                    sa_rstkt_confidence=sa_confidence,
                )
                loss = loss + divergence * self.AF_weight
                losses_af.update(divergence.item())

            # SDAC 同时对齐语义锚点分布、视觉特征和分类输出。
            sdac_loss = self._sdac_loss(s_inputs, s_features, cls_outputs, training_phase)
            if sdac_loss is not None:
                sdac_total, sdac_text, sdac_feat, sdac_logit, sdac_ref = sdac_loss
                loss = loss + sdac_total
                losses_sdac.update(sdac_total.item())
                losses_sdac_text.update(sdac_text.item())
                losses_sdac_feat.update(sdac_feat.item())
                losses_sdac_logit.update(sdac_logit.item())
                if sdac_ref is not None:
                    losses_sdac_ref.update(float(sdac_ref))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 参数更新后，再用学生模型的最新参数更新 EMA 教师。
            self._update_sdac_teacher(training_phase)

            batch_time.update(time.time() - end)
            end = time.time()
            self._write_scalars(
                epoch, train_iters, i, training_phase, batch_time,
                losses_ce, losses_tr, losses_af, losses_gscm,
                losses_sdac, losses_sdac_text, losses_sdac_feat, losses_sdac_logit, losses_sdac_ref,
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
                if losses_gscm.count > 0:
                    msg += 'Loss_gscm {:.4f} ({:.4f})\t'.format(losses_gscm.val, losses_gscm.avg)
                    msg += 'GSCM_w {:.3f}\t'.format(self._gscm_effective_weight())
                if losses_sdac.count > 0:
                    msg += 'Loss_sdac {:.4f} ({:.4f})\t'.format(losses_sdac.val, losses_sdac.avg)
                    msg += 'SDAC_t/f/l {:.4f}/{:.4f}/{:.4f}\t'.format(
                        losses_sdac_text.val, losses_sdac_feat.val, losses_sdac_logit.val)
                    if losses_sdac_ref.count > 0:
                        stats = getattr(self, '_sdac_ref_stats', None)
                        if stats is None:
                            msg += 'SDAC_ref_w {:.3f}\t'.format(losses_sdac_ref.val)
                        else:
                            msg += (
                                'SDAC_ref gamma={:.3f} risk={:.5f}/{:.5f} '
                                'conf={:.3f} js={:.5f} w={:.3f}/{:.3f}/{:.3f}\t'
                            ).format(
                                stats['gamma'], stats['risk_mean'], stats['risk_std'],
                                stats['confidence_mean'], stats['disagreement_mean'],
                                stats['weight_min'], stats['weight_mean'], stats['weight_max'])
                print(msg)

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
    def _sdac_enabled(self, training_phase):
        """判断当前阶段是否启用 SDAC。"""
        return (
            bool(getattr(self.args, 'sdac', False))
            and getattr(self.args, 'MODEL', '') == 'clip_vit'
            and float(getattr(self.args, 'sdac_weight', 0.0)) > 0.0
            and int(training_phase) >= int(getattr(self.args, 'sdac_start_phase', 2))
        )

    def _init_sdac_teacher(self, training_phase):
        """为当前阶段创建冻结的 SDAC EMA 教师模型。"""
        if self.sdac_teacher is not None and self.sdac_teacher_phase == training_phase:
            return
        self.sdac_teacher = copy.deepcopy(self._model_ref())
        if (getattr(self.args, 'reference_comparison_policy', 'stage_shared')
                == 'previous_stage_mismatch'):
            previous_stage = int(training_phase) - 2
            refresh = getattr(self.sdac_teacher, 'refresh_stage_reference', None)
            if previous_stage < 0 or not callable(refresh) or not refresh(previous_stage):
                raise RuntimeError(
                    'previous_stage_mismatch requires a stage-varying SDAC reference')
            metadata = self.sdac_teacher.get_semantic_reference_metadata()
            print('Comparison reference set: role=sdac_teacher stage={} sha256={}'.format(
                previous_stage, metadata.get('sha256')))
        self.sdac_teacher.cuda()
        self.sdac_teacher.eval()
        for p in self.sdac_teacher.parameters():
            p.requires_grad_(False)
        self.sdac_teacher_phase = training_phase

    @torch.no_grad()
    def _update_sdac_teacher(self, training_phase):
        """使用指数移动平均更新教师的浮点参数及缓冲区。"""
        if not self._sdac_enabled(training_phase) or self.sdac_teacher is None:
            return
        momentum = float(getattr(self.args, 'sdac_ema', 0.997))
        student_params = dict(self._model_ref().named_parameters())
        student_buffers = dict(self._model_ref().named_buffers())
        # 通过名称匹配参数，形状不一致或非浮点项直接跳过。
        for name, param in self.sdac_teacher.named_parameters():
            cur = student_params.get(name)
            if cur is None or cur.shape != param.shape or not torch.is_floating_point(param):
                continue
            current = cur.detach().to(device=param.device, dtype=param.dtype)
            # The frozen CLIP text tower is part of the fixed reference and
            # must remain an exact copy rather than an EMA-derived quantity.
            if not cur.requires_grad:
                param.copy_(current)
            else:
                param.mul_(momentum).add_(current, alpha=1.0 - momentum)
        for name, buf in self.sdac_teacher.named_buffers():
            cur = student_buffers.get(name)
            if cur is None or cur.shape != buf.shape:
                continue
            current = cur.detach().to(device=buf.device, dtype=buf.dtype)
            is_reference = 'text_anchor_' in name or 'semantic_reference_' in name
            if (is_reference
                    and getattr(self.args, 'reference_comparison_policy', 'stage_shared')
                    == 'previous_stage_mismatch'):
                # This negative control intentionally keeps the teacher in A_(t-1).
                continue
            if (is_reference or not torch.is_floating_point(buf)):
                buf.copy_(current)
            else:
                buf.mul_(momentum).add_(current, alpha=1.0 - momentum)
        self.sdac_teacher.eval()

    def _reverse_kl_per_sample(self, student_logits, teacher_logits, temperature=1.0):
        """返回逐样本 KL(student || teacher)，便于按参考可靠度加权。"""
        if (student_logits is None or teacher_logits is None
                or student_logits.shape != teacher_logits.shape):
            return None
        temperature = max(float(temperature), 1e-6)
        log_p_s = F.log_softmax(student_logits.float() / temperature, dim=1)
        log_p_t = F.log_softmax(teacher_logits.detach().float() / temperature, dim=1)
        p_s = log_p_s.exp()
        return (p_s * (log_p_s - log_p_t)).sum(dim=1) * (temperature * temperature)

    def _reverse_kl(self, student_logits, teacher_logits, temperature=1.0):
        """计算 KL(student || teacher)，并应用蒸馏温度的平方补偿。"""
        values = self._reverse_kl_per_sample(student_logits, teacher_logits, temperature)
        if values is None:
            ref = student_logits if isinstance(student_logits, torch.Tensor) else teacher_logits
            device = ref.device if isinstance(ref, torch.Tensor) else torch.device('cuda')
            return torch.zeros((), device=device)
        return values.mean()

    @staticmethod
    def _weighted_mean(values, weights=None, eps=1e-8):
        """对逐样本损失做可靠度加权；无权重时退化为普通均值。"""
        if values is None:
            return None
        if weights is None:
            return values.mean()
        weights = weights.detach().to(device=values.device, dtype=values.dtype)
        if weights.dim() != 1 or weights.numel() != values.numel():
            raise RuntimeError('SDAC reference reliability shape mismatch')
        return (values * weights).sum() / weights.sum().clamp_min(eps)

    def _sdac_ref_gamma(self):
        """Return the current strength of reference-conditioned consistency.

        The schedule starts from the vanilla SDAC solution (gamma=0) and
        linearly introduces reference conditioning only after the EMA teacher
        has become reasonably stable.
        """
        gamma_max = max(float(getattr(self.args, 'sdac_ref_gamma', 0.20)), 0.0)
        gamma_max = min(gamma_max, 0.95)
        epoch = int(getattr(self, '_current_epoch', 0))
        start_epoch = max(int(getattr(self.args, 'sdac_ref_start_epoch', 5)), 0)
        warmup_epochs = max(int(getattr(self.args, 'sdac_ref_warmup_epochs', 10)), 0)
        if epoch < start_epoch or gamma_max <= 0.0:
            return 0.0
        if warmup_epochs == 0:
            return gamma_max
        progress = min(float(epoch - start_epoch + 1) / float(warmup_epochs), 1.0)
        return gamma_max * progress

    def _sdac_reference_conditioning(self, student_anchor_logits, teacher_anchor_logits):
        """Build bounded per-sample consistency weights from external-reference risk.

        Reference risk is high when the EMA teacher is confident in the fixed
        reference coordinate system but the student and teacher disagree:

            d_i = JS(p_i^s || p_i^t) / log(2)
            c_i = 1 - H(p_i^t) / log(K)
            r_i = c_i * d_i

        The risk is standardized *within the current mini-batch* and converted
        to a bounded weight around 1:

            w_i = 1 + gamma * tanh((r_i - mean(r)) / (std(r) + eps))

        Hence gamma=0 exactly recovers vanilla uniform feature/logit
        consistency. The whole branch is detached so the model cannot optimize
        the reliability estimator itself.
        """
        if (student_anchor_logits is None or teacher_anchor_logits is None
                or student_anchor_logits.shape != teacher_anchor_logits.shape
                or student_anchor_logits.dim() != 2):
            self._sdac_ref_stats = None
            return None

        num_anchors = int(student_anchor_logits.size(1))
        if num_anchors <= 1:
            self._sdac_ref_stats = None
            return None

        with torch.no_grad():
            p_s = F.softmax(student_anchor_logits.detach().float(), dim=1)
            p_t = F.softmax(teacher_anchor_logits.detach().float(), dim=1)

            # Teacher confidence in the fixed external coordinate system.
            entropy_t = -(p_t * p_t.clamp_min(1e-8).log()).sum(dim=1)
            confidence_t = 1.0 - entropy_t / max(math.log(float(num_anchors)), 1e-8)
            confidence_t = confidence_t.clamp(0.0, 1.0)

            # Symmetric Jensen-Shannon disagreement, normalized to [0, 1].
            mixture = 0.5 * (p_s + p_t)
            log_mix = mixture.clamp_min(1e-8).log()
            kl_s = (p_s * (p_s.clamp_min(1e-8).log() - log_mix)).sum(dim=1)
            kl_t = (p_t * (p_t.clamp_min(1e-8).log() - log_mix)).sum(dim=1)
            disagreement = 0.5 * (kl_s + kl_t)
            disagreement = (disagreement / math.log(2.0)).clamp(0.0, 1.0)

            risk = (confidence_t * disagreement).clamp_min(0.0)
            risk_mean = risk.mean()
            risk_std = risk.std(unbiased=False)

            gamma = self._sdac_ref_gamma()
            std_eps = max(float(getattr(self.args, 'sdac_ref_std_eps', 1e-4)), 0.0)

            # During warmup (gamma=0) or an almost-uniform-risk batch, use None
            # so _weighted_mean executes the exact original mean path.
            if gamma <= 0.0 or float(risk_std.item()) <= std_eps:
                weights = None
                weight_min = weight_mean = weight_max = 1.0
            else:
                normalized_risk = (risk - risk_mean) / (risk_std + 1e-6)
                weights = 1.0 + gamma * torch.tanh(normalized_risk)
                weights = weights.detach()
                weight_min = float(weights.min().item())
                weight_mean = float(weights.mean().item())
                weight_max = float(weights.max().item())

            self._sdac_ref_stats = {
                'gamma': float(gamma),
                'risk_mean': float(risk_mean.item()),
                'risk_std': float(risk_std.item()),
                'confidence_mean': float(confidence_t.mean().item()),
                'disagreement_mean': float(disagreement.mean().item()),
                'weight_min': weight_min,
                'weight_mean': weight_mean,
                'weight_max': weight_max,
            }
            return weights

    def _permute_old_anchor_columns(self, values, training_phase, role):
        """Break old/new slot correspondence while preserving the anchor set."""
        if (values is None
                or getattr(self.args, 'reference_comparison_policy', 'stage_shared')
                != 'permuted_old_slots'):
            return values
        if getattr(self.args, 'permuted_old_slots_target', 'both') not in ('both', 'sdac'):
            return values
        model_ref = self._model_ref()
        getter = getattr(model_ref, 'get_reference_slot_permutation', None)
        if not callable(getter):
            raise RuntimeError('permuted_old_slots requires a reference permutation provider')
        stage_index = int(training_phase) - 1
        permutation = getter(stage_index, device=values.device)
        if values.dim() != 2 or values.size(1) != permutation.numel():
            raise RuntimeError('anchor permutation shape mismatch')
        logged = getattr(self, '_logged_comparison_permutations', None)
        if logged is None:
            logged = set()
            self._logged_comparison_permutations = logged
        key = (role, stage_index)
        if key not in logged:
            print('Comparison anchor slots permuted: role={} stage={} permutation={}'.format(
                role, stage_index, permutation.detach().cpu().tolist()))
            logged.add(key)
        return values.index_select(1, permutation)

    def _sdac_loss(self, inputs, features, logits, training_phase):
        """计算 SDAC 的语义、特征及分类输出三种一致性损失。"""
        if not self._sdac_enabled(training_phase):
            return None
        self._init_sdac_teacher(training_phase)
        with torch.no_grad():
            teacher_features, _, teacher_logits, _ = self.sdac_teacher(inputs, get_all_feat=True)
            if isinstance(teacher_features, tuple):
                teacher_features = teacher_features[0]

        zero = features.new_zeros(())
        # 语义分支：约束学生和教师在文本语义锚点上的分布一致。
        text_loss = zero
        ref_reliability = None
        student_anchor_logits = None
        teacher_anchor_logits = None
        model_ref = self._model_ref()
        if (getattr(model_ref, 'semantic_reference', 'semantic') != 'none'
                and hasattr(model_ref, 'semantic_anchor_logits')):
            student_anchor_logits = model_ref.semantic_anchor_logits(
                features,
                temperature=getattr(self.args, 'sdac_text_temp', 0.07),
            )
            if student_anchor_logits is not None:
                with torch.no_grad():
                    teacher_anchor_logits = self.sdac_teacher.semantic_anchor_logits(
                        teacher_features,
                        temperature=getattr(self.args, 'sdac_text_temp', 0.07),
                    )
                    teacher_anchor_logits = self._permute_old_anchor_columns(
                        teacher_anchor_logits, training_phase, role='sdac_teacher')
                text_loss = self._reverse_kl(
                    student_anchor_logits, teacher_anchor_logits, temperature=1.0)

                if bool(getattr(self.args, 'sdac_ref_guidance', False)):
                    ref_reliability = self._sdac_reference_conditioning(
                        student_anchor_logits, teacher_anchor_logits)

        if bool(getattr(self.args, 'sdac_ref_guidance', False)) and ref_reliability is None:
            raise RuntimeError(
                'SDAC reference guidance is enabled but aligned semantic-anchor responses are unavailable')

        # 特征分支：逐样本余弦距离，由 reference-conditioned risk 权重调节。
        feat_per_sample = 1.0 - F.cosine_similarity(
            F.normalize(features.float(), p=2, dim=1),
            F.normalize(teacher_features.detach().float(), p=2, dim=1),
            dim=1,
        )
        feat_loss = self._weighted_mean(feat_per_sample, ref_reliability)

        # 分类分支：逐样本反向 KL；启用参考引导时使用同一风险权重。
        logit_per_sample = self._reverse_kl_per_sample(
            logits,
            teacher_logits,
            temperature=getattr(self.args, 'sdac_temp', 4.0),
        )
        if logit_per_sample is None:
            logit_loss = zero
        else:
            logit_loss = self._weighted_mean(logit_per_sample, ref_reliability)

        total = float(getattr(self.args, 'sdac_weight', 0.2)) * (
            float(getattr(self.args, 'sdac_text_weight', 1.0)) * text_loss
            + float(getattr(self.args, 'sdac_feat_weight', 0.25)) * feat_loss
            + float(getattr(self.args, 'sdac_logit_weight', 0.5)) * logit_loss
        )
        # Keep the existing 5-value return contract. When conditioning is
        # enabled, log the mean weight (normally close to 1); richer diagnostics
        # are stored in self._sdac_ref_stats.
        if bool(getattr(self.args, 'sdac_ref_guidance', False)):
            stats = getattr(self, '_sdac_ref_stats', None)
            ref_mean = None if stats is None else float(stats['weight_mean'])
        else:
            self._sdac_ref_stats = None
            ref_mean = None
        return total, text_loss, feat_loss, logit_loss, ref_mean

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
                       losses_sdac, losses_sdac_text, losses_sdac_feat, losses_sdac_logit,
                       losses_sdac_ref):
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
            self.writer.add_scalar('loss/Loss_sdac_text_{}'.format(training_phase), losses_sdac_text.val, step)
            self.writer.add_scalar('loss/Loss_sdac_feat_{}'.format(training_phase), losses_sdac_feat.val, step)
            self.writer.add_scalar('loss/Loss_sdac_logit_{}'.format(training_phase), losses_sdac_logit.val, step)
            if losses_sdac_ref.count > 0:
                self.writer.add_scalar('sdac/Reference_weight_mean_{}'.format(training_phase), losses_sdac_ref.val, step)
                stats = getattr(self, '_sdac_ref_stats', None)
                if stats is not None:
                    self.writer.add_scalar('sdac/Reference_gamma_{}'.format(training_phase), stats['gamma'], step)
                    self.writer.add_scalar('sdac/Reference_risk_mean_{}'.format(training_phase), stats['risk_mean'], step)
                    self.writer.add_scalar('sdac/Reference_risk_std_{}'.format(training_phase), stats['risk_std'], step)
                    self.writer.add_scalar('sdac/Reference_confidence_{}'.format(training_phase), stats['confidence_mean'], step)
                    self.writer.add_scalar('sdac/Reference_JS_{}'.format(training_phase), stats['disagreement_mean'], step)
                    self.writer.add_scalar('sdac/Reference_weight_min_{}'.format(training_phase), stats['weight_min'], step)
                    self.writer.add_scalar('sdac/Reference_weight_max_{}'.format(training_phase), stats['weight_max'], step)
        self.writer.add_scalar('time/Time_{}'.format(training_phase), batch_time.val, step)

    def get_normal_affinity(self, x, Norm=0.1):
        """计算批内样本的归一化关系分布，Norm 为 softmax 温度。"""
        pre_matrix_origin = cosine_similarity(x, x)
        return F.softmax(pre_matrix_origin / Norm, dim=1)

    def _parse_data(self, inputs):
        """解析数据批次，并将图像和身份标签搬到 GPU。"""
        imgs, _, pids, cids, domains = inputs
        return imgs.cuda(), pids.cuda(), cids, domains

    def _model_ref(self):
        """返回去除并行包装后的实际模型。"""
        return getattr(self.model, 'module', self.model)
