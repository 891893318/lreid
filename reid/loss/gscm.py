"""GSCM（分组语义跨摄像头间隔）损失。

该模块把固定 CLIP 文本锚点按行人属性分组，并用组内概率相对均匀分布的
偏移描述样本语义。语义越相似的异 ID 样本会获得越大的负样本间隔，从而
迫使视觉编码器学习更有判别力的跨摄像头表示。
"""

import math

import torch
import torch.nn.functional as F


# 第 0 个提示词是通用“行人”描述，不参与属性分组；其余 24 个提示词与
# reid/models/clip_vit.py 中的 SRAR_TEXT_PROMPTS 一一对应，并按互斥属性分组。
DEFAULT_ANCHOR_GROUPS = (
    (1, 2, 3, 4, 5, 6),       # 上衣颜色
    (7, 8, 9, 10, 11),        # 下装类型
    (12, 13, 14),              # 携带物品
    (15, 16, 17),              # 头部外观
    (18, 19, 20),              # 行人视角
    (21, 22),                  # 行为状态
    (23, 24),                  # 图像质量与遮挡
)

# 消融实验使用的固定乱序分组：保持各组大小不变，但破坏真实属性结构。
# 固定排列可保证不同运行之间的对照结果可复现。
SHUFFLED_ANCHOR_GROUPS = (
    (15, 6, 4, 18, 3, 1),
    (24, 21, 22, 20, 9),
    (17, 16, 23),
    (19, 8, 12),
    (10, 13, 7),
    (11, 5),
    (14, 2),
)

# “全局组”消融：把全部属性锚点视为一个大组。
GLOBAL_ANCHOR_GROUPS = (tuple(range(1, 25)),)


def grouped_anchor_descriptor(anchor_logits, groups=DEFAULT_ANCHOR_GROUPS,
                              use_entropy_confidence=True, eps=1e-8):
    """由固定文本锚点构造带置信度的分组语义描述符。

    输入 ``anchor_logits`` 的形状为 [B, 25]。每组先做 softmax，再减去
    均匀分布，以消除组大小和公共响应造成的偏置；归一化后的残差方向表示
    属性偏好。可选的熵置信度会抑制组内分布模糊的样本。
    """
    if anchor_logits.dim() != 2:
        raise ValueError('anchor_logits must be a two-dimensional tensor')

    descriptors = []
    width = anchor_logits.size(1)
    for group in groups:
        if not group or max(group) >= width:
            raise ValueError('anchor group index exceeds anchor-logit width')

        indices = torch.as_tensor(
            group, device=anchor_logits.device, dtype=torch.long)
        # 只在同一互斥属性组内部归一化，得到可比较的属性概率。
        probs = F.softmax(anchor_logits.index_select(1, indices).float(), dim=1)
        uniform = 1.0 / len(group)
        # 均匀分布代表“无明确属性倾向”，因此使用概率残差描述语义方向。
        residual = probs - uniform
        direction = F.normalize(residual, p=2, dim=1, eps=eps)

        if use_entropy_confidence:
            entropy = -(
                probs.clamp_min(eps) * probs.clamp_min(eps).log()
            ).sum(dim=1)
            # 归一化熵越低，说明属性判断越明确，置信度越高。
            confidence = 1.0 - entropy / math.log(len(group))
        else:
            confidence = probs.new_ones(probs.size(0))
        descriptors.append(direction * confidence.clamp(0.0, 1.0).unsqueeze(1))

    return torch.cat(descriptors, dim=1)


def grouped_semantic_cross_camera_margin_loss(
        features, targets, cameras, anchor_logits, visual_temperature=0.07,
        base_margin=0.05, semantic_margin=0.10,
        groups=DEFAULT_ANCHOR_GROUPS, use_entropy_confidence=True,
        use_cross_camera=True):
    """拉近跨摄像头同 ID，并推远语义相似的异 ID 难负样本。

    正样本优先选择同 ID、不同摄像头的图像；若某个锚点样本在 batch 中
    没有跨摄像头正样本，则退化为全部同 ID 正样本。异 ID 的语义相似度越
    高，其附加间隔 ``base_margin + semantic_margin * similarity`` 越大。
    """
    if features.dim() != 2:
        raise ValueError('features must be a two-dimensional tensor')
    if features.size(0) != targets.numel():
        raise ValueError('features and targets must have the same batch size')
    if anchor_logits.size(0) != features.size(0):
        raise ValueError('anchor logits and features must have the same batch size')

    device = features.device
    targets = targets.detach().long().to(device)
    cameras = None if cameras is None else cameras.detach().long().to(device)
    batch_size = features.size(0)
    if batch_size < 2:
        return features.sum() * 0.0

    visual_temperature = max(float(visual_temperature), 1e-6)
    base_margin = max(float(base_margin), 0.0)
    semantic_margin = max(float(semantic_margin), 0.0)

    # 归一化特征的点积即余弦相似度。
    visual = F.normalize(features.float(), p=2, dim=1)
    visual_similarity = visual @ visual.t()

    # 文本锚点只负责生成语义权重，不应通过该分支反向更新模型。
    with torch.no_grad():
        descriptor = grouped_anchor_descriptor(
            anchor_logits.detach(),
            groups=groups,
            use_entropy_confidence=use_entropy_confidence,
        )
        semantic_similarity = descriptor @ descriptor.t()
        semantic_similarity = (semantic_similarity / len(groups)).clamp(0.0, 1.0)

    eye = torch.eye(batch_size, device=device, dtype=torch.bool)
    same_identity = targets[:, None].eq(targets[None, :]) & ~eye
    different_identity = ~targets[:, None].eq(targets[None, :]) & ~eye

    # 默认使用全部同 ID 样本；存在跨摄像头正样本时优先采用跨摄像头配对。
    positive_mask = same_identity
    if cameras is not None and use_cross_camera:
        cross_camera = cameras[:, None].ne(cameras[None, :])
        cross_camera_positive = same_identity & cross_camera
        has_cross_camera = cross_camera_positive.any(dim=1, keepdim=True)
        positive_mask = torch.where(
            has_cross_camera, cross_camera_positive, same_identity)

    valid = positive_mask.any(dim=1) & different_identity.any(dim=1)
    if not bool(valid.any()):
        return features.sum() * 0.0

    positive_logits = visual_similarity / visual_temperature
    # “长得越像但身份不同”的负样本获得越大的判别间隔。
    negative_margin = base_margin + semantic_margin * semantic_similarity
    negative_logits = (visual_similarity + negative_margin) / visual_temperature

    neg_inf = torch.finfo(positive_logits.dtype).min
    positive_lse = torch.logsumexp(
        positive_logits.masked_fill(~positive_mask, neg_inf), dim=1)
    negative_lse = torch.logsumexp(
        negative_logits.masked_fill(~different_identity, neg_inf), dim=1)
    # log-sum-exp 聚合一对多正负样本，softplus 给出平滑排序目标。
    return F.softplus(negative_lse[valid] - positive_lse[valid]).mean()
