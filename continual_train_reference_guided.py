from __future__ import print_function, absolute_import
import argparse
import json
import os.path as osp
import sys
import os
import shutil
import math
from datetime import datetime, timedelta

from torch.backends import cudnn
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model as make_resnet_model
from reid.models.clip_vit import make_model as make_clip_vit_model
from reid.trainer_reference_guided import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res


def _resolve_reproduce_root(logs_dir):
    logs_dir = osp.normpath(logs_dir)
    parts = logs_dir.split(os.sep)
    if parts and parts[0] == 'reproduce':
        return parts[0]
    return 'reproduce'


def _backup_code(src_dir, backup_dir, reproduce_root):
    reproduce_root = osp.abspath(reproduce_root)

    def ignore(dirpath, names):
        ignored = {
            '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
            # Derived post-hoc outputs can contain multi-GB feature caches and
            # are reproducible from checkpoints; never duplicate them per run.
            'analysis_psc',
        }
        if osp.abspath(dirpath) == osp.abspath(src_dir):
            ignored.add(osp.basename(reproduce_root))
            try:
                rel_reproduce = osp.relpath(reproduce_root, osp.abspath(src_dir))
                if not rel_reproduce.startswith('..'):
                    ignored.add(rel_reproduce.split(os.sep)[0])
            except ValueError:
                pass
        ignored.update(name for name in names if name.endswith(('.pyc', '.pyo')))
        return ignored

    shutil.copytree(src_dir, backup_dir, ignore=ignore)


def prepare_reproduce_outputs(args):
    reproduce_root_arg = getattr(args, 'reproduce_root', None)
    reproduce_root = osp.abspath(reproduce_root_arg) if reproduce_root_arg else osp.abspath(_resolve_reproduce_root(args.logs_dir))
    logs_setting_dir = 'setting{}'.format(int(args.setting))
    run_time = datetime.now().replace(microsecond=0)

    while True:
        timestamp = run_time.strftime('%Y年%m月%d日-%H时%M分%S秒')
        run_id = 'code_{}'.format(timestamp)
        logs_dir = osp.join(reproduce_root, 'logs', logs_setting_dir, run_id)
        pth_dir = osp.join(reproduce_root, 'pth', run_id)
        code_backup_dir = osp.join(reproduce_root, 'code', run_id)
        if not any(osp.exists(path) for path in (logs_dir, pth_dir, code_backup_dir)):
            break
        run_time += timedelta(seconds=1)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(reproduce_root, exist_ok=True)
    _backup_code(osp.abspath(osp.dirname(__file__)), code_backup_dir, reproduce_root)

    args.original_logs_dir = args.logs_dir
    args.reproduce_root = reproduce_root
    args.logs_setting_dir = logs_setting_dir
    args.run_id = run_id
    args.logs_dir = logs_dir
    args.pth_dir = pth_dir
    args.code_backup_dir = code_backup_dir
    return args


def make_lreid_model(args, num_class):
    if args.MODEL == '50x':
        return make_resnet_model(args, num_class=num_class, camera_num=0, view_num=0)
    if args.MODEL == 'clip_vit':
        return make_clip_vit_model(args, num_class=num_class, camera_num=0, view_num=0)
    raise AssertionError("the model {} is not supported!".format(args.MODEL))


def get_model_out_channel(args):
    if args.MODEL == '50x':
        return 2048
    if args.MODEL == 'clip_vit':
        return 512
    raise AssertionError("the model {} is not supported!".format(args.MODEL))


def _semantic_reference_metadata(model):
    """Return JSON/checkpoint-safe provenance for the fixed reference."""
    model_ref = getattr(model, 'module', model)
    getter = getattr(model_ref, 'get_semantic_reference_metadata', None)
    return getter() if callable(getter) else None


def _materialize_and_record_semantic_reference(args, model):
    """Build the fixed reference once and record the exact experimental input."""
    model_ref = getattr(model, 'module', model)
    getter = getattr(model_ref, 'get_text_anchors', None)
    if callable(getter) and getattr(args, 'semantic_reference', 'semantic') != 'none':
        getter()
    metadata = _semantic_reference_metadata(model)
    if metadata is None:
        return
    fpath = osp.join(args.logs_dir, 'semantic_reference.json')
    with open(fpath, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print('Semantic reference: {}'.format(json.dumps(metadata, sort_keys=True)))
    print('Semantic reference metadata: {}'.format(fpath))


def _refresh_stage_reference(model, stage_index):
    """Install one shared bank before copying student/old/EMA stage branches."""
    model_ref = getattr(model, 'module', model)
    refresh = getattr(model_ref, 'refresh_stage_reference', None)
    if not callable(refresh) or not refresh(stage_index):
        return
    metadata = _semantic_reference_metadata(model)
    print('Stage reference refreshed: stage={} sha256={}'.format(
        stage_index, metadata.get('sha256') if metadata else None))


def _set_comparison_reference(args, model, stage_index, role):
    """Optionally put a comparison-only branch in the previous stage bank."""
    if getattr(args, 'reference_comparison_policy', 'stage_shared') != 'previous_stage_mismatch':
        return
    if stage_index <= 0:
        return
    model_ref = getattr(model, 'module', model)
    refresh = getattr(model_ref, 'refresh_stage_reference', None)
    previous_stage = int(stage_index) - 1
    if not callable(refresh) or not refresh(previous_stage):
        raise RuntimeError(
            'previous_stage_mismatch requires a stage-varying reference')
    metadata = _semantic_reference_metadata(model)
    print('Comparison reference set: role={} stage={} sha256={}'.format(
        role, previous_stage, metadata.get('sha256') if metadata else None))


def prepare_model_for_checkpoint(model, checkpoint):
    """Reject accidental evaluation/resume with a different reference source."""
    saved = checkpoint.get('semantic_reference')
    current = _semantic_reference_metadata(model)
    # Checkpoints predating reference controls have no provenance and retain the
    # legacy loading behavior.
    if saved is None or current is None:
        return
    keys = ('source', 'count', 'configured_count', 'dimension', 'wording',
            'template_policy', 'prompts', 'groups')
    random_sources = ('random_vector', 'stage_random_vector')
    if saved.get('source') in random_sources or current.get('source') in random_sources:
        keys += ('seed',)
    mismatch = [key for key in keys
                if key in saved and saved.get(key) != current.get(key)]
    if mismatch:
        details = ', '.join('{}: checkpoint={!r}, current={!r}'.format(
            key, saved.get(key), current.get(key)) for key in mismatch)
        raise ValueError('semantic reference configuration mismatch ({})'.format(details))


def build_optimizer_params(model, args):
    params = []
    head_lr = args.head_lr
    if head_lr is None:
        head_lr = 3.5e-4 if args.MODEL == 'clip_vit' else args.lr

    clip_backbone_params = 0
    clip_head_params = 0
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue

        lr = args.lr
        if args.MODEL == 'clip_vit' and (
                'classifier' in key or 'bottleneck' in key):
            lr = head_lr
            clip_head_params += value.numel()
        elif args.MODEL == 'clip_vit':
            clip_backbone_params += value.numel()

        params += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]

    if args.MODEL == 'clip_vit':
        print('CLIP optimizer lr: backbone={} head={} backbone_params={} head_params={}'.format(
            args.lr, head_lr, clip_backbone_params, clip_head_params))
    return params


def _validate_semantic_reference_args(args, argument_parser):
    """Keep the no-reference control free of hidden semantic inputs."""
    # Legacy reference-floor option is still accepted for old scripts, but the
    # new reference-conditioned SDAC no longer uses floor-clamped reliability.
    ref_floor = float(getattr(args, 'sdac_ref_floor', 0.05))
    if not math.isfinite(ref_floor) or ref_floor < 0.0 or ref_floor > 1.0:
        argument_parser.error('--sdac-ref-floor must be a finite value in [0, 1]')

    ref_gamma = float(getattr(args, 'sdac_ref_gamma', 0.20))
    if not math.isfinite(ref_gamma) or ref_gamma < 0.0 or ref_gamma >= 1.0:
        argument_parser.error('--sdac-ref-gamma must be a finite value in [0, 1)')
    ref_start = int(getattr(args, 'sdac_ref_start_epoch', 5))
    ref_warmup = int(getattr(args, 'sdac_ref_warmup_epochs', 10))
    if ref_start < 0:
        argument_parser.error('--sdac-ref-start-epoch must be >= 0')
    if ref_warmup < 0:
        argument_parser.error('--sdac-ref-warmup-epochs must be >= 0')
    ref_std_eps = float(getattr(args, 'sdac_ref_std_eps', 1e-4))
    if not math.isfinite(ref_std_eps) or ref_std_eps < 0.0:
        argument_parser.error('--sdac-ref-std-eps must be a finite value >= 0')
    source = getattr(args, 'semantic_reference', 'semantic')
    comparison_policy = getattr(args, 'reference_comparison_policy', 'stage_shared')
    if (comparison_policy == 'previous_stage_mismatch'
            and source != 'stage_random_vector'):
        argument_parser.error(
            '--reference-comparison-policy previous_stage_mismatch requires '
            '--semantic-reference stage_random_vector')
    alpha_schedule = getattr(args, 'sa_ltkc_alpha_schedule', None)
    if alpha_schedule:
        try:
            alpha_values = [float(value.strip()) for value in alpha_schedule.split(',')]
        except ValueError:
            argument_parser.error('--sa-ltkc-alpha-schedule must be comma-separated floats')
        if len(alpha_values) != 4:
            argument_parser.error('--sa-ltkc-alpha-schedule requires exactly four transition alphas')
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0
               for value in alpha_values):
            argument_parser.error('--sa-ltkc-alpha-schedule values must be finite and in [0, 1]')
    if source != 'none':
        return
    conflicts = []
    if bool(getattr(args, 'gscm', False)) and float(getattr(args, 'gscm_weight', 0.0)) > 0.0:
        conflicts.append('--gscm')
    if bool(getattr(args, 'sa_rstkt', False)) and float(getattr(args, 'sa_rstkt_weight', 0.0)) > 0.0:
        conflicts.append('--sa-rstkt')
    if (bool(getattr(args, 'sdac', False))
            and float(getattr(args, 'sdac_text_weight', 0.0)) != 0.0):
        conflicts.append('--sdac-text-weight must be 0')
    if bool(getattr(args, 'sdac_ref_guidance', False)):
        conflicts.append('--sdac-ref-guidance requires a nonempty semantic reference')
    if (bool(getattr(args, 'sa_ltkc', False))
            and float(getattr(args, 'sa_ltkc_semantic_weight', 0.0)) != 0.0):
        conflicts.append('--sa-ltkc-semantic-weight must be 0')
    if conflicts:
        argument_parser.error(
            '--semantic-reference none conflicts with {}'.format(', '.join(conflicts)))


def main():
    args = parser.parse_args()
    _validate_semantic_reference_args(args, parser)
    args = prepare_reproduce_outputs(args)

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    log_name = 'log.txt'
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    print("Reproduce root: {}".format(args.reproduce_root))
    print("Run id: {}".format(args.run_id))
    print("Logs dir: {}".format(args.logs_dir))
    print("PTH dir: {}".format(args.pth_dir))
    print("Code backup dir: {}".format(args.code_backup_dir))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    else:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_lreid_model(args, num_class=first_train_set[1])

    model.cuda()
    model = DataParallel(model)
    _materialize_and_record_semantic_reference(args, model)
    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        prepare_model_for_checkpoint(model, checkpoint)
        copy_state_dict(checkpoint['state_dict'], model)     #    
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            prepare_model_for_checkpoint(model, checkpoint)
            copy_state_dict(checkpoint['state_dict'], model)

                         
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, step + 1)
            model = linear_combination(args, model, model_old, best_alpha)

            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
                'semantic_reference': _semantic_reference_metadata(model),
            }, True, fpath=osp.join(args.pth_dir, save_name))
        # extract_test_features(model, all_train_sets, all_test_only_sets, args.test_folder)
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)

        exit(0)
    

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        prepare_model_for_checkpoint(model, checkpoint)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    out_channel = get_model_out_channel(args)


    # train on the datasets squentially
    for set_index in range(0, len(training_set)):       
        # Refresh before taking any old-model/EMA copies so every branch within
        # this stage uses the same bank. Fixed references are a no-op here.
        _refresh_stage_reference(model, set_index)
        model_old = copy.deepcopy(model)
        _set_comparison_reference(args, model_old, set_index, role='sdlc_old')
        model = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res)
        if set_index>0:
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, set_index)
            model = linear_combination(args, model, model_old, best_alpha)
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)    
        # This is the actual boundary model used to initialize the next stage.
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': args.epochs0 if set_index == 0 else args.epochs,
            'mAP': 0.,
            'stage_index': set_index,
            'stage_name': training_set[set_index],
            'post_sdlc': bool(set_index > 0),
            'semantic_reference': _semantic_reference_metadata(model),
        }, True, fpath=osp.join(
            args.pth_dir, '{}_consolidated_checkpoint.pth.tar'.format(training_set[set_index])))
    print('finished')
def get_normal_affinity(x,Norm=100):
    """把特征余弦相似度转换为行归一化的样本关系概率矩阵。"""
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix



# SDLC控制开关
def _sa_ltkc_enabled(args):
    """SA-LTKC 当前仅对 CLIP ViT 主干生效。"""
    return bool(getattr(args, 'sa_ltkc', False)) and getattr(args, 'MODEL', '') == 'clip_vit'

# SDLC第一步Compute Relation Drift
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    """根据关系漂移和语义锚点漂移估计 SA-LTKC 的阶段融合系数。"""
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]  # trainloader of current dataset
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new)
    Affin_old = get_normal_affinity(features_all_old)

    # 新旧模型亲和矩阵的平均 L1 差异刻画关系结构漂移。
    Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()

    if not _sa_ltkc_enabled(args):
        alpha=float(1-Difference)
        return alpha

    relation_diff = Difference.float().clamp(0.0, 1.0)
    semantic_diff = _semantic_anchor_drift(
        args, model, model_old, features_all_new, features_all_old,
        stage_index=set_index)
    relation_weight = max(float(getattr(args, 'sa_ltkc_relation_weight', 1.0)), 0.0)
    semantic_weight = max(float(getattr(args, 'sa_ltkc_semantic_weight', 0.5)), 0.0)
    weight_sum = max(relation_weight + semantic_weight, 1e-6)
    # 两类漂移加权融合；漂移越大，alpha 越小，参数越偏向旧模型。
    drift = (relation_weight * relation_diff + semantic_weight * semantic_diff) / weight_sum
    min_alpha = float(getattr(args, 'sa_ltkc_min_alpha', 0.55))
    max_alpha = float(getattr(args, 'sa_ltkc_max_alpha', 0.98))
    computed_alpha = float((1.0 - drift).clamp(min=min_alpha, max=max_alpha))
    alpha_schedule = getattr(args, 'sa_ltkc_alpha_schedule', None)
    if alpha_schedule:
        alpha_values = [float(value.strip()) for value in alpha_schedule.split(',')]
        schedule_index = int(set_index) - 1
        if schedule_index < 0 or schedule_index >= len(alpha_values):
            raise RuntimeError('SA-LTKC alpha schedule does not cover stage {}'.format(set_index))
        alpha = float(alpha_values[schedule_index])
        print(('SA-LTKC alpha on {}: relation_diff={:.4f} semantic_diff={:.4f} '
               'computed={:.4f} alpha_source=fixed_schedule alpha={:.4f}').format(
            name_new, float(relation_diff), float(semantic_diff), computed_alpha, alpha))
    else:
        alpha = computed_alpha
        print('SA-LTKC alpha on {}: relation_diff={:.4f} semantic_diff={:.4f} alpha={:.4f}'.format(
            name_new, float(relation_diff), float(semantic_diff), alpha))
    return alpha

@torch.no_grad()
# SDLC 第二步Compute Semantic Anchor Drift
def _semantic_anchor_probs_from_features(args, model, features):
    """分块计算特征在固定文本锚点上的概率，避免一次性占用过多显存。"""
    model_ref = getattr(model, 'module', model)
    if not hasattr(model_ref, 'semantic_anchor_logits'):
        return None
    try:
        device = next(model_ref.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = features.detach().float()
    outputs = []
    chunk_size = max(1, int(getattr(args, 'sa_ltkc_semantic_chunk', 512)))
    temp = getattr(args, 'sa_ltkc_text_temp', getattr(args, 'sdac_text_temp', 0.07))
    for chunk in features.split(chunk_size, dim=0):
        logits = model_ref.semantic_anchor_logits(chunk.to(device), temperature=temp)
        if logits is None:
            return None
        probs = F.softmax(logits.float(), dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        outputs.append(probs.detach().cpu())
    if not outputs:
        return None
    return torch.cat(outputs, dim=0)

# SDLC 第二步Compute Semantic Anchor Drift
def _semantic_anchor_drift(args, model, model_old, features_new, features_old,
                           stage_index=None):
    """以总变差距离度量新旧模型语义锚点分布漂移，返回值位于 [0, 1]。"""
    probs_new = _semantic_anchor_probs_from_features(args, model, features_new)
    probs_old = _semantic_anchor_probs_from_features(args, model_old, features_old)
    if (probs_old is not None
            and getattr(args, 'reference_comparison_policy', 'stage_shared')
            == 'permuted_old_slots'
            and getattr(args, 'permuted_old_slots_target', 'both') in ('both', 'sdlc')):
        model_ref = getattr(model, 'module', model)
        getter = getattr(model_ref, 'get_reference_slot_permutation', None)
        if not callable(getter) or stage_index is None:
            raise RuntimeError(
                'permuted_old_slots requires stage index and permutation provider')
        permutation = getter(stage_index, device=probs_old.device)
        if probs_old.dim() != 2 or probs_old.size(1) != permutation.numel():
            raise RuntimeError('SDLC anchor permutation shape mismatch')
        probs_old = probs_old.index_select(1, permutation)
        print('Comparison anchor slots permuted: role=sdlc_old stage={} permutation={}'.format(
            stage_index, permutation.detach().cpu().tolist()))
    if probs_new is None or probs_old is None or probs_new.shape != probs_old.shape:
        return torch.zeros((), dtype=torch.float32)
    # 概率分布的总变差距离，理论取值范围为 [0, 1]。
    return (0.5 * (probs_new - probs_old).abs().sum(dim=1).mean()).float().clamp(0.0, 1.0)

# SDLC 第三步Compute Adaptation Weights  （Layer-wise）
def _sa_ltkc_param_alpha(args, name, base_alpha):
    """依据参数所属层级，把全局 alpha 调整为逐参数融合系数。"""
    if not _sa_ltkc_enabled(args):
        return float(base_alpha)

    min_alpha = float(getattr(args, 'sa_ltkc_min_alpha', 0.55))
    max_alpha = float(getattr(args, 'sa_ltkc_max_alpha', 0.98))

    def clamp_alpha(value):
        return min(max(float(value), min_alpha), max_alpha)

    normalized_name = name[7:] if name.startswith('module.') else name
    # 冻结文本塔与固定锚点保持当前值，不参与新旧视觉参数折中。
    if (normalized_name.startswith('text_encoder.')
            or normalized_name.startswith('text_anchor_')
            or normalized_name.startswith('semantic_reference_')):
        return 1.0
    # 分类头和 BN 使用独立系数，避免类别扩展时被过度回拉。
    if (normalized_name.startswith('classifier.')
            or normalized_name.startswith('bottleneck.')):
        return clamp_alpha(getattr(args, 'sa_ltkc_head_alpha', 0.95))

    # ViT 深层更贴近任务语义：从浅到深线性增大 alpha 缩放系数。
    marker = 'image_encoder.transformer.resblocks.'
    if marker in normalized_name:
        suffix = normalized_name.split(marker, 1)[1]
        block_text = suffix.split('.', 1)[0]
        if block_text.isdigit():
            block_idx = int(block_text)
            num_blocks = max(1, int(getattr(args, 'sa_ltkc_num_blocks', 12)))
            progress = float(block_idx) / float(max(num_blocks - 1, 1))
            low_scale = float(getattr(args, 'sa_ltkc_low_scale', 0.85))
            high_scale = float(getattr(args, 'sa_ltkc_high_scale', 1.05))
            scale = low_scale + (high_scale - low_scale) * progress
            return clamp_alpha(float(base_alpha) * scale)

    if normalized_name.startswith('image_encoder.'):
        return clamp_alpha(float(base_alpha) * float(getattr(args, 'sa_ltkc_low_scale', 0.85)))
    return clamp_alpha(base_alpha)

# SDLC第四、五阶段，融合模型
def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    """在域训练结束后执行 SA-LTKC 新旧参数融合，并返回下一阶段模型。"""
    # 上一阶段冻结模型参数。
    model_old_state_dict = model_old.state_dict()
    # 当前域刚训练完成的模型参数。
    model_state_dict = model.state_dict()

    # 在副本上融合，避免原地修改当前模型或旧模型。
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    # 对每个可匹配的浮点参数执行 alpha*当前 + (1-alpha)*历史。
    alpha_stats = {}
    for k, v in model_state_dict.items():
        if k not in model_old_state_dict:
            continue
        normalized_name = k[7:] if k.startswith('module.') else k
        if (normalized_name.startswith('text_encoder.')
                or normalized_name.startswith('text_anchor_')
                or normalized_name.startswith('semantic_reference_')):
            # The reference is an experimental input. Copy it bit-for-bit and
            # never turn it into an EMA/interpolation result.
            model_new_state_dict[k] = v.clone()
            continue
        # 整数/布尔缓冲区（如计数器）直接采用当前模型值。
        if not torch.is_floating_point(v):
            model_new_state_dict[k] = v
            continue
        param_alpha = _sa_ltkc_param_alpha(args, k, alpha)
        if _sa_ltkc_enabled(args):
            if (normalized_name.startswith('classifier.')
                    or normalized_name.startswith('bottleneck.')):
                group = 'head'
            elif 'image_encoder.transformer.resblocks.' in normalized_name:
                group = 'vit_block'
            elif normalized_name.startswith('image_encoder.'):
                group = 'vit_shared'
            else:
                group = 'other'
            alpha_stats.setdefault(group, []).append(param_alpha)
        if model_old_state_dict[k].shape == v.shape:
            model_new_state_dict[k] = param_alpha * v + (1 - param_alpha) * model_old_state_dict[k]
        else:
            # 分类类别扩展会导致首维变大，只融合旧类别对应的参数切片。
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = (
                param_alpha * v[:num_class_old]
                + (1 - param_alpha) * model_old_state_dict[k]
            )
    model_new.load_state_dict(model_new_state_dict)
    if _sa_ltkc_enabled(args) and alpha_stats:
        stat_text = []
        for group in sorted(alpha_stats.keys()):
            values = alpha_stats[group]
            stat_text.append('{}={:.3f}'.format(group, sum(values) / max(len(values), 1)))
        print('SA-LTKC layer alpha: {}'.format(' '.join(stat_text)))
    return model_new



def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs          
    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains
    
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        _set_comparison_reference(args, old_model, set_index, role='rsrt_old')
        old_model = old_model.cuda()
        old_model.eval()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    
        # Initialize classifer with class centers    
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        if hasattr(model.module, 'num_classes'):
            model.module.num_classes = add_num + num_classes
        model.cuda()

    # Re-initialize optimizer
    params = build_optimizer_params(model, args)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=args.warmup_factor, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(cfg, args, model, add_num + num_classes,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,
                      )
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
                'semantic_reference': _semantic_reference_metadata(model),
            }, True, fpath=osp.join(args.pth_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)                    
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
                'semantic_reference': _semantic_reference_metadata(model),
            }, True, fpath=osp.join(args.pth_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model 

def test_model(model, all_train_sets, all_test_sets, set_index,  logger_res=None):
    begin = 0
    evaluator = Evaluator(model)
        
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    train_mAP=0
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))

        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)  # ,training_phase=i+1)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()


    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_test_sets[i]
        print('Results on {}'.format(name))
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True)

        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)

    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_R1_unseen = torch.tensor(R1_all).mean()

    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x', 'clip_vit'])
    parser.add_argument('--semantic-anchor-count', type=int, default=25,
                        help="number of fixed semantic anchors used by CLIP-ViT")
    parser.add_argument('--semantic-anchor-wording', type=str, default='default',
                        choices=['default', 'terse', 'detailed'],
                        help="wording variant for fixed semantic anchor prompts")
    parser.add_argument('--semantic-reference', type=str, default='semantic',
                        choices=['none', 'random_vector', 'stage_random_vector',
                                 'irrelevant_text', 'semantic'],
                        help="fixed reference source shared by all semantic modules")
    parser.add_argument('--reference-seed', type=int, default=0,
                        help="independent seed for normalized random-vector references")
    parser.add_argument('--reference-comparison-policy', type=str, default='stage_shared',
                        choices=['stage_shared', 'previous_stage_mismatch',
                                 'permuted_old_slots'],
                        help=('bank used by old/EMA comparison branches: current stage bank '
                              'or the deliberately mismatched previous-stage bank'))
    parser.add_argument('--permuted-old-slots-target', type=str, default='both',
                        choices=['both', 'sdac', 'sdlc'],
                        help=('when --reference-comparison-policy=permuted_old_slots, choose '
                              'which direct old/current anchor comparisons receive the old-slot '
                              'permutation'))
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--head-lr', type=float, default=None,
                        help="learning rate of classifier/bottleneck, default: same as --lr, or 3.5e-4 for clip_vit")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--warmup-factor', type=float, default=0.01)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/root/data/lreid')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('logs/try'))
    parser.add_argument('--reproduce-root', type=str, default=None,
                        help="optional root for logs, checkpoints and code snapshots")

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")
    parser.add_argument('--sa-rstkt', dest='sa_rstkt', action='store_true', default=False,
                        help="enable Semantic-Anchored Reliability-aware R-STKT")
    parser.add_argument('--no-sa-rstkt', dest='sa_rstkt', action='store_false',
                        help="disable Semantic-Anchored Reliability-aware R-STKT")
    parser.add_argument('--sa-rstkt-weight', default=0.08, type=float,
                        help="SA-RSTKT semantic target calibration strength")
    parser.add_argument('--sa-rstkt-text-temp', default=0.07, type=float,
                        help="temperature for SA-RSTKT CLIP text-anchor distribution")
    parser.add_argument('--sa-rstkt-semantic-weight', default=0.5, type=float,
                        help="same-ID semantic-anchor pull strength inside SA-RSTKT")
    parser.add_argument('--sa-rstkt-neg-weight', default=0.2, type=float,
                        help="different-ID semantic-anchor suppression strength inside SA-RSTKT")
    parser.add_argument('--sa-rstkt-conf-floor', default=0.05, type=float,
                        help="minimum pair reliability for SA-RSTKT weighted relation KL")
    parser.add_argument('--sa-rstkt-consistency-power', default=1.0, type=float,
                        help="power for old-new visual consistency reliability in SA-RSTKT")
    parser.add_argument('--sa-rstkt-start-phase', default=2, type=int,
                        help="training phase from which SA-RSTKT is enabled")
    parser.add_argument('--sa-rstkt-start-epoch', default=5, type=int,
                        help="epoch to start SA-RSTKT inside each incremental domain")
    parser.add_argument('--sa-rstkt-warmup-epochs', default=10, type=int,
                        help="epochs to linearly warm up SA-RSTKT strength")
    parser.add_argument('--sa-ltkc', dest='sa_ltkc', action='store_true', default=False,
                        help="enable Semantic-drift-aware layer-wise long-term knowledge consolidation")
    parser.add_argument('--no-sa-ltkc', dest='sa_ltkc', action='store_false',
                        help="disable Semantic-drift-aware layer-wise long-term knowledge consolidation")
    parser.add_argument('--sa-ltkc-relation-weight', default=1.0, type=float,
                        help="weight of visual relation drift when estimating SA-LTKC alpha")
    parser.add_argument('--sa-ltkc-semantic-weight', default=0.5, type=float,
                        help="weight of CLIP semantic-anchor drift when estimating SA-LTKC alpha")
    parser.add_argument('--sa-ltkc-text-temp', default=0.07, type=float,
                        help="temperature for SA-LTKC CLIP text-anchor distributions")
    parser.add_argument('--sa-ltkc-alpha-schedule', default=None, type=str,
                        help=('optional four-value comma-separated alpha schedule for '
                              'the four incremental transitions'))
    parser.add_argument('--sa-ltkc-semantic-chunk', default=512, type=int,
                        help="chunk size for SA-LTKC semantic drift estimation")
    parser.add_argument('--sa-ltkc-min-alpha', default=0.55, type=float,
                        help="minimum new-model alpha for SA-LTKC parameter fusion")
    parser.add_argument('--sa-ltkc-max-alpha', default=0.98, type=float,
                        help="maximum new-model alpha for SA-LTKC parameter fusion")
    parser.add_argument('--sa-ltkc-low-scale', default=0.85, type=float,
                        help="SA-LTKC alpha multiplier for lower/general visual layers")
    parser.add_argument('--sa-ltkc-high-scale', default=1.05, type=float,
                        help="SA-LTKC alpha multiplier for higher/specialized visual layers")
    parser.add_argument('--sa-ltkc-head-alpha', default=0.95, type=float,
                        help="new-model alpha for classifier and bottleneck in SA-LTKC")
    parser.add_argument('--sa-ltkc-num-blocks', default=12, type=int,
                        help="number of ViT transformer blocks used by SA-LTKC layer scaling")
    parser.add_argument('--gscm', dest='gscm', action='store_true', default=False,
                        help="enable fixed-anchor Grouped Semantic Cross-camera Margin")
    parser.add_argument('--no-gscm', dest='gscm', action='store_false',
                        help="disable Grouped Semantic Cross-camera Margin")
    parser.add_argument('--gscm-weight', default=0.10, type=float,
                        help="overall GSCM auxiliary-loss weight")
    parser.add_argument('--gscm-visual-temp', default=0.07, type=float,
                        help="temperature for GSCM visual pair logits")
    parser.add_argument('--gscm-text-temp', default=0.07, type=float,
                        help="temperature for fixed CLIP anchors used by GSCM")
    parser.add_argument('--gscm-base-margin', default=0.05, type=float,
                        help="base margin applied to different-identity pairs")
    parser.add_argument('--gscm-semantic-margin', default=0.10, type=float,
                        help="extra margin for semantic look-alike negatives")
    parser.add_argument('--gscm-anchor-groups', default='semantic',
                        choices=['semantic', 'global', 'shuffled'],
                        help="anchor grouping used by GSCM")
    parser.add_argument('--gscm-entropy-confidence',
                        dest='gscm_entropy_confidence', action='store_true',
                        default=True,
                        help="weight GSCM attribute groups by entropy confidence")
    parser.add_argument('--no-gscm-entropy-confidence',
                        dest='gscm_entropy_confidence', action='store_false',
                        help="use uniform confidence for GSCM attribute groups")
    parser.add_argument('--gscm-cross-camera', dest='gscm_cross_camera',
                        action='store_true', default=True,
                        help="prefer cross-camera positives in GSCM")
    parser.add_argument('--no-gscm-cross-camera', dest='gscm_cross_camera',
                        action='store_false',
                        help="use all same-identity positives in GSCM")
    parser.add_argument('--gscm-start-phase', default=1, type=int,
                        help="training phase from which GSCM is enabled")
    parser.add_argument('--gscm-start-epoch', default=0, type=int,
                        help="epoch to start GSCM inside each domain")
    parser.add_argument('--gscm-warmup-epochs', default=5, type=int,
                        help="epochs to linearly warm up the GSCM weight")
    parser.add_argument('--sdac', dest='sdac', action='store_true', default=False,
                        help="enable Self-Distilled Anchor Consistency for CLIP continual learning")
    parser.add_argument('--no-sdac', dest='sdac', action='store_false',
                        help="disable Self-Distilled Anchor Consistency")
    parser.add_argument('--sdac-weight', default=0.0, type=float,
                        help="overall SDAC loss weight")
    parser.add_argument('--sdac-text-weight', default=1.0, type=float,
                        help="SDAC CLIP text-anchor reverse-KL weight")
    parser.add_argument('--sdac-feat-weight', default=0.25, type=float,
                        help="SDAC EMA feature consistency weight")
    parser.add_argument('--sdac-logit-weight', default=0.5, type=float,
                        help="SDAC EMA classifier-logit consistency weight")
    parser.add_argument('--sdac-temp', default=4.0, type=float,
                        help="temperature for SDAC classifier-logit reverse-KL")
    parser.add_argument('--sdac-text-temp', default=0.07, type=float,
                        help="temperature for SDAC CLIP text-anchor logits")
    parser.add_argument('--sdac-ema', default=0.997, type=float,
                        help="EMA momentum for SDAC teacher")
    parser.add_argument('--sdac-ref-guidance', dest='sdac_ref_guidance',
                        action='store_true', default=False,
                        help="enable reference-conditioned risk weighting for SDAC feature/logit consistency")
    parser.add_argument('--no-sdac-ref-guidance', dest='sdac_ref_guidance',
                        action='store_false',
                        help="use vanilla uniform SDAC feature/logit consistency")
    parser.add_argument('--sdac-ref-gamma', default=0.20, type=float,
                        help="maximum bounded modulation strength; weights are approximately 1 +/- gamma")
    parser.add_argument('--sdac-ref-start-epoch', default=5, type=int,
                        help="epoch to start reference-conditioned consistency inside each incremental domain")
    parser.add_argument('--sdac-ref-warmup-epochs', default=10, type=int,
                        help="epochs to linearly ramp reference conditioning from 0 to sdac-ref-gamma")
    parser.add_argument('--sdac-ref-std-eps', default=1e-4, type=float,
                        help="fallback to vanilla SDAC when within-batch reference-risk std is below this value")
    parser.add_argument('--sdac-ref-floor', default=0.05, type=float,
                        help="legacy option retained for old scripts; unused by reference-conditioned SDAC")
    parser.add_argument('--sdac-start-phase', default=2, type=int,
                        help="training phase from which SDAC is enabled")
    main()
