"""CLIP ViT backbone with fixed reference controls for lifelong ReID.

The current project keeps a prompt-free method:
CLIP-ViT + SDAC/GSCM + SA-RSTKT + SA-LTKC.  The CLIP text tower is
frozen and provides pedestrian or unrelated text anchors. Supplementary
controls can instead use fixed random vectors or disable the reference.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip


TEXT_ANCHOR_SPECS = (
    ('generic', (
        'a surveillance photo of a pedestrian.',
        'a pedestrian in a surveillance image.',
        'a camera image showing one pedestrian.',
    )),
    ('upper_color', (
        'a photo of a person wearing a black top.',
        'a pedestrian with a black upper garment.',
        'a person whose upper clothing is black.',
    )),
    ('upper_color', (
        'a photo of a person wearing a white top.',
        'a pedestrian with a white upper garment.',
        'a person whose upper clothing is white.',
    )),
    ('upper_color', (
        'a photo of a person wearing a red top.',
        'a pedestrian with a red upper garment.',
        'a person whose upper clothing is red.',
    )),
    ('upper_color', (
        'a photo of a person wearing a blue top.',
        'a pedestrian with a blue upper garment.',
        'a person whose upper clothing is blue.',
    )),
    ('upper_color', (
        'a photo of a person wearing a green top.',
        'a pedestrian with a green upper garment.',
        'a person whose upper clothing is green.',
    )),
    ('upper_color', (
        'a photo of a person wearing a yellow top.',
        'a pedestrian with a yellow upper garment.',
        'a person whose upper clothing is yellow.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing dark pants.',
        'a pedestrian with dark lower clothing.',
        'a person whose lower clothing is dark.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing light pants.',
        'a pedestrian with light lower clothing.',
        'a person whose lower clothing is light.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing shorts.',
        'a pedestrian wearing shorts.',
        'a person whose lower garment is shorts.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing a skirt.',
        'a pedestrian wearing a skirt.',
        'a person whose lower garment is a skirt.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing long pants.',
        'a pedestrian wearing long trousers.',
        'a person whose lower garment is long pants.',
    )),
    ('carried_object', (
        'a photo of a person carrying a backpack.',
        'a pedestrian with a backpack.',
        'a person carrying a backpack on the body.',
    )),
    ('carried_object', (
        'a photo of a person carrying a shoulder bag.',
        'a pedestrian with a shoulder bag.',
        'a person carrying a bag over the shoulder.',
    )),
    ('carried_object', (
        'a photo of a person carrying no bag.',
        'a pedestrian without a visible bag.',
        'a person carrying no obvious bag.',
    )),
    ('head_appearance', (
        'a photo of a person wearing a hat.',
        'a pedestrian wearing a hat.',
        'a person with a hat on the head.',
    )),
    ('head_appearance', (
        'a photo of a person with short hair.',
        'a pedestrian with short hair.',
        'a person whose hair is short.',
    )),
    ('head_appearance', (
        'a photo of a person with long hair.',
        'a pedestrian with long hair.',
        'a person whose hair is long.',
    )),
    ('viewpoint', (
        'a front view photo of a pedestrian.',
        'a pedestrian viewed from the front.',
        'a person facing the camera.',
    )),
    ('viewpoint', (
        'a back view photo of a pedestrian.',
        'a pedestrian viewed from the back.',
        'a person facing away from the camera.',
    )),
    ('viewpoint', (
        'a side view photo of a pedestrian.',
        'a pedestrian viewed from the side.',
        'a person seen in side view.',
    )),
    ('action', (
        'a photo of a walking pedestrian.',
        'a pedestrian who is walking.',
        'a person walking in the scene.',
    )),
    ('action', (
        'a photo of a standing pedestrian.',
        'a pedestrian who is standing.',
        'a person standing in the scene.',
    )),
    ('quality_occlusion', (
        'a clear photo of a pedestrian.',
        'a clear surveillance image of a pedestrian.',
        'a person image with little blur or occlusion.',
    )),
    ('quality_occlusion', (
        'an occluded photo of a pedestrian.',
        'a surveillance image of an occluded pedestrian.',
        'a person image with visible occlusion.',
    )),
    ('upper_color', (
        'a photo of a person wearing a gray top.',
        'a pedestrian with a gray upper garment.',
        'a person whose upper clothing is gray.',
    )),
    ('upper_color', (
        'a photo of a person wearing a patterned top.',
        'a pedestrian with a patterned upper garment.',
        'a person whose upper clothing has patterns.',
    )),
    ('lower_appearance', (
        'a photo of a person wearing jeans.',
        'a pedestrian wearing jeans.',
        'a person whose lower garment is jeans.',
    )),
    ('carried_object', (
        'a photo of a person carrying a handbag.',
        'a pedestrian with a handbag.',
        'a person carrying a handbag.',
    )),
    ('carried_object', (
        'a photo of a person carrying an umbrella.',
        'a pedestrian with an umbrella.',
        'a person carrying an umbrella.',
    )),
    ('head_appearance', (
        'a photo of a person wearing glasses.',
        'a pedestrian wearing glasses.',
        'a person with glasses on the face.',
    )),
    ('head_appearance', (
        'a photo of a person wearing a face mask.',
        'a pedestrian wearing a face mask.',
        'a person with a face mask.',
    )),
    ('viewpoint', (
        'a three-quarter view photo of a pedestrian.',
        'a pedestrian viewed from an oblique angle.',
        'a person seen from a diagonal viewpoint.',
    )),
    ('action', (
        'a photo of a running pedestrian.',
        'a pedestrian who is running.',
        'a person running in the scene.',
    )),
    ('quality_occlusion', (
        'a low-resolution photo of a pedestrian.',
        'a blurry surveillance image of a pedestrian.',
        'a person image with low visual quality.',
    )),
)

TEXT_ANCHOR_PROMPTS = tuple(spec[1][0] for spec in TEXT_ANCHOR_SPECS[:25])

# Backward-compatible name for existing visualization utilities.
SRAR_TEXT_PROMPTS = TEXT_ANCHOR_PROMPTS

SEMANTIC_REFERENCE_TYPE_IDS = {
    'semantic': 0,
    'random_vector': 1,
    'irrelevant_text': 2,
    'none': 3,
    'stage_random_vector': 4,
}

# These concepts deliberately avoid pedestrians, clothing, poses and accessories.
# Their slot positions retain the original group sizes for the source control.
IRRELEVANT_TEXT_CONCEPTS = (
    'a spiral galaxy', 'a lunar crater', 'a coral reef', 'a waterfall',
    'a volcano', 'a snowflake', 'a sand dune', 'a glacier', 'a thundercloud',
    'a rainbow', 'a comet', 'an asteroid', 'a crystal', 'a seashell',
    'a pine cone', 'a sunflower', 'a cactus', 'a mushroom', 'a fern',
    'an oak tree', 'a bamboo grove', 'a mossy rock', 'a frozen lake',
    'a mountain peak', 'a desert canyon', 'a lava field', 'an ocean wave',
    'a river delta', 'a limestone cave', 'a salt flat', 'a hot spring',
    'a star cluster', 'a nebula', 'a solar eclipse', 'a quartz geode',
)
IRRELEVANT_TEXT_ANCHOR_SPECS = tuple(
    (semantic_spec[0], (
        'a photo of {}.'.format(concept),
        'an image showing {}.'.format(concept),
        'a picture of {}.'.format(concept),
    ))
    for semantic_spec, concept in zip(TEXT_ANCHOR_SPECS, IRRELEVANT_TEXT_CONCEPTS)
)


def build_text_anchor_prompts(count=25, wording='default', reference='semantic'):
    """Return the selected fixed anchor prompts for sensitivity experiments."""
    count = int(count)
    if count <= 0 or count > len(TEXT_ANCHOR_SPECS):
        raise ValueError('semantic_anchor_count must be in [1, {}]'.format(
            len(TEXT_ANCHOR_SPECS)))
    wording_to_index = {'default': 0, 'terse': 1, 'detailed': 2}
    if wording not in wording_to_index:
        raise ValueError('unknown semantic_anchor_wording: {}'.format(wording))
    if reference not in ('semantic', 'irrelevant_text'):
        raise ValueError('text prompts are unavailable for reference: {}'.format(reference))
    prompt_index = wording_to_index[wording]
    specs = (TEXT_ANCHOR_SPECS if reference == 'semantic'
             else IRRELEVANT_TEXT_ANCHOR_SPECS)
    return tuple(spec[1][prompt_index] for spec in specs[:count])


def _shuffled_groups_from_groups(groups):
    attrs = [idx for group in groups for idx in group]
    if not attrs:
        return tuple()
    permuted = list(reversed(attrs[::2] + attrs[1::2]))
    shuffled, cursor = [], 0
    for group in groups:
        group_len = len(group)
        shuffled.append(tuple(permuted[cursor:cursor + group_len]))
        cursor += group_len
    return tuple(shuffled)


def build_text_anchor_groups(count=25):
    """Build semantic/global/shuffled groups for the selected anchor prefix."""
    group_order = []
    grouped = {}
    for idx, (name, _) in enumerate(TEXT_ANCHOR_SPECS[:int(count)]):
        if idx == 0:
            continue
        if name not in grouped:
            group_order.append(name)
            grouped[name] = []
        grouped[name].append(idx)

    semantic = tuple(tuple(grouped[name]) for name in group_order
                     if len(grouped[name]) > 1)
    global_group = (tuple(idx for idx in range(1, int(count))),)
    shuffled = _shuffled_groups_from_groups(semantic)
    return semantic, global_group, shuffled


def _load_clip(backbone, height, width, stride):
    h_res = (height - 16) // stride + 1
    w_res = (width - 16) // stride + 1
    path = clip._download(clip._MODELS[backbone])
    try:
        model = torch.jit.load(path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(path, map_location='cpu')
    return clip.build_model(state_dict or model.state_dict(), h_res, w_res, stride).float()


class CLIPTextEncoder(nn.Module):
    """Light wrapper around the CLIP text tower; parameters stay frozen."""

    def __init__(self, clip_model):
        super(CLIPTextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_text):
        x = self.token_embedding(tokenized_text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot = tokenized_text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection
        return x


class CLIPViTBackbone(nn.Module):
    """CLIP ViT-B/16 model used by the final prompt-free method."""

    def __init__(self, args, num_classes):
        super(CLIPViTBackbone, self).__init__()
        self.model_name = 'ViT-B-16'
        self.num_classes = int(num_classes)

        print('using CLIP ViT-B/16 as a backbone')
        clip_model = _load_clip(
            self.model_name,
            height=int(args.height),
            width=int(args.width),
            stride=16,
        )
        self.out_channel = int(clip_model.text_projection.shape[-1])
        self.image_encoder = clip_model.visual
        self.text_encoder = CLIPTextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        anchor_count = int(getattr(args, 'semantic_anchor_count', 25))
        anchor_wording = getattr(args, 'semantic_anchor_wording', 'default')
        self.semantic_reference = getattr(args, 'semantic_reference', 'semantic')
        self.reference_seed = int(getattr(args, 'reference_seed', 0))
        if self.semantic_reference not in SEMANTIC_REFERENCE_TYPE_IDS:
            raise ValueError('unknown semantic_reference: {}'.format(self.semantic_reference))
        # Validate the shared count/wording even for the vector and no-reference
        # controls, without constructing pedestrian prompts for those controls.
        if anchor_count <= 0 or anchor_count > len(TEXT_ANCHOR_SPECS):
            raise ValueError('semantic_anchor_count must be in [1, {}]'.format(
                len(TEXT_ANCHOR_SPECS)))
        if anchor_wording not in ('default', 'terse', 'detailed'):
            raise ValueError('unknown semantic_anchor_wording: {}'.format(anchor_wording))
        self.semantic_anchor_count = anchor_count
        self.semantic_anchor_wording = anchor_wording
        has_text = self.semantic_reference in ('semantic', 'irrelevant_text')
        anchor_prompts = (build_text_anchor_prompts(
            anchor_count, anchor_wording, reference=self.semantic_reference)
            if has_text else tuple())
        groups = (build_text_anchor_groups(anchor_count)
                  if self.semantic_reference != 'none' else (tuple(), tuple(), tuple()))
        self.semantic_anchor_prompts = anchor_prompts
        self.semantic_anchor_groups = groups[0]
        self.semantic_anchor_groups_global = groups[1]
        self.semantic_anchor_groups_shuffled = groups[2]
        effective_count = anchor_count if self.semantic_reference != 'none' else 0
        print('Fixed reference: source={} count={} wording={} seed={} groups={}'.format(
            self.semantic_reference, effective_count, anchor_wording,
            self.reference_seed, self.semantic_anchor_groups))

        anchor_tokens = (clip.tokenize(list(anchor_prompts), truncate=True)
                         if has_text else torch.empty(0, 0, dtype=torch.long))
        self.register_buffer('text_anchor_tokens', anchor_tokens, persistent=False)
        # A dedicated CPU generator leaves the training RNG stream unchanged.
        # All variants therefore initialize their visual/classifier weights alike.
        if self.semantic_reference in ('random_vector', 'stage_random_vector'):
            anchor_vectors = self._make_random_reference(stage_index=0)
        else:
            anchor_vectors = torch.zeros(effective_count, self.out_channel)
        self.register_buffer('text_anchor_vectors', anchor_vectors)
        self.register_buffer('text_anchor_vectors_ready', torch.tensor(
            not has_text, dtype=torch.bool))
        self.register_buffer('semantic_reference_type_id', torch.tensor(
            SEMANTIC_REFERENCE_TYPE_IDS[self.semantic_reference], dtype=torch.long))
        self.register_buffer('semantic_reference_seed', torch.tensor(
            self.reference_seed, dtype=torch.long))
        self.register_buffer('semantic_reference_stage', torch.tensor(
            0 if self.semantic_reference == 'stage_random_vector' else -1,
            dtype=torch.long))
        self._cached_text_anchors = None
        self._cached_anchor_device = None
        self._cached_semantic_subspace = {}
        # Training-only, stage-specific basis constructed from the frozen old
        # visual model and fixed semantic groups.  It is deliberately excluded
        # from checkpoints and inference.
        self.register_buffer('_srsr_basis', torch.empty(0, self.out_channel),
                             persistent=False)

        self.bottleneck = nn.BatchNorm1d(self.out_channel)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = nn.Linear(self.out_channel, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def _make_random_reference(self, stage_index=0):
        """Create a normalized random bank without advancing the training RNG."""
        generator = torch.Generator(device='cpu')
        stage_seed = int(self.reference_seed) + int(stage_index)
        generator.manual_seed(stage_seed % (2 ** 63 - 1))
        return F.normalize(torch.randn(
            self.semantic_anchor_count, self.out_channel, generator=generator),
            p=2, dim=1)

    def get_reference_slot_permutation(self, stage_index, device=None):
        """Return a deterministic non-identity permutation without touching train RNG."""
        count = int(self.text_anchor_vectors.shape[0])
        if count <= 1:
            return torch.arange(count, dtype=torch.long, device=device)
        generator = torch.Generator(device='cpu')
        permutation_seed = self.reference_seed + 1000003 + int(stage_index)
        generator.manual_seed(permutation_seed % (2 ** 63 - 1))
        permutation = torch.randperm(count, generator=generator)
        # The sanity control must never silently degenerate to identity.
        if torch.equal(permutation, torch.arange(count)):
            permutation = torch.roll(permutation, shifts=1)
        return permutation.to(device=device)

    @torch.no_grad()
    def refresh_stage_reference(self, stage_index):
        """Refresh only the stage-varying control; all other banks stay fixed."""
        if self.semantic_reference != 'stage_random_vector':
            return False
        stage_index = int(stage_index)
        anchors = self._make_random_reference(stage_index).to(
            device=self.text_anchor_vectors.device,
            dtype=self.text_anchor_vectors.dtype)
        self.text_anchor_vectors.copy_(anchors)
        self.text_anchor_vectors_ready.fill_(True)
        self.semantic_reference_stage.fill_(stage_index)
        self._cached_text_anchors = None
        self._cached_anchor_device = None
        self._cached_semantic_subspace = {}
        return True

    def _extract_features(self, x):
        feat_pen, feat_last, feat_proj = self.image_encoder(x)
        cls_feat = feat_proj[:, 0].float()
        global_feat = F.normalize(cls_feat, p=2, dim=1)
        final_tokens = feat_last[:, 0].detach().float()
        bn_feat = self.bottleneck(global_feat)
        return global_feat, bn_feat, final_tokens

    def forward(self, x, domains=None, training_phase=None, get_all_feat=False, epoch=0):
        global_feat, bn_feat, final_tokens = self._extract_features(x)
        cls_outputs = self.classifier(bn_feat)

        if get_all_feat is True:
            return global_feat, bn_feat, cls_outputs, final_tokens
        if self.training is False:
            return global_feat
        return global_feat, bn_feat, cls_outputs, final_tokens

    @torch.no_grad()
    def get_text_anchors(self):
        if self.semantic_reference == 'none':
            return None
        anchor_device = self.text_anchor_vectors.device
        if (self._cached_text_anchors is self.text_anchor_vectors
                and self._cached_anchor_device == anchor_device):
            return self._cached_text_anchors

        if not bool(self.text_anchor_vectors_ready.item()):
            self.text_encoder.eval()
            anchors = self.text_encoder(self.text_anchor_tokens)
            anchors = F.normalize(anchors.float(), p=2, dim=1)
            self.text_anchor_vectors.copy_(anchors)
            self.text_anchor_vectors_ready.fill_(True)
        self._cached_text_anchors = self.text_anchor_vectors
        self._cached_anchor_device = anchor_device
        return self.text_anchor_vectors

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Older checkpoints generated anchors on demand and contain none of
        # these buffers. Retain that behavior while supporting strict loading.
        legacy_text_reference = (
            prefix + 'text_anchor_vectors' not in state_dict
            and self.semantic_reference in ('semantic', 'irrelevant_text'))
        for name in ('text_anchor_vectors', 'text_anchor_vectors_ready',
                     'semantic_reference_type_id', 'semantic_reference_seed',
                     'semantic_reference_stage'):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        if legacy_text_reference:
            # Rebuild using the loaded frozen text tower, including when the
            # destination model happened to materialize anchors before loading.
            state_dict[prefix + 'text_anchor_vectors'].zero_()
            state_dict[prefix + 'text_anchor_vectors_ready'].fill_(False)
        self._cached_text_anchors = None
        self._cached_anchor_device = None
        self._cached_semantic_subspace = {}
        super(CLIPViTBackbone, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def get_semantic_reference_metadata(self):
        """Return JSON-serializable provenance and a checksum of frozen vectors."""
        anchors = self.get_text_anchors()
        digest = None
        if anchors is not None:
            digest = hashlib.sha256(
                anchors.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {
            'source': self.semantic_reference,
            'seed': self.reference_seed,
            'stage': int(self.semantic_reference_stage.item()),
            'refresh_policy': ('per_stage' if self.semantic_reference == 'stage_random_vector'
                               else 'fixed'),
            'count': int(self.text_anchor_vectors.shape[0]),
            'configured_count': self.semantic_anchor_count,
            'dimension': self.out_channel,
            'wording': self.semantic_anchor_wording,
            'template_policy': 'one template selected by wording; no averaging',
            'prompts': list(self.semantic_anchor_prompts),
            'groups': [list(group) for group in self.semantic_anchor_groups],
            'sha256': digest,
        }

    # Backward-compatible name used by the visualization scripts.
    get_srar_text_anchors = get_text_anchors

    def semantic_anchor_logits(self, image_features, temperature=0.07):
        anchors = self.get_text_anchors()
        if anchors is None:
            return None
        anchors = anchors.detach()
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        temperature = max(float(temperature), 1e-6)
        return image_features @ anchors.t() / temperature

    @torch.no_grad()
    def get_semantic_subspace_basis(self, energy=0.95, construction='group_centered',
                                    rank=0):
        """Return orthonormal row bases in the 512-D CLIP feature space.

        ``group_centered`` removes the shared prompt/pedestrian direction inside
        each attribute group before SVD.  Consequently the retained directions
        encode contrasts such as black-vs-white or front-vs-back instead of the
        common wording shared by every prompt.  The returned tensor has shape
        [rank, feature_dim], so projection is performed without materializing a
        feature_dim x feature_dim matrix: ``(z @ basis.T) @ basis``.
        """
        energy = float(energy)
        if not 0.0 < energy <= 1.0:
            raise ValueError('semantic subspace energy must be in (0, 1]')
        if construction not in ('group_centered', 'raw'):
            raise ValueError('unknown semantic subspace construction: {}'.format(
                construction))
        anchors = self.get_text_anchors()
        if anchors is None or anchors.numel() == 0:
            return None

        rank = int(rank)
        if rank < 0:
            raise ValueError('semantic subspace rank must be non-negative')
        cache_key = (construction, round(energy, 8), rank,
                     anchors.device, anchors.dtype)
        cached = self._cached_semantic_subspace.get(cache_key)
        if cached is not None:
            return cached

        # This basis is built once, has at most a few dozen rows, and is then
        # cached.  Run the decomposition on CPU to avoid relying on accelerator
        # backends for a one-off small SVD.
        matrix = anchors.detach().float().cpu()
        if construction == 'group_centered':
            contrasts = []
            for group in self.semantic_anchor_groups:
                if len(group) < 2:
                    continue
                group_vectors = matrix[list(group)]
                contrasts.append(group_vectors - group_vectors.mean(
                    dim=0, keepdim=True))
            if not contrasts:
                return None
            matrix = torch.cat(contrasts, dim=0)

        # For A = U S V^T with A in R^{K x d}, rows of V^T are the bases in
        # the d-dimensional CLIP feature space.  Discard numerical null modes.
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        tolerance = (singular_values.max() * max(matrix.shape)
                     * torch.finfo(singular_values.dtype).eps)
        nonzero = singular_values > tolerance
        singular_values = singular_values[nonzero]
        vh = vh[nonzero]
        if singular_values.numel() == 0:
            return None
        cumulative = singular_values.square().cumsum(0)
        cumulative = cumulative / cumulative[-1].clamp_min(1e-12)
        retained_rank = (min(rank, int(singular_values.numel())) if rank > 0
                         else int(torch.searchsorted(
                             cumulative, cumulative.new_tensor(energy),
                             right=False).item()) + 1)
        basis = vh[:retained_rank].to(
            device=anchors.device, dtype=torch.float32).contiguous()
        self._cached_semantic_subspace[cache_key] = basis
        print(('Semantic contrast subspace: construction={} energy={:.3f} '
               'requested_rank={} rank={}/{}').format(
                   construction, energy, rank, retained_rank,
                   int(singular_values.numel())))
        return basis

    def project_semantic_subspaces(self, image_features, energy=0.95,
                                   construction='group_centered', rank=0):
        """Split features into orthogonal semantic and residual components."""
        basis = self.get_semantic_subspace_basis(
            energy=energy, construction=construction, rank=rank)
        if basis is None:
            raise RuntimeError('semantic subspace requires a nonempty reference bank')
        basis = basis.to(device=image_features.device, dtype=image_features.dtype)
        semantic = image_features.matmul(basis.t()).matmul(basis)
        residual = image_features - semantic
        return semantic, residual

    @torch.no_grad()
    def set_role_routed_subspace(self, basis):
        """Install one fixed stage-level semantic-role visual basis."""
        if basis is None:
            self._srsr_basis = self._srsr_basis.new_empty(0, self.out_channel)
            return
        basis = basis.detach().to(
            device=self.text_anchor_vectors.device, dtype=torch.float32)
        if basis.dim() != 2 or basis.size(1) != self.out_channel:
            raise ValueError('SRSR basis must have shape [rank, {}]'.format(
                self.out_channel))
        if basis.size(0) > 0:
            gram = basis.matmul(basis.t())
            eye = torch.eye(basis.size(0), device=gram.device, dtype=gram.dtype)
            if not torch.allclose(gram, eye, atol=2e-4, rtol=2e-4):
                raise ValueError('SRSR basis rows must be orthonormal')
        self._srsr_basis = basis.contiguous()

    def project_role_routed_subspaces(self, image_features):
        """Project onto the installed stage basis and its orthogonal residual."""
        basis = self._srsr_basis
        if basis.numel() == 0:
            raise RuntimeError('SRSR stage basis has not been prepared')
        basis = basis.to(device=image_features.device, dtype=image_features.dtype)
        stable = image_features.matmul(basis.t()).matmul(basis)
        return stable, image_features - stable

    @torch.no_grad()
    def semantic_anchor_scores(self, image_features, temperature=0.07):
        logits = self.semantic_anchor_logits(image_features, temperature=temperature)
        if logits is None:
            return None
        probs = F.softmax(logits.float(), dim=1)
        probs = F.normalize(probs, p=2, dim=1)
        return (probs @ probs.t()).clamp(0.0, 1.0)


def make_model(arg, num_class, camera_num, view_num, pretrain=True):
    print('===========building CLIP ViT-B/16===========')
    return CLIPViTBackbone(arg, num_class)
