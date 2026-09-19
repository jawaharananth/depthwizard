"""HTC-AdaBins head: adaptive-bin classification + head-tail cut for height.

Reference: Chen et al., "Vision Transformers for Monocular Height Estimation
via Content-Adaptive Bins and Class Balancing" (submitted to IEEE TGRS,
arXiv:2309.16486, sections III-B/C/D). Equation numbers in the docstrings
below refer to that paper; this module reimplements them independently
against our own DPT backbone (train_height.py's AutoModelForDepthEstimation),
not against any other team's code.

WHY THIS EXISTS

train_height.py's plain regression head (a single conv down to 1 channel,
trained with weighted L1) has a known, named failure mode in this literature:
a scalar regression head trained on a long-tailed height distribution (~57%
of pixels near 0 m, a handful of pixels above 30 m) is cheapest to solve by
predicting close to the background everywhere and eating the loss on the
rare tall pixels. STRUCTURE_WEIGHT in train_height.py fights this with a
loss reweighting, which helps but does not change what the head is *able*
to represent -- it is still one number per pixel with no notion that ground
and roofs come from different distributions.

The HTC-AdaBins module replaces the scalar head with a distribution: N
adaptive height bins whose edges are placed by the network per-image (not
fixed at training time), a foreground/background gate that gives ground and
building pixels their OWN bin-probability maps rather than forcing one head
to fit both, and a final continuous height as the probability-weighted mean
of the bin centres (Eqn. 11). Whether it actually beats train_height.py's
existing head on OUR data is an open, measured question -- see the
`licensing` gate in train_height.py, which this module does not bypass.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HTCAdaBinsHead(nn.Module):
    """Adaptive-bin classification head with a head-tail cut, over a DPT feature map.

    Args:
        in_channels: channel count of the feature map handed in (the DPT
            neck's output, before its own final 1x1 conv to a scalar).
        n_bins: number of adaptive bins (Eqn. 3-4). The paper's N; kept
            modest by default since our tiles are far fewer than GBH's.
        h_min, h_max: the domain the bins span, metres AGL. h_min slightly
            negative to give the lowest bin room for DSM/DTM noise around 0,
            matching gamus.py's own valid-pixel convention.
        vit_tokens: how many transformer tokens feed the global branch's
            foreground/background embeddings (the paper's m in Gfg/Gbg).
            Kept separate from the bin-width token count deliberately --
            Eqn. 2 uses e1 for bin widths and the *following* m tokens for
            the range-attention maps, so they must not overlap.
    """

    def __init__(self, in_channels: int, n_bins: int = 64,
                h_min: float = -3.0, h_max: float = 120.0,
                vit_tokens: int = 4, vit_heads: int = 4, vit_layers: int = 2):
        super().__init__()
        self.n_bins = int(n_bins)
        self.h_min, self.h_max = float(h_min), float(h_max)
        self.vit_tokens = int(vit_tokens)

        # Local branch, Eqn. 1: L = conv3x3(F).
        self.local_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                    padding=1)

        # Global branch: patchify the feature map down to a short token
        # sequence and refine it with a small transformer encoder, per
        # Eqn. 2. The paper uses a full ViT on its own patch embedding; we
        # reuse the DPT feature map itself (already a deep representation)
        # and patchify it 2x, which is enough tokens for e1 (bin widths)
        # plus two disjoint groups of vit_tokens each (foreground,
        # background) without needing thousands of positions.
        self.patch = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
        needed = 1 + 2 * vit_tokens
        self.pos = nn.Parameter(torch.zeros(1, max(needed, 8), in_channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=vit_heads,
            dim_feedforward=in_channels * 2, batch_first=True)
        self.vit = nn.TransformerEncoder(enc_layer, num_layers=vit_layers)

        # Bin-width predictor: e1 -> fc -> softmax (Eqn. 3).
        self.bin_width_fc = nn.Linear(in_channels, self.n_bins)

        # Per-pixel bin-probability convs, one for foreground and one for
        # background (Eqn. 7). Applied to L x G (cross-product, Eqn. 5/7),
        # so their input channels match the local branch's.
        self.prob_fg = nn.Conv2d(in_channels, self.n_bins, kernel_size=1)
        self.prob_bg = nn.Conv2d(in_channels, self.n_bins, kernel_size=1)

        # Head-tail gate: a binary foreground/background classifier read
        # off the foreground range-attention map (Eqn. 8). One channel,
        # sigmoid outside this module (BCE-with-logits wants raw logits).
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1)

    def _tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """Patchify the feature map into a short token sequence for the ViT branch."""
        p = self.patch(feat)                          # (B, C, H/2, W/2)
        b, c, h, w = p.shape
        tok = p.flatten(2).transpose(1, 2)             # (B, H/2*W/2, C)
        n = tok.shape[1]
        need = self.pos.shape[1]
        if n >= need:
            tok = tok[:, :need, :]
        else:
            # Pad with zeros if the feature map is too small to have this
            # many spatial positions (only happens on tiny smoke-test crops).
            pad = tok.new_zeros(b, need - n, c)
            tok = torch.cat([tok, pad], dim=1)
        return self.vit(tok + self.pos)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """feat: (B, C, H, W) DPT neck output, same spatial size the final
        depth head would have upsampled from. Returns bin edges, centres,
        combined probabilities, and the raw gate logit (for its own loss).
        """
        b = feat.shape[0]
        local = self.local_conv(feat)                              # Eqn. 1
        tok = self._tokens(feat)                                   # Eqn. 2

        e_bin = tok[:, 0, :]                                        # e1
        e_fg = tok[:, 1:1 + self.vit_tokens, :].mean(dim=1)         # pool(e2..e_{m+1})
        e_bg = tok[:, 1 + self.vit_tokens:1 + 2 * self.vit_tokens, :].mean(dim=1)

        # Eqn. 3-4: relative widths, softmax-normalised so they are
        # positive and sum to (h_max - h_min); cumulative sum gives edges.
        rel_w = F.softmax(self.bin_width_fc(e_bin), dim=-1)         # (B, N)
        widths = rel_w * (self.h_max - self.h_min)
        edges = torch.cat(
            [widths.new_full((b, 1), self.h_min),
             self.h_min + torch.cumsum(widths, dim=1)], dim=1)      # (B, N+1)
        centres = 0.5 * (edges[:, :-1] + edges[:, 1:])               # Eqn. 10

        # Eqn. 5/7: cross-product of local features with each global
        # embedding, broadcast spatially, then a 1x1 conv to bin logits.
        gfg = e_fg[:, :, None, None]
        gbg = e_bg[:, :, None, None]
        r_fg = local * gfg
        r_bg = local * gbg
        p_fg = F.softmax(self.prob_fg(r_fg), dim=1)                  # Eqn. 6/7
        p_bg = F.softmax(self.prob_bg(r_bg), dim=1)

        gate_logit = self.gate(r_fg)                                 # Eqn. 8 (pre-sigmoid)
        gate = (torch.sigmoid(gate_logit) > 0.5).to(p_fg.dtype)
        probs = gate * p_fg + (1.0 - gate) * p_bg                    # Eqn. 9

        height = (probs * centres[:, :, None, None]).sum(dim=1, keepdim=True)  # Eqn. 11

        return {"height": height, "probs": probs, "edges": edges,
                "centres": centres, "gate_logit": gate_logit}


# --------------------------------------------------------------------------
# Losses -- Eqns. 12-25.
# --------------------------------------------------------------------------

def chamfer_bin_edge_loss(edges: torch.Tensor, target: torch.Tensor,
                          valid: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
    """Eqn. 13: bidirectional Chamfer distance between bin edges and the
    ground-truth height values actually present in the tile, so the edges
    settle near the quantiles of the real distribution instead of a fixed
    grid that may put most bins where GAMUS has no data.
    """
    total = edges.new_zeros(())
    n = 0
    for i in range(edges.shape[0]):
        m = valid[i] > 0
        t = target[i][m]
        if t.numel() == 0:
            continue
        if t.numel() > max_points:
            idx = torch.randint(0, t.numel(), (max_points,), device=t.device)
            t = t[idx]
        d = (t[:, None] - edges[i][None, :]).abs()
        total = total + d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
        n += 1
    return total / max(n, 1)


def head_tail_cut_loss(gate_logit: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Eqn. 14: cross-entropy on the foreground/background gate against the
    actual threshold (height > 1 m), so the gate learns the split directly
    rather than only through the downstream height loss.
    """
    tgt = (target > threshold).float().unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(gate_logit, tgt, reduction="none")
    v = valid.unsqueeze(1).float()
    return (loss * v).sum() / v.sum().clamp_min(1.0)


def distribution_constraint_loss(probs: torch.Tensor, edges: torch.Tensor,
                                 target: torch.Tensor, valid: torch.Tensor,
                                 fg_threshold: float = 1.0,
                                 n_points: int = 4096,
                                 p_clamp: float = 1e-3) -> torch.Tensor:
    """Eqns. 15-25: pull the predicted bin distribution toward a reference
    distribution whose scale is solved analytically from the model's own
    confidence, rather than assumed as a fixed hyperparameter.

    Foreground pixels (height > threshold) get a Gaussian reference
    (Eqn. 15-20): assume the truth sits at the centre of whichever bin the
    model gave it, solve sigma from that bin's probability via the inverse
    error function (Eqn. 20), then integrate that Gaussian over every bin
    (Eqn. 21) to get the reference distribution the KL term (Eqn. 22)
    supervises against.

    Background pixels get a uniform reference of the same implied width
    (Eqn. 23-25) instead of a Gaussian, because ground height is close to
    constant and a sharp Gaussian would invent confidence that is not
    there.
    """
    n_bins = probs.shape[1]
    root2 = math.sqrt(2.0)
    total = probs.new_zeros(())
    n = 0

    for i in range(probs.shape[0]):
        idx_all = (valid[i] > 0).flatten().nonzero(as_tuple=False).squeeze(1)
        if idx_all.numel() == 0:
            continue
        if idx_all.numel() > n_points:
            sel = torch.randint(0, idx_all.numel(), (n_points,), device=idx_all.device)
            idx_all = idx_all[sel]

        t = target[i].flatten()[idx_all]                        # (K,)
        p = probs[i].flatten(1).transpose(0, 1)[idx_all]        # (K, N)
        e = edges[i]                                             # (N+1,)

        j = (torch.bucketize(t.contiguous(), e.contiguous()) - 1).clamp(0, n_bins - 1)
        bin_w = (e[j + 1] - e[j]).clamp_min(1e-4)
        p_mode = p.gather(1, j[:, None]).squeeze(1).clamp(p_clamp, 1.0 - p_clamp)

        # Eqn. 20: sigma from the mode-bin probability, assuming the truth
        # sits at the bin centre (the paper's tractable simplification).
        sigma = (bin_w / (2.0 * root2 * torch.erfinv(p_mode))).clamp(1e-3, 1e3)
        cdf = 0.5 * (1.0 + torch.erf((e[None, :] - t[:, None]) / (sigma[:, None] * root2)))
        ref_gauss = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)     # Eqn. 21

        # Eqn. 23-25: uniform reference of the width the same mode
        # probability implies, centred on the truth.
        width = (bin_w / p_mode).clamp_min(bin_w)
        lo, hi = t - 0.5 * width, t + 0.5 * width
        a = torch.maximum(e[None, :-1], lo[:, None])
        c = torch.minimum(e[None, 1:], hi[:, None])
        ref_unif = (c - a).clamp_min(0.0) / width[:, None]

        is_fg = (t > fg_threshold)[:, None]
        ref = torch.where(is_fg, ref_gauss, ref_unif)
        ref = ref / ref.sum(dim=1, keepdim=True).clamp_min(1e-8)

        kl = (ref * (ref.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(dim=1)
        total = total + kl.mean()
        n += 1

    return total / max(n, 1)


class HTCDepthModel(nn.Module):
    """AutoModelForDepthEstimation's backbone/neck/head.conv1/conv2, with the
    stock head's final conv3 (32 -> 1 scalar) replaced by an HTCAdaBinsHead.

    conv1/conv2/activation1 are the pretrained DPT head layers up to the
    32-channel feature map conv3 would otherwise collapse to a single
    height value; they are kept and reused (not reinitialised) so the
    HTC head starts from the same learned features train_height.py's plain
    regression head already relies on, rather than from scratch.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
                n_bins: int = 64, h_min: float = -3.0, h_max: float = 120.0):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        base = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.backbone = base.backbone
        self.neck = base.neck
        self.conv1 = base.head.conv1
        self.conv2 = base.head.conv2
        self.activation1 = base.head.activation1
        self.patch_size = int(getattr(base.config, "patch_size", 14))
        self.head_in_index = int(getattr(base.config, "head_in_index", -1))
        feat_channels = self.conv2.out_channels
        self.htc_head = HTCAdaBinsHead(feat_channels, n_bins=n_bins,
                                       h_min=h_min, h_max=h_max)
        del base

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, H, W = pixel_values.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False)
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]

        x = self.conv1(feat)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))
        return self.htc_head(x)


def htc_dc_loss(out: Dict[str, torch.Tensor], target: torch.Tensor,
                valid: torch.Tensor, weights: Optional[Dict[str, float]] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine Eqns. 12-14 + 22 into one objective, weighted per
    train_height.py's convention of returning a stats dict for logging.

    Default weights follow the paper's own ablation (Table VI): the bin
    edge and HTC terms are cheap regularisers and get small coefficients;
    the pixel height loss dominates.
    """
    w = {"height": 1.0, "chamfer": 0.05, "htc": 0.5, "dist": 0.1}
    if weights:
        w.update(weights)

    height, edges, gate_logit, probs = (out["height"].squeeze(1), out["edges"],
                                        out["gate_logit"], out["probs"])

    valid_f = valid.float()
    l1 = ((height - target).abs() * valid_f).sum() / valid_f.sum().clamp_min(1.0)  # Eqn. 12
    chamfer = chamfer_bin_edge_loss(edges, target, valid)
    htc = head_tail_cut_loss(gate_logit, target, valid)
    dist = distribution_constraint_loss(probs, edges, target, valid)

    loss = (w["height"] * l1 + w["chamfer"] * chamfer
            + w["htc"] * htc + w["dist"] * dist)
    stats = {"loss": float(loss.detach()), "l1": float(l1.detach()),
             "chamfer": float(chamfer.detach()), "htc": float(htc.detach()),
             "dist": float(dist.detach())}
    return loss, stats
