"""
Tiled inference for WCT2, so large content images can be processed on GPUs
that don't have enough VRAM to run the whole image at once (e.g. GTX 1650,
~4GB, vs. ~5.5GB needed for a single 1024px pass).

Naively splitting content into tiles and running WCT2.transfer() on each tile
independently produces visible seams: WCT's whitening step normalizes each
tile using that tile's OWN local pixel statistics (mean/covariance), so two
neighboring tiles with different local content end up whitened differently,
creating a block-per-tile color bias that a soft blend mask alone can't hide.

The fix used here: compute the whitening statistics ONCE from a small
downsampled proxy of the WHOLE content image (same idea already used for the
style image — its stats are representative even at reduced resolution), then
apply that SAME whitening transform to every tile. This reproduces what a
single full-resolution pass would do (one global content statistic, applied
everywhere), while only the cheap conv encode/decode work is repeated per
tile — which is what actually costs VRAM.
"""
import torch

from wct2.core import svd as _svd, get_squeeze_feat, get_rank


def _tile_starts(length: int, tile: int, overlap: int) -> list:
    """1D tile start offsets covering [0, length) with `tile`-sized windows.
    The last tile is snapped flush to the far edge so the whole length is
    covered even when it doesn't divide evenly by the stride."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _feather_ramp(size: int, feather: int, at_start_edge: bool, at_end_edge: bool,
                   device, dtype) -> torch.Tensor:
    """1D blend weights for one tile along one axis: ramps 0->1 over `feather`
    pixels at a side that has a neighboring tile, stays 1 at a side that is
    the true image boundary (nothing to blend with there)."""
    w = torch.ones(size, device=device, dtype=dtype)
    feather = min(feather, size // 2) if size > 1 else 0
    if feather > 0:
        ramp = torch.linspace(0, 1, feather, device=device, dtype=dtype)
        if not at_start_edge:
            w[:feather] = ramp
        if not at_end_edge:
            w[-feather:] = ramp.flip(0)
    return w


def _content_whitening(feat: torch.Tensor, device):
    """(mean, whitening_matrix, min, max) from a content feature map — the
    same math as wct2.core.wct_core's content-side computation, factored out
    so it can be computed once globally and reused across tiles."""
    f = get_squeeze_feat(feat)
    cmin, cmax = f.min(), f.max()
    mean = torch.mean(f, 1)
    _, e, v = _svd(f, iden=True, device=device)
    k = get_rank(e, f.size(0))
    d = e[:k].pow(-0.5)
    W = v[:, :k] @ torch.diag(d) @ v[:, :k].t()
    return mean, W, cmin, cmax


def _style_coloring(feat: torch.Tensor, device, weight=1):
    """(EDE, mean) from a style feature map — the style-side half of
    wct2.core.wct_core, factored out so it's computed once per level instead
    of once per tile (style doesn't change between tiles)."""
    f = get_squeeze_feat(feat)
    s_mean = torch.mean(f, 1)
    _, e, v = _svd(f, iden=True, device=device)
    k = get_rank(e, f.size(0))
    d = e[:k].pow(0.5)
    EDE = v[:, :k] @ (torch.diag(d) * weight) @ v[:, :k].t()
    return EDE, s_mean


def _make_wct_fn(content_stats, style_stats):
    mean, W, cmin, cmax = content_stats
    EDE, s_mean = style_stats

    def fn(tile_feat: torch.Tensor, alpha: float) -> torch.Tensor:
        f = get_squeeze_feat(tile_feat)
        whitened = W @ (f - mean.unsqueeze(1))
        target = EDE @ whitened + s_mean.unsqueeze(1)
        target = target.clamp(cmin, cmax)
        target = target.view_as(tile_feat)
        return alpha * target + (1 - alpha) * tile_feat

    return fn


@torch.no_grad()
def precompute_wct_params(model, content_feats_global, content_skips_global,
                           style_feats, style_skips, device):
    """Build the per-level/per-skip-component WCT functions that every tile
    will share, from a global (whole-image, possibly downsampled) content
    pass and the already-computed style pass. Mirrors WCT2.transfer_at."""
    params = {'encoder': {}, 'skip': {}, 'decoder': {}}

    if 'encoder' in model.transfer_at:
        for level, style_feat in style_feats['encoder'].items():
            c_stats = _content_whitening(content_feats_global['encoder'][level], device)
            s_stats = _style_coloring(style_feat, device)
            params['encoder'][level] = _make_wct_fn(c_stats, s_stats)

    if 'skip' in model.transfer_at:
        for skip_level in ('pool1', 'pool2', 'pool3'):
            params['skip'][skip_level] = {}
            for component in (0, 1, 2):
                c_stats = _content_whitening(content_skips_global[skip_level][component], device)
                s_stats = _style_coloring(style_skips[skip_level][component], device)
                params['skip'][skip_level][component] = _make_wct_fn(c_stats, s_stats)

    if 'decoder' in model.transfer_at:
        for level, style_feat in style_feats['decoder'].items():
            c_stats = _content_whitening(content_feats_global['decoder'][level], device)
            s_stats = _style_coloring(style_feat, device)
            params['decoder'][level] = _make_wct_fn(c_stats, s_stats)

    return params


@torch.no_grad()
def tiled_transfer(model, content: torch.Tensor, content_proxy: torch.Tensor,
                    style_feats, style_skips, alpha: float = 1.0,
                    tile_size: int = 384, overlap: int = 48) -> torch.Tensor:
    """content: (1, 3, H, W) tensor on the model's device, full working
    resolution — this is what gets tiled.
    content_proxy: (1, 3, h, w) tensor, a smaller/downsampled version of the
    SAME image, used only to compute global whitening statistics once.
    style_feats/style_skips: from model.get_all_feature(style).
    Returns a (1, 3, H, W) tensor — same shape as content.
    """
    device = model.device
    _, _, H, W = content.shape

    if H <= tile_size and W <= tile_size:
        # small enough to process directly — one tile, no seams possible,
        # so there's no need for the global-stats machinery at all.
        return model.transfer_with_style(content, style_feats, style_skips, alpha=alpha)

    content_feats_global, content_skips_global = model.get_all_feature(content_proxy)
    wct_params = precompute_wct_params(model, content_feats_global, content_skips_global,
                                        style_feats, style_skips, device)
    del content_feats_global, content_skips_global
    torch.cuda.empty_cache()

    ys = _tile_starts(H, tile_size, overlap)
    xs = _tile_starts(W, tile_size, overlap)
    feather = overlap // 2

    out_accum = torch.zeros_like(content)
    weight_accum = torch.zeros(1, 1, H, W, device=device, dtype=content.dtype)

    for y0 in ys:
        th = min(tile_size, H - y0)
        for x0 in xs:
            tw = min(tile_size, W - x0)
            tile = content[:, :, y0:y0 + th, x0:x0 + tw]
            result = model.transfer_tile(tile, wct_params, alpha=alpha)
            if result.shape[-2:] != (th, tw):
                result = torch.nn.functional.interpolate(
                    result, size=(th, tw), mode="bilinear", align_corners=False)

            wy = _feather_ramp(th, feather, at_start_edge=(y0 == 0), at_end_edge=(y0 + th == H),
                                device=device, dtype=content.dtype)
            wx = _feather_ramp(tw, feather, at_start_edge=(x0 == 0), at_end_edge=(x0 + tw == W),
                                device=device, dtype=content.dtype)
            mask = (wy.view(-1, 1) * wx.view(1, -1)).view(1, 1, th, tw)

            out_accum[:, :, y0:y0 + th, x0:x0 + tw] += result * mask
            weight_accum[:, :, y0:y0 + th, x0:x0 + tw] += mask

            del tile, result
            torch.cuda.empty_cache()

    return out_accum / weight_accum
