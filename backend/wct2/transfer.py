"""
Adapted from clovaai/WCT2 (transfer.py), MIT licensed.
Original: https://github.com/clovaai/WCT2
Stripped down to the reusable WCT2 class (no CLI / argparse / bulk runner —
those live in the original repo if you ever need them).
"""
import os

import torch

from .model import WaveEncoder, WaveDecoder
from .core import feature_wct


class WCT2:
    def __init__(self, model_path=None, transfer_at=('encoder', 'skip', 'decoder'),
                 option_unpool='cat5', device='cuda:0', verbose=False):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model_checkpoints')

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - {'encoder', 'decoder', 'skip'}), \
            'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(
            os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)),
            map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(
            os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)),
            map_location=lambda storage, loc: storage))
        self.encoder.eval()
        self.decoder.eval()

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    @torch.no_grad()
    def transfer(self, content, style, alpha=1.0):
        """content, style: normalized image tensors, shape (1, 3, H, W), values in [0, 1].
        No segmentation maps — global photorealistic style transfer."""
        style_feats, style_skips = self.get_all_feature(style)
        return self.transfer_with_style(content, style_feats, style_skips, alpha=alpha)

    @torch.no_grad()
    def transfer_with_style(self, content, style_feats, style_skips, alpha=1.0):
        """Same as transfer(), but takes pre-computed style_feats/style_skips
        (from get_all_feature) instead of a style image. Lets callers encode a
        style/reference image once and reuse it across many content tiles —
        needed for tiled processing of large content images without re-running
        the (relatively expensive) style encode/decode pass per tile."""
        content_segment, style_segment, label_set, label_indicator = None, None, None, None
        content_feat, content_skips = content, {}

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(
                        content_skips[skip_level][component], style_skips[skip_level][component],
                        content_segment, style_segment,
                        label_set, label_indicator,
                        alpha=alpha, device=self.device)

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat

    @torch.no_grad()
    def transfer_tile(self, content, wct_params, alpha=1.0):
        """Like transfer_with_style, but instead of deriving whitening/coloring
        stats from this call's own content+style feature maps, applies
        precomputed per-level functions from tiling.precompute_wct_params.
        Used to process one content tile with whitening statistics shared
        across every tile of the same image (see tiling.py for why that
        matters — per-tile-local statistics cause visible tile seams)."""
        content_feat, content_skips = content, {}

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct_params['encoder']:
                content_feat = wct_params['encoder'][level](content_feat, alpha)

        if 'skip' in self.transfer_at:
            for skip_level in ('pool1', 'pool2', 'pool3'):
                for component in (0, 1, 2):
                    content_skips[skip_level][component] = wct_params['skip'][skip_level][component](
                        content_skips[skip_level][component], alpha)

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in wct_params['decoder']:
                content_feat = wct_params['decoder'][level](content_feat, alpha)
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat
