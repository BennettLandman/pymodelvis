from __future__ import annotations

from .base import Adapter, _trunk, type_count


class TransformerAdapter(Adapter):
    """ViT / Swin / generic transformer encoders."""

    name = "transformer"

    def match(self, model, trace, graph=None) -> float:
        n = type_count(model, r"(MultiheadAttention|Attention|Transformer|EncoderBlock|SwinTransformerBlock|ViT)")
        return 0.8 if n else 0.0

    def defaults(self, model, trace):
        # transformers carry a few "massive activation" channels that are nearly constant across tokens;
        # a robust spread (p90 - p10 over tokens) picks channels with real spatial structure
        return {"token_mode": "auto", "capture_attention": True, "channel_strategy": "spread"}

    def concepts(self, graph):
        out = {}
        trunk = _trunk(graph)
        tok = [s for s in trunk if s.summary is not None and s.summary.kind in ("tokens", "image2d")]
        if not tok:
            return None
        n = len(tok)
        for i, s in enumerate(tok):
            if i == 0 and s.summary.kind == "image2d" and "Conv" in s.type_name:
                out[s.key] = "PATCH EMBEDDING"
            else:
                f = (i + 0.5) / n
                out[s.key] = "LOCAL MIXING" if f < 0.4 else ("CONTEXT MIXING" if f < 0.75 else "GLOBAL CONTEXT")
        return out
