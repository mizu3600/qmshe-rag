from collections.abc import Sequence

import numpy as np

from qmshe.providers import DeterministicEmbedder, ProviderError, SiliconFlowClient


class TextEncoder:
    def __init__(self, allow_fallback: bool = True, fallback_dimension: int = 128):
        self.allow_fallback = allow_fallback
        self.fallback = DeterministicEmbedder(fallback_dimension)
        try:
            self.remote = SiliconFlowClient()
        except ProviderError:
            self.remote = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.remote is not None:
            try:
                return self.remote.embed(texts)
            except ProviderError:
                if not self.allow_fallback:
                    raise
        if not self.allow_fallback:
            raise ProviderError("remote embedding is unavailable and fallback is disabled")
        return self.fallback.embed(texts)
