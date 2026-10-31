"""Bounded CPU cache of identical, complete Qwen visual encoder inputs.

Cache whole calls, never regroup images: batch-dependent BF16 rounding must
remain identical to the original single-request execution.
"""
import hashlib
from collections import OrderedDict


def image_key(images, max_pixels):
    digest = hashlib.sha256(str(max_pixels).encode())
    for image in images:
        digest.update(repr((image.mode, image.size)).encode())
        digest.update(image.tobytes())
    return digest.digest()


def map_tensors(value, function):
    if isinstance(value, tuple):
        return tuple(map_tensors(x, function) for x in value)
    if isinstance(value, list):
        return [map_tensors(x, function) for x in value]
    return function(value)


class VisualFeatureCache:
    def __init__(self, encoder, max_bytes=8 * 1024**3):
        self.encoder = encoder
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.key = None
        self.hits = self.misses = 0
        self.enabled = True

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def __call__(self, pixel_values, image_grid_thw=None):
        if not self.enabled or self.key is None:
            return self.encoder(pixel_values, image_grid_thw)
        key = self.key
        if key in self.entries:
            self.hits += 1
            value, size = self.entries.pop(key)
            self.entries[key] = value, size
            return map_tensors(value, lambda x: x.to(pixel_values.device))
        self.misses += 1
        result = self.encoder(pixel_values, image_grid_thw)
        size = 0

        def store(tensor):
            nonlocal size
            size += tensor.numel() * tensor.element_size()
            return tensor.detach().to("cpu", copy=True)

        value = map_tensors(result, store)
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = value, size
            self.bytes += size
        return result
