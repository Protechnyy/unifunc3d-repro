"""CPU-only image cache; preserve the original decode/rotation."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


class ImageCache:
    def __init__(self, capacity=1024, workers=8):
        self.capacity = capacity
        self.images = OrderedDict()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def clear(self):
        self.images.clear()

    def close(self):
        self.pool.shutdown(wait=True)
        self.clear()

    def get(self, frames, upright):
        keys = [(str(f.rgb), f.rotation, upright,
                 f.rgb.stat().st_size, f.rgb.stat().st_mtime_ns) for f in frames]
        missing = {k: f for k, f in zip(keys, frames) if k not in self.images}

        def decode(item):
            key, frame = item
            return key, frame.image(upright)

        # Read independent images concurrently; return in original order.
        loaded = dict(self.pool.map(decode, missing.items()))
        result = []
        for key in keys:
            im = loaded[key] if key in loaded else self.images[key]
            self.images[key] = im
            self.images.move_to_end(key)
            result.append(im.copy())  # Callers cannot mutate cached pixels.
        while len(self.images) > self.capacity:
            self.images.popitem(last=False)
        return result
