"""CPU-only coarse image cache; preserve the original decode/rotation/resize."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


class CoarseImageCache:
    def __init__(self, capacity=1024, workers=8):
        self.capacity = capacity
        self.images = OrderedDict()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def clear(self):
        self.images.clear()

    def close(self):
        self.pool.shutdown(wait=True)
        self.clear()

    def get(self, frames, upright, width, height):
        keys = [(str(f.rgb), f.rotation, upright, width, height,
                 f.rgb.stat().st_size, f.rgb.stat().st_mtime_ns) for f in frames]
        missing = {k: f for k, f in zip(keys, frames) if k not in self.images}

        def decode(item):
            key, frame = item
            im = frame.image(upright)
            size = (width, height) if im.width >= im.height else (height, width)
            return key, im.resize(size)

        # Read/resize independent images concurrently; return in original order.
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
