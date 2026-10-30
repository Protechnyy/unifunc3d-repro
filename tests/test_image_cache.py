import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from unifunc3d.data import Frame
from unifunc3d.image_cache import CoarseImageCache


class ImageCacheTests(unittest.TestCase):
    def test_exact_pixels_order_rotation_eviction_and_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            frames = []
            for i in range(4):
                path = Path(folder) / f'{i}.jpg'
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (38, 51, 3), dtype=np.uint8)).save(path)
                frames.append(Frame('v', i, path, path, path, np.eye(4), i))
            cache = CoarseImageCache(capacity=2, workers=2)
            try:
                for order in (frames, frames[::-1], [frames[2], frames[2], frames[0]]):
                    for im, frame in zip(cache.get(order, True, 32, 24), order):
                        original = frame.image(True)
                        expected = original.resize((32, 24) if original.width >= original.height else (24, 32))
                        np.testing.assert_array_equal(im, expected)
                        im.paste((255, 0, 0), (0, 0, im.width, im.height))
                    self.assertLessEqual(len(cache.images), 2)
            finally:
                cache.close()
