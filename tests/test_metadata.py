import json
from pathlib import Path
import tempfile
import unittest

from unifunc3d.metadata import benchmark_metadata


class MetadataTests(unittest.TestCase):
    def test_pinned_description_exception_preserves_other_checks(self):
        original, _ = benchmark_metadata('val')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'exceptions.json'
            path.write_text(json.dumps({'437294': {'descriptions_sha256': 'a' * 64}}))
            changed, exceptions = benchmark_metadata('val', path)
        self.assertEqual(changed['437294']['descriptions_sha256'], 'a' * 64)
        for key in ('crop_sha256', 'annotations_sha256', 'desc_ids', 'n_points'):
            self.assertEqual(changed['437294'][key], original['437294'][key])
        self.assertEqual(benchmark_metadata('val')[0], original)
        self.assertEqual(set(exceptions), {'437294'})

    def test_rejects_geometry_override_and_malformed_hash(self):
        for fields in ({'crop_sha256': 'a' * 64}, {'descriptions_sha256': '*'}):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'exceptions.json'
                path.write_text(json.dumps({'437294': fields}))
                with self.assertRaises(ValueError):
                    benchmark_metadata('val', path)


if __name__ == '__main__':
    unittest.main()
