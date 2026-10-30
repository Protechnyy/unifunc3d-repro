"""Geometry/index alignment and temporal/protocol regression tests, no GPU needed."""
import tempfile
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from unifunc3d.data import Frame
from unifunc3d.geometry import agreement, lift_frame, project_mask
from unifunc3d.grounding import (coarse_indices, parse_grounding, parse_verification,
                                window_indices)


class GroundingTests(unittest.TestCase):
    def test_four_offsets_cover_timeline(self):
        samples = [coarse_indices(1024, 64, 4, k) for k in range(4)]
        self.assertTrue(all(len(x) == 64 for x in samples))
        self.assertEqual(len(set(sum(samples, []))), 256)
        self.assertEqual(samples[-1][-1], 1023)
        self.assertTrue(all(0 <= i < 3 for i in coarse_indices(3)))

    def test_windows_use_timestamps_not_frame_numbers(self):
        self.assertEqual(window_indices([0, .01, .02, .5, 1], 2, 3), [0, 1, 2])
        self.assertEqual(window_indices([0, 1, 2], 0, 3), [0])

    def test_strict_coordinates_and_frame_mapping(self):
        text = '<affordance> functional object: drawer handle; <frame 17>: clear; (650, 501) </affordance>'
        parsed = parse_grounding(text, 64)
        self.assertEqual(parsed['functional_object'], 'drawer handle')
        self.assertEqual(parsed['frame'], 17)
        self.assertEqual(parsed['points'], [(650, 501)])
        self.assertIsNone(parse_grounding(text, 8)['frame'])
        self.assertEqual(parse_grounding('<affordance> (-1, 1), (1001, 5) </affordance>')['points'], [])
        self.assertEqual(parse_grounding('<affordance> NONE </affordance>')['points'], [])
        self.assertTrue(parse_verification('YES. It is the handle.'))
        self.assertFalse(parse_verification('NO. It includes the door.'))
        self.assertFalse(parse_verification('It might be YES or NO.'))


class GeometryTests(unittest.TestCase):
    def test_occlusion_out_of_view_and_behind_camera(self):
        xyz = np.array([[0, 0, 2], [0, 0, 4], [0, 0, -2], [9, 0, 2], [2, 0, 2]], float)
        intrinsic = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        depth = np.full((3, 4), 2.)
        mask = np.zeros_like(depth, dtype=bool)
        mask[1, 1] = True
        np.testing.assert_array_equal(project_mask(xyz, mask, depth, intrinsic, np.eye(4), .25, 2), [0])
        depth[1, 1] = 0
        self.assertEqual(len(project_mask(xyz, mask, depth, intrinsic, np.eye(4))), 0)

    def test_pose_translation_and_all_image_rotations(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            depth = np.full((3, 4), 2000, dtype=np.uint16)
            Image.fromarray(depth).save(folder / 'depth.png')
            (folder / 'intrinsics.pincam').write_text('4 3 2 2 1 1')
            original = np.zeros((3, 4), bool)
            original[1, 2] = True
            pose = np.eye(4)
            pose[0, 3] = -10
            xyz = np.array([[11, 0, 2], [10, 0, 2], [0, 0, 2]], float)
            for rotation in range(4):
                frame = Frame('v', 1., folder / 'rgb.jpg', folder / 'depth.png',
                              folder / 'intrinsics.pincam', pose, rotation)
                upright = np.rot90(original, -rotation)
                np.testing.assert_array_equal(lift_frame(xyz, upright, frame), [0])

    def test_agreement_strict_threshold_and_empty(self):
        np.testing.assert_array_equal(agreement([0, 7, 8, 10], .7), [2, 3])
        self.assertEqual(len(agreement([0, 0, 0])), 0)
        self.assertEqual(len(agreement([1, 1, 1])), 0)
        self.assertEqual(len(agreement([])), 0)


class PipelineTests(unittest.TestCase):
    def test_cached_end_to_end_without_annotations(self):
        from unifunc3d.config import Config
        from unifunc3d.pipeline import Pipeline

        class FakeQwen:
            calls = 0

            def ask(self, images, prompt, **kwargs):
                self.calls += 1
                if prompt.startswith('Given'):
                    text = '<affordance> functional object: button; <frame 0>: clear; (500, 500) </affordance>'
                elif prompt.startswith('Identify'):
                    text = '<affordance> (500, 500) </affordance>'
                else:
                    text = 'YES'
                return {'text': text}

        class FakeSAM:
            def segment(self, image, points):
                mask = np.zeros((3, 4), bool)
                mask[1, 1] = True
                return [{'mask': mask, 'score': .9, 'point': [500, 500]}]

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'K.pincam').write_text('4 3 2 2 1 1')
            Image.fromarray(np.full((3, 4), 2000, np.uint16)).save(folder / 'depth.png')
            frames = []
            for i in range(4):
                # Real frame keys contain decimal timestamps within one second.
                rgb = folder / f'video_10.{i:03d}.jpg'
                Image.new('RGB', (4, 3)).save(rgb)
                frames.append(Frame('video', float(i), rgb, folder / 'depth.png',
                                    folder / 'K.pincam', np.eye(4), 0))
            scene = SimpleNamespace(visit_id='test', videos={'video': frames},
                                    xyz=np.array([[0, 0, 2], [1, 0, 2], [0, 0, 4]], np.float32))
            desc = {'desc_id': 'query', 'description': 'Press the button'}
            pipeline = Pipeline(Config(coarse_frames=2, iterations=2), folder / 'output')
            self.addCleanup(pipeline.close)
            pipeline.qwen = FakeQwen()
            pipeline.sam = FakeSAM()
            first = pipeline.query(scene, desc)
            count = pipeline.qwen.calls
            second = pipeline.query(scene, desc)
            self.assertEqual(first, {'desc_id': 'query', 'pred_mask': [0]})
            self.assertEqual(first, second)
            self.assertEqual(pipeline.qwen.calls, count)
            # Both overlapping windows cover the same candidate frame. Each view
            # contributes only once, even though it appears in both windows.
            votes = np.load(folder / 'output/traces/test/query/votes.npz')
            self.assertEqual(int(votes['votes'][0]), 4)
            self.assertEqual(count, 10)  # 2 coarse + 4 fine + 4 verification


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional inference dependency: torch')
class PatchProjectionTests(unittest.TestCase):
    def test_linear_projection_preserves_conv_weights_and_outputs(self):
        import torch
        from unifunc3d.torch_utils import LinearPatchEmbedding
        conv = torch.nn.Conv3d(3, 7, (2, 4, 4), stride=(2, 4, 4)).double()
        patch_embed = SimpleNamespace(proj=conv)
        replacement = LinearPatchEmbedding(patch_embed)
        x = torch.randn(9, 96, dtype=torch.float64)
        expected = conv(x.reshape(9, 3, 2, 4, 4)).reshape(9, 7)
        torch.testing.assert_close(replacement(x), expected, rtol=1e-12, atol=1e-12)
        self.assertIs(replacement.proj.weight, conv.weight)
        self.assertIs(replacement.proj.bias, conv.bias)

    def test_overlapping_conv_is_not_silently_rewritten(self):
        import torch
        from unifunc3d.torch_utils import LinearPatchEmbedding
        with self.assertRaises(ValueError):
            LinearPatchEmbedding(SimpleNamespace(proj=torch.nn.Conv3d(3, 7, 3, stride=1)))


if __name__ == '__main__':
    unittest.main()
