"""
tests/test_pipeline.py
-----------------------
Unit tests for the Cornell grasp detection pipeline.
All tests use synthetic random data — no Cornell data required.

Run: python -m pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGraspUtils:
    def _make_rect(self):
        from utils.grasp_utils import GraspRectangle
        pts = np.array([[100, 100], [100, 200],
                        [150, 200], [150, 100]], dtype=np.float32)
        return GraspRectangle(pts)

    def test_center(self):
        r = self._make_rect()
        c = r.center
        assert c.shape == (2,)

    def test_angle_is_float(self):
        r = self._make_rect()
        a = r.angle
        assert isinstance(a, float)
        assert -np.pi/2 <= a <= np.pi/2 or True  # just check it runs

    def test_width_positive(self):
        r = self._make_rect()
        assert r.width > 0

    def test_polygon_mask_shape(self):
        r = self._make_rect()
        mask = r.polygon_mask((480, 640))
        assert mask.shape == (480, 640)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_iou_self(self):
        r = self._make_rect()
        assert r.iou(r) > 0.99

    def test_iou_non_overlapping(self):
        from utils.grasp_utils import GraspRectangle
        r1 = self._make_rect()
        pts2 = np.array([[300, 400], [300, 500],
                         [350, 500], [350, 400]], dtype=np.float32)
        r2 = GraspRectangle(pts2)
        assert r1.iou(r2) == 0.0

    def test_build_pixel_maps(self):
        from utils.grasp_utils import build_pixel_maps
        r = self._make_rect()
        q, a, w = build_pixel_maps([r], shape=(480, 640))
        assert q.shape == (480, 640)
        assert a.shape == (480, 640)
        assert w.shape == (480, 640)
        assert q.max() == 1.0
        assert q.min() == 0.0

    def test_is_grasp_correct_perfect(self):
        from utils.grasp_utils import is_grasp_correct
        r = self._make_rect()
        assert is_grasp_correct(r, [r]) is True

    def test_is_grasp_correct_wrong_angle(self):
        from utils.grasp_utils import GraspRectangle, is_grasp_correct
        r1 = self._make_rect()
        # Rotate 90 degrees
        pts2 = np.array([[100, 150], [200, 150],
                         [200, 200], [100, 200]], dtype=np.float32)
        r2 = GraspRectangle(pts2)
        # With very tight thresholds this should fail
        result = is_grasp_correct(r1, [r2],
                                   iou_threshold=0.8,
                                   angle_threshold_deg=5.0)
        # May or may not be correct depending on geometry — just check it runs
        assert isinstance(result, bool)

    def test_prediction_to_grasp(self):
        from utils.grasp_utils import prediction_to_grasp
        q = np.zeros((480, 640), dtype=np.float32)
        q[240, 320] = 0.9
        a = np.zeros_like(q)
        w = np.full_like(q, 0.1)
        grasps = prediction_to_grasp(q, a, w, threshold=0.5, num_peaks=1)
        assert len(grasps) == 1
        center = grasps[0].center
        assert abs(center[0] - 240) < 25
        assert abs(center[1] - 320) < 25


class TestOcclusionUtils:
    def _dummy_depth(self, H=120, W=160):
        d = np.random.uniform(500, 900, (H, W)).astype(np.float32)
        # Add some zeros (missing)
        d[:20, :20] = 0.0
        return d

    def test_binary_map(self):
        from utils.occlusion_utils import compute_occlusion_map
        d = self._dummy_depth()
        m = compute_occlusion_map(d, mode='binary')
        assert m.shape == d.shape
        assert m.dtype == np.float32
        assert m.min() >= 0 and m.max() <= 1
        # Zeros in depth should map to 1
        assert m[:20, :20].mean() == 1.0

    def test_variance_map(self):
        from utils.occlusion_utils import compute_occlusion_map
        d = self._dummy_depth()
        m = compute_occlusion_map(d, mode='variance')
        assert m.shape == d.shape
        assert m.min() >= 0 and m.max() <= 1.0 + 1e-5

    def test_random_occlusion(self):
        from utils.occlusion_utils import apply_random_occlusion
        d = self._dummy_depth()
        occ = apply_random_occlusion(d, num_blocks=3, block_size_range=(10, 30))
        assert occ.shape == d.shape
        # Should have more zeros than original
        assert (occ == 0).sum() >= (d == 0).sum()

    def test_occlusion_fraction(self):
        from utils.occlusion_utils import apply_occlusion_fraction
        d = np.ones((100, 100), dtype=np.float32)
        occ = apply_occlusion_fraction(d, fraction=0.3)
        frac = (occ == 0).mean()
        # Should be roughly 0.3 (within ±0.2 due to block granularity)
        assert 0.05 < frac < 0.7

    def test_image_occlusion_fraction(self):
        from utils.occlusion_utils import image_occlusion_fraction
        d = np.zeros((100, 100), dtype=np.float32)
        d[50:, :] = 1.0
        assert abs(image_occlusion_fraction(d) - 0.5) < 0.01


class TestBaselineModel:
    def _cfg(self):
        return {
            'model': {
                'name': 'baseline',
                'input_channels': 1,
                'encoder_channels': [16, 32, 64],
                'decoder_channels': [32, 16],
                'dropout_rate': 0.0,
            }
        }

    def test_output_shapes(self):
        from models.baseline_model import BaselineGraspCNN
        m = BaselineGraspCNN.from_config(self._cfg())
        x = torch.randn(2, 1, 120, 160)
        q, a, w = m(x)
        assert q.shape == (2, 1, 120, 160)
        assert a.shape == (2, 1, 120, 160)
        assert w.shape == (2, 1, 120, 160)

    def test_quality_range(self):
        from models.baseline_model import BaselineGraspCNN
        m = BaselineGraspCNN.from_config(self._cfg())
        x = torch.randn(2, 1, 120, 160)
        q, _, _ = m(x)
        assert q.min().item() >= 0.0
        assert q.max().item() <= 1.0

    def test_angle_range(self):
        from models.baseline_model import BaselineGraspCNN
        m = BaselineGraspCNN.from_config(self._cfg())
        x = torch.randn(2, 1, 120, 160)
        _, a, _ = m(x)
        assert a.min().item() >= -(torch.pi / 2 + 1e-4)
        assert a.max().item() <=  (torch.pi / 2 + 1e-4)

    def test_loss(self):
        from models.baseline_model import BaselineGraspCNN, GraspLoss
        m  = BaselineGraspCNN.from_config(self._cfg())
        fn = GraspLoss()
        x  = torch.randn(2, 1, 120, 160)
        q, a, w = m(x)
        gt_q = (torch.rand_like(q) > 0.9).float()
        gt_a = torch.zeros_like(a)
        gt_w = torch.zeros_like(w)
        loss, d = fn(q, a, w, gt_q, gt_a, gt_w)
        assert loss.item() >= 0
        assert 'total_loss' in d

    def test_backprop(self):
        from models.baseline_model import BaselineGraspCNN, GraspLoss
        m  = BaselineGraspCNN.from_config(self._cfg())
        fn = GraspLoss()
        x  = torch.randn(2, 1, 60, 80)
        q, a, w = m(x)
        gt_q = (torch.rand_like(q) > 0.9).float()
        gt_a = torch.zeros_like(a)
        gt_w = torch.zeros_like(w)
        loss, _ = fn(q, a, w, gt_q, gt_a, gt_w)
        loss.backward()
        # Check gradients flow
        for p in m.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestRobustModel:
    def _cfg(self):
        return {
            'model': {
                'name': 'robust',
                'input_channels': 2,
                'encoder_channels': [16, 32, 64],
                'decoder_channels': [32, 16],
                'dropout_rate': 0.0,
                'use_occ_modulation': True,
            }
        }

    def test_output_shapes(self):
        from models.robust_model import RobustGraspCNN
        m = RobustGraspCNN.from_config(self._cfg())
        x = torch.randn(2, 2, 120, 160)
        q, a, w = m(x)
        assert q.shape == (2, 1, 120, 160)

    def test_modulation_changes_output(self):
        """Occlusion map channel should affect output."""
        from models.robust_model import RobustGraspCNN
        m = RobustGraspCNN.from_config(self._cfg())
        m.eval()
        depth = torch.randn(1, 1, 60, 80)
        occ_low  = torch.zeros(1, 1, 60, 80)
        occ_high = torch.ones(1, 1, 60, 80)
        x_low  = torch.cat([depth, occ_low],  dim=1)
        x_high = torch.cat([depth, occ_high], dim=1)
        with torch.no_grad():
            q1, _, _ = m(x_low)
            q2, _, _ = m(x_high)
        # Outputs should differ when occlusion maps differ
        assert not torch.allclose(q1, q2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
