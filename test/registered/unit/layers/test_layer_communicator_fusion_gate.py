import types
import unittest
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.flashinfer_comm_fusion import is_hybrid_moe_ep_tp
from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    """Minimal stand-in for the attributes should_fuse_mlp_allreduce_with_next_layer reads."""
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestPostExpertsSkipPolarity(unittest.TestCase):
    """Exactly one of the two post-experts reductions may be fused away.

    The next layer's fused residual+LN reduces over a single group, so with both
    moe_ep_size > 1 and moe_tp_size > 1 it can absorb only the MoE-TP reduction
    (the group its workspace rendezvouses on). Skipping the EP one as well
    reduces over half the peers and silently corrupts activations -- observed as
    garbage completions on Qwen3-30B-A3B with --tp-size 4 --ep-size 2.

    Getting the polarity backwards drops the *other* reduction and reproduces
    the same symptom, so both directions are pinned here.
    """

    def _skip(self, *, is_tp_path, moe_ep_size, moe_tp_size, moe_dp_size=1):
        with get_forward().scoped(
            fuse_mlp_allreduce=True, mlp_reduce_scatter=False
        ), get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            moe_dp_size=moe_dp_size,
        ):
            return should_skip_post_experts_all_reduce(is_tp_path=is_tp_path)

    def test_hybrid_skips_tp_and_keeps_ep(self):
        self.assertTrue(self._skip(is_tp_path=True, moe_ep_size=2, moe_tp_size=2))
        self.assertFalse(self._skip(is_tp_path=False, moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_skips_both(self):
        # moe_ep_size == 1: there is no EP reduction to keep.
        self.assertTrue(self._skip(is_tp_path=True, moe_ep_size=1, moe_tp_size=4))
        self.assertTrue(self._skip(is_tp_path=False, moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_skips_both(self):
        # moe_tp_size == 1: the EP reduction is the one that gets fused.
        self.assertTrue(self._skip(is_tp_path=True, moe_ep_size=4, moe_tp_size=1))
        self.assertTrue(self._skip(is_tp_path=False, moe_ep_size=4, moe_tp_size=1))

    def test_moe_dp_does_not_change_the_split(self):
        # moe_dp_size changes how many peers the reductions cover in total, but
        # there are still two of them and the deferred one is still the TP one.
        self.assertTrue(
            self._skip(is_tp_path=True, moe_ep_size=2, moe_tp_size=2, moe_dp_size=2)
        )
        self.assertFalse(
            self._skip(is_tp_path=False, moe_ep_size=2, moe_tp_size=2, moe_dp_size=2)
        )

    def test_reduce_scatter_still_skips_both(self):
        # Reduce-scatter replaces the reduction outright, so the hybrid split
        # must not leak into that path.
        with get_forward().scoped(
            fuse_mlp_allreduce=False, mlp_reduce_scatter=True
        ), get_parallel().override(moe_ep_size=2, moe_tp_size=2, moe_dp_size=1):
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=True))
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=False))


class TestResolveFusionGroup(unittest.TestCase):
    """The workspace rendezvous, the fused kernel's world size, and (in the
    allreduce-only path) the group tagging all read this one function. They used
    to duplicate the rule, and a disagreement makes the kernel reduce across the
    wrong peers with no error -- so the mapping is pinned here.
    """

    def _resolve(self, *, use_attn_tp_group, moe_ep_size=1, moe_tp_size=1):
        with patch.object(
            fusion, "get_attn_tp_group", return_value="ATTN_TP"
        ), patch.object(
            fusion, "get_moe_tp_group", return_value="MOE_TP"
        ), patch.object(
            fusion, "get_moe_ep_group", return_value="MOE_EP"
        ), get_parallel().override(
            moe_ep_size=moe_ep_size,
            moe_ep_rank=0,
            moe_tp_size=moe_tp_size,
            moe_tp_rank=0,
            attn_tp_size=4,
            attn_tp_rank=0,
        ):
            world_size, _, group = fusion.resolve_fusion_group(
                use_attn_tp_group=use_attn_tp_group
            )
            return world_size, group

    def test_moe_group_is_the_one_the_deferred_reduction_uses(self):
        # Hybrid: the TP reduction is the deferred one, so the workspace must
        # sit on the MoE-TP group -- not the EP group, which stays inline.
        self.assertEqual(
            self._resolve(use_attn_tp_group=False, moe_ep_size=2, moe_tp_size=2),
            (2, "MOE_TP"),
        )
        # Pure EP: no TP reduction exists, so the EP one is deferred.
        self.assertEqual(
            self._resolve(use_attn_tp_group=False, moe_ep_size=4, moe_tp_size=1),
            (4, "MOE_EP"),
        )
        # Pure TP: unchanged.
        self.assertEqual(
            self._resolve(use_attn_tp_group=False, moe_ep_size=1, moe_tp_size=4),
            (4, "MOE_TP"),
        )

    def test_attn_group_is_independent_of_moe_topology(self):
        self.assertEqual(
            self._resolve(use_attn_tp_group=True, moe_ep_size=2, moe_tp_size=2),
            (4, "ATTN_TP"),
        )


class TestHybridMoeEpTpPredicate(unittest.TestCase):
    def test_requires_both_dimensions(self):
        for ep, tp, expected in ((2, 2, True), (1, 4, False), (4, 1, False)):
            with self.subTest(ep=ep, tp=tp):
                with get_parallel().override(moe_ep_size=ep, moe_tp_size=tp):
                    self.assertEqual(is_hybrid_moe_ep_tp(), expected)


class TestFuseMlpAllReduceGate(unittest.TestCase):
    def _should_fuse(self, *, moe_ep_size, moe_tp_size):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_fuses(self):
        # Hybrid is no longer gated off here: correctness now comes from
        # should_skip_post_experts_all_reduce() keeping the EP reduction inline.
        self.assertTrue(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


if __name__ == "__main__":
    unittest.main()
