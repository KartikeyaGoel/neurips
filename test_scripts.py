"""
Preflight checks for the HalfCheetah curriculum pipeline.

These tests are small by design. They check run wiring and experimental
invariants before launching the expensive multi-seed experiment.
"""

import os
import tempfile

import curriculum_pipeline as cp


def _within_bounds(params, bounds):
    return all(bounds[k][0] <= params[k] <= bounds[k][1] for k in bounds)


def test_1_fixed_probe_and_test_sets():
    print("\nTest 1: fixed probe and held-out test sets...")

    probes_a = cp.fixed_probe_params()
    probes_b = cp.fixed_probe_params()
    assert probes_a == probes_b, "Probe params must be deterministic"
    assert len(probes_a) == 50, "Expected 50 fixed probe environments"
    assert _within_bounds(probes_a[0], cp.DEFAULT_BOUNDS)

    test_a = cp.fixed_test_params(seed=0)
    test_b = cp.fixed_test_params(seed=0)
    test_c = cp.fixed_test_params(seed=1)
    assert test_a == test_b, "Held-out test params must be deterministic"
    assert test_a != test_c, "Different seeds should get different test sets"
    assert len(test_a) == 20, "Expected 20 held-out test environments"
    assert _within_bounds(test_a[0], cp.DEFAULT_BOUNDS)

    print("  PASS")


def test_2_seeded_log_paths():
    print("\nTest 2: seeded log paths...")

    assert cp._seeded_log_path(cp.CURRICULUM_LOG, 2) == 'curriculum_log_seed2.csv'
    assert cp._seeded_log_path(cp.BASELINE_LOG, 2) == 'baseline_log_seed2.csv'

    print("  PASS")


def test_3_halfcheetah_xml_patch_and_cleanup():
    print("\nTest 3: HalfCheetah XML patching and cleanup...")

    params = {'gravity': -12.0, 'friction': 0.8, 'torso_mass': 4.0}
    env = cp.ParameterizedCheetahEnv(params)
    tmp_path = env.xml_path
    obs, _ = env.reset(seed=123)
    patched_g = float(env.unwrapped.model.opt.gravity[2])

    assert obs.shape[0] > 0, "HalfCheetah observation should be non-empty"
    assert abs(patched_g - params['gravity']) < 1e-4, (
        f"MuJoCo gravity is {patched_g}, expected {params['gravity']}"
    )

    env.close()
    assert not os.path.exists(tmp_path), f"Temp XML file leaked: {tmp_path}"

    print("  PASS")


def test_4_baseline_loop_plumbing_without_ppo():
    print("\nTest 4: vectorized baseline plumbing with mocks...")

    created_vec_env_sizes = []
    learned_steps = []

    class FakeEnv:
        def __init__(self, params=None):
            self.params = params

        def close(self):
            pass

    class FakeDummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]
            self.n_envs = len(self.envs)
            created_vec_env_sizes.append(self.n_envs)

        def close(self):
            for env in self.envs:
                env.close()

    class FakePPO:
        def __init__(self, _policy, env, verbose=0, device='auto', seed=0):
            self.env = env
            self.num_timesteps = 0

        @classmethod
        def load(cls, _path, env=None, seed=0, **_kwargs):
            return cls('MlpPolicy', env, seed=seed)

        def set_env(self, env):
            self.env = env

        def learn(self, total_timesteps, reset_num_timesteps=True):
            learned_steps.append((total_timesteps, reset_num_timesteps, self.env.n_envs))
            self.num_timesteps += 2048 * self.env.n_envs  # PPO collects a full rollout

        def save(self, _path):
            pass

    def fake_eval(_model, _env, n_eval_episodes=1, deterministic=True, warn=False):
        return 1.23, 0.0

    old_ppo = cp.PPO
    old_dummy_vec_env = cp.DummyVecEnv
    old_eval = cp.sb3_eval
    old_env = cp.ParameterizedCheetahEnv
    try:
        cp.PPO = FakePPO
        cp.DummyVecEnv = FakeDummyVecEnv
        cp.sb3_eval = fake_eval
        cp.ParameterizedCheetahEnv = FakeEnv

        with tempfile.TemporaryDirectory() as tmpdir:
            out_log = os.path.join(tmpdir, 'baseline_log.csv')
            cp.run_baseline(
                total_steps=16_384,
                eval_freq=8192,
                steps_per_iter=2048,
                n_envs=4,
                out_log=out_log,
                seed=0,
            )
            expected_log = os.path.join(tmpdir, 'baseline_log_seed0.csv')
            assert os.path.exists(expected_log), "Seed-specific baseline log not written"
    finally:
        cp.PPO = old_ppo
        cp.DummyVecEnv = old_dummy_vec_env
        cp.sb3_eval = old_eval
        cp.ParameterizedCheetahEnv = old_env

    assert created_vec_env_sizes == [4, 4, 4], (
        f"Expected bootstrap + 2 training vec envs of size 4, got {created_vec_env_sizes}"
    )
    assert learned_steps == [(2048, True, 4), (2048, False, 4)]

    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("HalfCheetah Curriculum Preflight")
    print("=" * 60)

    tests = [
        test_1_fixed_probe_and_test_sets,
        test_2_seeded_log_paths,
        test_3_halfcheetah_xml_patch_and_cleanup,
        test_4_baseline_loop_plumbing_without_ppo,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")

    print(f"\n{'=' * 60}")
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("Preflight passed. The multi-seed HalfCheetah run is ready.")
    else:
        print("Fix failures before launching the expensive run.")
    print("=" * 60)
