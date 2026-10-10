"""
Curriculum Learning Pipeline for HalfCheetah
=============================================
Full pipeline: procedural env generation → behavioral embeddings →
difficulty surrogate (regression MLP) → curriculum loop → baseline comparison.

Run order:
  python curriculum_pipeline.py --phase train      # ~1-2 hrs, trains PPO checkpoints
  python curriculum_pipeline.py --phase dataset    # ~2-4 hrs, generates surrogate_training_data.csv
  python curriculum_pipeline.py --phase surrogate  # ~5 min, trains + validates surrogate
  python curriculum_pipeline.py --phase embed_compare  # expert vs curriculum embeddings (no norm_stats.npz)
  python curriculum_pipeline.py --phase bounds     # prints Goldilocks percentiles
  python curriculum_pipeline.py --phase run        # seeds 0,1,2: curriculum + baseline each
                                                   # append --warm-start to load strongest checkpoint
  python curriculum_pipeline.py --phase plot       # aggregate plot + stats from seed logs
  python curriculum_pipeline.py --phase full       # surrogate if needed, run, then plot
"""

import os
import re
import argparse
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

REWARD_MIN = -464.0
REWARD_MAX = 5026.0
SUCCESS_THRESHOLD = 1500.0      # used only for legacy binary labels
GOLDILOCKS_LOW  = 0.19          # 25th-pct of surrogate predictions on expert
GOLDILOCKS_HIGH = 0.24          # ~65th-pct of surrogate predictions on expert

DEFAULT_BOUNDS = {
    'gravity':    [-12.0, -4.0],   # Earth = -9.81
    'friction':   [0.2,   1.5],    # default ≈ 0.4
    'torso_mass': [3.0,  12.0],    # default ≈ 6.0
}
DEFAULT_PARAMS = {'gravity': -9.81, 'friction': 0.4, 'torso_mass': 6.0}

CHECKPOINT_DIR     = './checkpoints'
CSV_PATH           = 'surrogate_training_data.csv'
SURROGATE_PATH     = 'best_surrogate.pt'
CURRICULUM_LOG     = 'curriculum_log.csv'
BASELINE_LOG       = 'baseline_log.csv'
NORM_STATS_PATH    = 'norm_stats.npz'

# If True: curriculum + baseline start from random PPO weights (fair comparison vs expert
# surrogate; surrogate still guides env choice). Override with `--warm-start` on `--phase run`.
CURRICULUM_FROM_SCRATCH = True
PROBE_SEED = 42
TEST_ENV_SEED_OFFSET = 1000

# ─────────────────────────────────────────
# REWARD NORMALISATION
# ─────────────────────────────────────────

def normalize_reward(r):
    return np.clip((r - REWARD_MIN) / (REWARD_MAX - REWARD_MIN), 0.0, 1.0)

def denormalize_reward(r_norm):
    return r_norm * (REWARD_MAX - REWARD_MIN) + REWARD_MIN

def normalize_embedding(emb: np.ndarray, emb_mean: np.ndarray,
                         emb_std: np.ndarray) -> np.ndarray:
    """Safe normalization that handles near-zero std dimensions."""
    constant_dims        = emb_std < 0.01
    std_safe             = np.where(constant_dims, 1.0, emb_std)
    emb_n                = (emb - emb_mean) / std_safe
    emb_n[constant_dims] = 0.0
    return emb_n

# ─────────────────────────────────────────
# 1. PROCEDURAL ENVIRONMENT GENERATOR
# ─────────────────────────────────────────

def sample_random_params(rng=None):
    """Sample one physics configuration from the HalfCheetah parameter bounds."""
    if rng is None:
        return {k: float(np.random.uniform(v[0], v[1])) for k, v in DEFAULT_BOUNDS.items()}
    return {k: float(rng.uniform(v[0], v[1])) for k, v in DEFAULT_BOUNDS.items()}


def create_patched_xml(params: dict) -> str:
    """Patch HalfCheetah MuJoCo XML with custom physics params. Returns tmp file path."""
    import gymnasium.envs.mujoco as mj_envs
    base_path = os.path.dirname(mj_envs.__file__)
    xml_path  = os.path.join(base_path, 'assets', 'half_cheetah.xml')

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Gravity
    option = root.find('option')
    if option is not None:
        option.set('gravity', f"0 0 {params['gravity']:.4f}")

    # Torso mass
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for body in worldbody.findall('body'):
            if body.get('name') == 'torso':
                geom = body.find('geom')
                if geom is not None:
                    geom.set('mass', f"{params['torso_mass']:.4f}")

    # Friction
    default = root.find('default')
    if default is not None:
        geom_default = default.find('geom')
        if geom_default is not None:
            geom_default.set('friction', f"{params['friction']:.4f} .1 .1")

    fd, tmp_path = tempfile.mkstemp(suffix='.xml')
    with os.fdopen(fd, 'w') as f:
        tree.write(tmp_path, encoding='unicode')
    return tmp_path


class ParameterizedCheetahEnv(gym.Wrapper):
    def __init__(self, params=None):
        self.params   = params if params is not None else sample_random_params()
        self.xml_path = create_patched_xml(self.params)
        env = gym.make('HalfCheetah-v5', xml_file=self.xml_path)
        super().__init__(env)

    def close(self):
        super().close()
        if os.path.exists(self.xml_path):
            os.remove(self.xml_path)

def _seeded_log_path(base_path, seed):
    stem = base_path.replace('.csv', '')
    return f'{stem}_seed{seed}.csv'


def fixed_probe_params(n_probe_envs=50):
    """Probe environments must match surrogate dataset generation exactly."""
    rng = np.random.RandomState(PROBE_SEED)
    return [sample_random_params(rng=rng) for _ in range(n_probe_envs)]


def fixed_test_params(seed, n_test_envs=20):
    """Matched held-out test environments for curriculum and baseline."""
    rng = np.random.RandomState(seed + TEST_ENV_SEED_OFFSET)
    return [
        sample_random_params(rng=rng)
        for _ in range(n_test_envs)
    ]

# ─────────────────────────────────────────
# 2. TRAIN BASE PPO AGENT
# ─────────────────────────────────────────

def train_base_agent(total_timesteps=1_000_000, save_dir=CHECKPOINT_DIR):
    print('─' * 60)
    print('PHASE: train_base_agent')
    print('─' * 60)
    os.makedirs(save_dir, exist_ok=True)

    env   = ParameterizedCheetahEnv(DEFAULT_PARAMS)
    model = PPO('MlpPolicy', env, verbose=0, device='auto')

    cb = CheckpointCallback(
        save_freq  = 100_000,
        save_path  = save_dir,
        name_prefix= 'ppo_cheetah',
    )
    model.learn(total_timesteps=total_timesteps, callback=cb)
    env.close()
    print(f'Done. Checkpoints saved to {save_dir}/')

# ─────────────────────────────────────────
# 3. BEHAVIORAL EMBEDDING
# ─────────────────────────────────────────

def evaluate_policy_custom(model, env, n_episodes=3):
    """Returns (mean_reward, std_reward, success_rate)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            total += r
        rewards.append(total)
    rewards = np.array(rewards)
    success = (rewards > SUCCESS_THRESHOLD).mean()
    return rewards.mean(), rewards.std(), success


class RandomPolicy:
    """Zero-skill baseline — acts uniformly at random."""
    def predict(self, obs, deterministic=True):
        return np.random.uniform(-1, 1, size=(6,)), None


def compute_behavioral_embedding(model, probe_params_list: list) -> np.ndarray:
    """
    Rolls model through each probe environment (3 episodes each).
    Returns a 150-D vector: [mean_rew, std_rew, failure_rate] × 50 probes.
    """
    embedding = []
    for params in probe_params_list:
        env = ParameterizedCheetahEnv(params)
        mean_r, std_r, success = evaluate_policy_custom(model, env, n_episodes=3)
        env.close()
        embedding.extend([mean_r, std_r, 1.0 - success])
    return np.array(embedding, dtype=np.float32)

# ─────────────────────────────────────────
# 4. DATASET GENERATION
# ─────────────────────────────────────────

def extract_steps(ckpt_name: str) -> int:
    m = re.search(r'(\d+)_steps', ckpt_name)
    return int(m.group(1)) if m else 0


def build_dataset(
    checkpoint_dir      = CHECKPOINT_DIR,
    n_probe_envs        = 50,
    n_rollouts_per_ckpt = 1000,
    out_csv             = CSV_PATH,
):
    print('─' * 60)
    print('PHASE: build_dataset')
    print('─' * 60)

    probe_params = fixed_probe_params(n_probe_envs)

    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
        key=extract_steps,
    )
    print(f'Found {len(checkpoints)} checkpoints.')

    rows = []
    for ckpt_name in tqdm(checkpoints, desc='Checkpoints'):
        model = PPO.load(os.path.join(checkpoint_dir, ckpt_name))
        skill_emb = compute_behavioral_embedding(model, probe_params)

        for _ in range(n_rollouts_per_ckpt):
            rp  = sample_random_params()
            env = ParameterizedCheetahEnv(rp)
            mean_r, _, success = evaluate_policy_custom(model, env, n_episodes=1)
            env.close()

            row = {
                'checkpoint' : ckpt_name,
                'step_count' : extract_steps(ckpt_name),
                'gravity'    : rp['gravity'],
                'friction'   : rp['friction'],
                'torso_mass' : rp['torso_mass'],
                'mean_reward': mean_r,
                'success_label': float(success),
                'target'     : float(normalize_reward(mean_r)),
            }
            for i, v in enumerate(skill_emb):
                row[f'emb_{i}'] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'Saved {len(df)} rows → {out_csv}')

# ─────────────────────────────────────────
# 5. DIFFICULTY SURROGATE (MLP, regression)
# ─────────────────────────────────────────

class DifficultySurrogate(nn.Module):
    def __init__(self, skill_dim: int, env_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(skill_dim + env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),          # output ∈ [0,1] = normalized reward
        )

    def forward(self, skill_emb, env_params):
        return self.net(torch.cat([skill_emb, env_params], dim=-1)).squeeze(-1)


def load_and_preprocess(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    if 'step_count' not in df.columns:
        df['step_count'] = df['checkpoint'].apply(extract_steps)
    df = df.sort_values('step_count').reset_index(drop=True)

    if 'target' not in df.columns:
        df['target'] = normalize_reward(df['mean_reward'].values).clip(0, 1)

    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    env_cols  = ['gravity', 'friction', 'torso_mass']

    X_emb = df[emb_cols].values.astype(np.float32)
    X_env = df[env_cols].values.astype(np.float32)
    y     = df['target'].values.astype(np.float32)

    env_mean, env_std = X_env.mean(0), X_env.std(0)
    emb_mean, emb_std = X_emb.mean(0), X_emb.std(0)

    X_env_n = (X_env - env_mean) / (env_std + 1e-8)

    constant_dims     = emb_std < 0.01
    emb_std_clamped   = np.where(constant_dims, 1.0, emb_std)
    X_emb_n           = (X_emb - emb_mean) / emb_std_clamped
    X_emb_n[:, constant_dims] = 0.0   # zero signal → zero input

    n_constant = constant_dims.sum()
    if n_constant > 0:
        print(f"[preprocess] Zeroed {n_constant}/{len(emb_std)} constant embedding dims")

    return (df, X_emb_n, X_env_n, y, emb_cols,
            env_mean, env_std, emb_mean, emb_std)


def train_surrogate(csv_path=CSV_PATH, surrogate_path=SURROGATE_PATH,
                    norm_stats_path=NORM_STATS_PATH):
    print('─' * 60)
    print('PHASE: train_surrogate')
    print('─' * 60)

    df, X_emb_n, X_env_n, y, emb_cols, env_mean, env_std, emb_mean, emb_std = \
        load_and_preprocess(csv_path)

    # Split by checkpoint — NOT random — to test generalisation to new skill levels
    ckpts_ordered   = df['checkpoint'].unique()          # sorted by step_count already
    test_ckpt       = ckpts_ordered[-1]
    val_ckpt        = ckpts_ordered[-2]
    train_mask      = ~df['checkpoint'].isin([test_ckpt, val_ckpt])
    val_mask        = df['checkpoint'] == val_ckpt
    test_mask       = df['checkpoint'] == test_ckpt

    def to_tensors(mask):
        return (torch.tensor(X_emb_n[mask]),
                torch.tensor(X_env_n[mask]),
                torch.tensor(y[mask]))

    emb_tr, env_tr, y_tr = to_tensors(train_mask)
    emb_val, env_val, y_val = to_tensors(val_mask)
    emb_te, env_te, y_te   = to_tensors(test_mask)

    print(f'Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}')

    loader = DataLoader(TensorDataset(emb_tr, env_tr, y_tr),
                        batch_size=256, shuffle=True)

    model     = DifficultySurrogate(skill_dim=len(emb_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val, patience_ctr, patience = float('inf'), 0, 10

    for epoch in range(200):
        model.train()
        total_loss = 0.0
        for eb, envb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(eb, envb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(emb_val, env_val)
            val_loss = criterion(val_pred, y_val).item()
            val_mae  = np.abs(
                denormalize_reward(val_pred.numpy()) -
                denormalize_reward(y_val.numpy())
            ).mean()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | train={total_loss/len(loader):.4f} '
                  f'| val={val_loss:.4f} | val_MAE={val_mae:.1f} reward pts')

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), surrogate_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # ── Held-out evaluation ──
    model.load_state_dict(torch.load(surrogate_path))
    model.eval()
    with torch.no_grad():
        test_pred = model(emb_te, env_te).numpy()

    true_r = denormalize_reward(y_te.numpy())
    pred_r = denormalize_reward(test_pred)
    mae    = np.abs(pred_r - true_r).mean()
    r      = np.corrcoef(pred_r, true_r)[0, 1]
    rand_mae = np.abs(true_r - true_r.mean()).mean()

    print(f'\n=== HELD-OUT EXPERT TEST ===')
    print(f'MAE:              {mae:.1f} reward pts')
    print(f'Pearson r:        {r:.3f}')
    print(f'Random-baseline MAE: {rand_mae:.1f}')

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_r, pred_r, alpha=0.3, s=10)
    plt.plot([REWARD_MIN, REWARD_MAX], [REWARD_MIN, REWARD_MAX], 'r--', label='Perfect')
    plt.xlabel('Actual Reward'); plt.ylabel('Predicted Reward')
    plt.title(f'Surrogate on Held-Out Expert\nr={r:.3f}, MAE={mae:.1f}')
    plt.legend(); plt.tight_layout()
    plt.savefig('surrogate_scatter.png', dpi=150)
    print('Scatter saved → surrogate_scatter.png')

    # Save normalisation stats for use in the curriculum loop
    np.savez(norm_stats_path,
             env_mean=env_mean, env_std=env_std,
             emb_mean=emb_mean, emb_std=emb_std)
    print(f'Norm stats saved → {norm_stats_path}')

    return model, (env_mean, env_std, emb_mean, emb_std)

# ─────────────────────────────────────────
# 6. DIAGNOSTIC — does the surrogate use skill embeddings?
# ─────────────────────────────────────────

def run_diagnostic(csv_path=CSV_PATH, surrogate_path=SURROGATE_PATH):
    print('─' * 60)
    print('PHASE: diagnostic')
    print('─' * 60)

    df, X_emb_n, X_env_n, y, emb_cols, *_ = load_and_preprocess(csv_path)

    surrogate = DifficultySurrogate(skill_dim=len(emb_cols))
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()

    expert_mask   = df['step_count'] == df['step_count'].max()
    beginner_mask = df['step_count'] == df['step_count'].min()

    emb_exp = torch.tensor(X_emb_n[expert_mask])
    env_exp = torch.tensor(X_env_n[expert_mask])
    emb_beg = torch.tensor(X_emb_n[beginner_mask])

    n = min(emb_exp.shape[0], emb_beg.shape[0])

    with torch.no_grad():
        pred_normal  = surrogate(emb_exp, env_exp).numpy()
        pred_zeroed  = surrogate(torch.zeros_like(emb_exp), env_exp).numpy()
        pred_expert  = surrogate(emb_exp[:n], env_exp[:n]).numpy()
        pred_beginner= surrogate(emb_beg[:n], env_exp[:n]).numpy()

    delta_zero  = np.abs(pred_normal - pred_zeroed).mean()
    skill_delta = (pred_expert - pred_beginner).mean()

    print(f'Mean change when embedding zeroed:  {delta_zero:.4f}  '
          f'({delta_zero*(REWARD_MAX-REWARD_MIN):.1f} reward pts)')
    print(f'Expert emb mean prediction:         {pred_expert.mean():.4f}')
    print(f'Beginner emb mean prediction:       {pred_beginner.mean():.4f}')
    print(f'Skill effect (expert - beginner):   {skill_delta:.4f}  '
          f'({skill_delta*(REWARD_MAX-REWARD_MIN):.1f} reward pts)')
    print()
    if delta_zero * (REWARD_MAX - REWARD_MIN) > 200:
        print('✓ Surrogate is using skill embeddings meaningfully.')
    else:
        print('✗ WARNING: surrogate appears to be ignoring skill embeddings.')
        print('  → Consider training PPO to 3M steps for larger skill gap.')

# ─────────────────────────────────────────
# 7. GOLDILOCKS BOUNDS ANALYSIS
# ─────────────────────────────────────────

def run_bounds(csv_path=CSV_PATH, surrogate_path=SURROGATE_PATH):
    print('─' * 60)
    print('PHASE: bounds')
    print('─' * 60)

    df, X_emb_n, X_env_n, y, emb_cols, *_ = load_and_preprocess(csv_path)

    surrogate = DifficultySurrogate(skill_dim=len(emb_cols))
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()

    expert_mask = df['step_count'] == df['step_count'].max()
    emb_exp     = torch.tensor(X_emb_n[expert_mask])
    env_exp     = torch.tensor(X_env_n[expert_mask])

    with torch.no_grad():
        preds = surrogate(emb_exp, env_exp).numpy()

    for q in [0, 10, 25, 50, 65, 75, 90, 100]:
        v = np.percentile(preds, q)
        print(f'  {q:3d}th pct: {v:.4f}  ({denormalize_reward(v):.0f} reward pts)')

    low  = np.percentile(preds, 25)
    high = np.percentile(preds, 65)
    print(f'\nRecommended Goldilocks: LOW={low:.4f}  HIGH={high:.4f}')
    print(f'(Edit GOLDILOCKS_LOW / GOLDILOCKS_HIGH at top of file if needed)')

# ─────────────────────────────────────────
# 8. CURRICULUM LOOP
# ─────────────────────────────────────────

def _load_surrogate(emb_dim, surrogate_path=SURROGATE_PATH):
    """Load trained surrogate weights. Normalisation means/std come from ``load_and_preprocess``."""
    surrogate = DifficultySurrogate(skill_dim=emb_dim)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location='cpu'))
    surrogate.eval()
    return surrogate


def _pick_goldilocks_env(surrogate, skill_emb_n, env_mean, env_std,
                          emb_mean=None, emb_std=None, ood_reference_emb=None,
                          n_candidates=500, ood_threshold=None):
    """
    Sample n_candidates random envs, return the params of one in the
    Goldilocks zone (or the closest one if the zone is empty).

    If ``ood_threshold`` is set, ``ood_reference_emb`` must be provided. OOD is
    detected when the L2 norm of that normalized embedding exceeds the threshold.
    Returns ``(chosen, n_gold, fallback, is_ood, l2)`` in that case; otherwise
    ``(chosen, n_gold, fallback)``. ``emb_mean`` / ``emb_std`` are accepted for
    call-site compatibility.
    """
    if ood_threshold is not None:
        if ood_reference_emb is None:
            raise ValueError('ood_reference_emb is required when ood_threshold is set')
        l2 = float(np.linalg.norm(np.asarray(ood_reference_emb, dtype=np.float64)))
        if l2 > ood_threshold:
            return {}, 0, True, True, l2

    candidates = [sample_random_params() for _ in range(n_candidates)]
    c_arr      = np.array([[c['gravity'], c['friction'], c['torso_mass']]
                            for c in candidates], dtype=np.float32)
    c_n        = (c_arr - env_mean) / (env_std + 1e-8)

    skill_t = torch.tensor(skill_emb_n, dtype=torch.float32)
    c_t     = torch.tensor(c_n)

    with torch.no_grad():
        preds = surrogate(skill_t.unsqueeze(0).repeat(n_candidates, 1), c_t).numpy()

    mask = (preds >= GOLDILOCKS_LOW) & (preds <= GOLDILOCKS_HIGH)
    if mask.sum() > 0:
        idx = np.random.choice(np.where(mask)[0])
        chosen, n_gold, fb = candidates[idx], int(mask.sum()), False
    else:
        center = (GOLDILOCKS_LOW + GOLDILOCKS_HIGH) / 2.0
        idx    = int(np.argmin(np.abs(preds - center)))
        chosen, n_gold, fb = candidates[idx], 0, True

    if ood_threshold is not None:
        l2 = float(np.linalg.norm(np.asarray(ood_reference_emb, dtype=np.float64)))
        return chosen, n_gold, fb, False, l2
    return chosen, n_gold, fb


from stable_baselines3.common.vec_env import DummyVecEnv


def run_curriculum_loop(
    csv_path=CSV_PATH,
    surrogate_path=SURROGATE_PATH,
    total_curriculum_steps=2_000_000,
    eval_freq=200_000,
    steps_per_iter=2048,
    n_envs=4,
    out_log=CURRICULUM_LOG,
    from_scratch=CURRICULUM_FROM_SCRATCH,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('─' * 60)
    print(f'PHASE: vectorized curriculum loop (seed={seed})')
    print('─' * 60)

    out_log = _seeded_log_path(out_log, seed)

    df, X_emb_n, _, _, emb_cols, env_mean, env_std, emb_mean, emb_std = \
        load_and_preprocess(csv_path)

    surrogate = _load_surrogate(len(emb_cols), surrogate_path)

    probe_params = fixed_probe_params()
    test_env_params = fixed_test_params(seed)

    bootstrap_vec = DummyVecEnv([
        lambda: ParameterizedCheetahEnv(DEFAULT_PARAMS) for _ in range(n_envs)
    ])

    if from_scratch:
        print('Policy init: random scratch weights')
        model = PPO(
            'MlpPolicy', bootstrap_vec, verbose=0, device='auto', seed=seed,
        )
    else:
        ckpts = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip')],
            key=extract_steps,
        )
        best_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        print(f'Policy warm-start: {best_ckpt}')
        model = PPO.load(best_ckpt, env=bootstrap_vec, seed=seed)

    steps_done, log = 0, []
    first_learn = True
    start_steps = 0 if from_scratch else model.num_timesteps  # warm-start keeps its count

    while steps_done < total_curriculum_steps:
        skill_emb = compute_behavioral_embedding(model, probe_params)
        skill_emb_n = normalize_embedding(skill_emb, emb_mean, emb_std)

        env_fns    = []
        total_gold = 0
        is_ood     = False
        l2         = 0.0

        for _ in range(n_envs):
            chosen_params, n_gold, _fallback, is_ood, l2 = _pick_goldilocks_env(
                surrogate, skill_emb_n, env_mean, env_std,
                emb_mean=emb_mean, emb_std=emb_std,
                ood_reference_emb=skill_emb_n,
                ood_threshold=45.0
            )
            total_gold += n_gold
            # OOD: surrogate is not trusted, so this env is uniform random (= baseline)
            p = chosen_params if chosen_params else sample_random_params()
            env_fns.append(lambda params=p: ParameterizedCheetahEnv(params))

        vec_env = DummyVecEnv(env_fns)
        model.set_env(vec_env)

        reset_ts = first_learn and from_scratch
        model.learn(
            total_timesteps=steps_per_iter,
            reset_num_timesteps=reset_ts,
        )
        first_learn = False
        vec_env.close()

        # learn() always collects a full rollout (n_steps * n_envs), so count real env steps
        prev_steps, steps_done = steps_done, model.num_timesteps - start_steps

        if steps_done // eval_freq > prev_steps // eval_freq:
            test_rewards = []
            for tp in test_env_params:
                te = ParameterizedCheetahEnv(tp)
                mean_r, _ = sb3_eval(
                    model, te, n_eval_episodes=1, deterministic=True, warn=False)
                te.close()
                test_rewards.append(mean_r)

            mean_test = float(np.mean(test_rewards))
            log.append({
                'steps': steps_done,
                'mean_test_reward': mean_test,
                'avg_gold': total_gold / n_envs,
                'is_ood': is_ood,
                'l2': l2,
            })
            print(
                f'Step {steps_done:6d} | test_reward={mean_test:7.1f} | '
                f'avg_gold={total_gold/n_envs:.0f} | ood={is_ood} | l2={l2:.1f}'
            )

    bootstrap_vec.close()
    log_df = pd.DataFrame(log)
    log_df.to_csv(out_log, index=False)
    model.save(f'curriculum_agent_seed{seed}')
    return model, log_df


# ─────────────────────────────────────────
# 9. BASELINE (random environments, no curriculum)
# ─────────────────────────────────────────

def run_baseline(
    total_steps=2_000_000,
    eval_freq=200_000,
    steps_per_iter=2048,
    n_envs=4,
    out_log=BASELINE_LOG,
    from_scratch=CURRICULUM_FROM_SCRATCH,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('─' * 60)
    print(f'PHASE: vectorized baseline loop (seed={seed})')
    print('─' * 60)

    out_log = _seeded_log_path(out_log, seed)

    test_env_params = fixed_test_params(seed)

    bootstrap_vec = DummyVecEnv([
        lambda: ParameterizedCheetahEnv(DEFAULT_PARAMS) for _ in range(n_envs)
    ])
    if from_scratch:
        print('Baseline policy init: random (scratch; matches curriculum protocol)')
        model = PPO(
            'MlpPolicy', bootstrap_vec, verbose=0, device='auto', seed=seed,
        )
    else:
        ckpts = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip')],
            key=extract_steps,
        )
        best_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        print(f'Baseline warm-start: {best_ckpt}')
        model = PPO.load(best_ckpt, env=bootstrap_vec, seed=seed)

    steps_done, log = 0, []
    first_learn = True
    start_steps = 0 if from_scratch else model.num_timesteps  # warm-start keeps its count

    while steps_done < total_steps:
        rand_env_fns = [
            lambda p=sample_random_params(): ParameterizedCheetahEnv(p)
            for _ in range(n_envs)
        ]
        vec_env = DummyVecEnv(rand_env_fns)
        model.set_env(vec_env)
        reset_ts = first_learn and from_scratch
        model.learn(
            total_timesteps=steps_per_iter,
            reset_num_timesteps=reset_ts,
        )
        first_learn = False
        vec_env.close()
        prev_steps, steps_done = steps_done, model.num_timesteps - start_steps

        if steps_done // eval_freq > prev_steps // eval_freq:
            test_rewards = []
            for tp in test_env_params:
                te = ParameterizedCheetahEnv(tp)
                mean_r, _ = sb3_eval(
                    model, te, n_eval_episodes=1,
                    deterministic=True, warn=False,
                )
                te.close()
                test_rewards.append(mean_r)
            mean_test = float(np.mean(test_rewards))
            log.append({'steps': steps_done, 'mean_test_reward': mean_test})
            print(f'Step {steps_done:6d} | test_reward={mean_test:7.1f}')

    bootstrap_vec.close()
    log_df = pd.DataFrame(log)
    log_df.to_csv(out_log, index=False)
    model.save(f'baseline_agent_seed{seed}')
    print(f'Baseline done. Log → {out_log}')
    return model, log_df


def run_multiseed(seeds=(0, 1, 2), from_scratch=CURRICULUM_FROM_SCRATCH):
    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f'SEED {seed}: CURRICULUM')
        print(f"{'=' * 60}")
        run_curriculum_loop(seed=seed, from_scratch=from_scratch)

        print(f"\n{'=' * 60}")
        print(f'SEED {seed}: BASELINE')
        print(f"{'=' * 60}")
        run_baseline(seed=seed, from_scratch=from_scratch)

    plot_comparison(seeds=seeds)


def run_full_experiment(seeds=(0, 1, 2), from_scratch=CURRICULUM_FROM_SCRATCH):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run --phase train and --phase dataset first."
        )

    print('Training surrogate from current code and dataset.')
    train_surrogate()

    run_multiseed(seeds=seeds, from_scratch=from_scratch)


# ─────────────────────────────────────────
# 10. PLOT COMPARISON
# ─────────────────────────────────────────

def plot_comparison(seeds=(0, 1, 2)):
    def load_logs(pattern):
        dfs = []
        for seed in seeds:
            path = pattern.replace('.csv', f'_seed{seed}.csv')
            if os.path.exists(path):
                d = pd.read_csv(path)
                d['seed'] = seed
                dfs.append(d)
        return dfs

    curr_dfs     = load_logs(CURRICULUM_LOG)
    baseline_dfs = load_logs(BASELINE_LOG)

    if not curr_dfs or not baseline_dfs:
        print("No log files found. Run --phase run first.")
        return

    curr_df     = pd.concat(curr_dfs)
    baseline_df = pd.concat(baseline_dfs)

    def summarize(df):
        return df.groupby('steps')['mean_test_reward'].agg(['mean', 'std']).reset_index()

    c_summary = summarize(curr_df)
    b_summary = summarize(baseline_df)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(c_summary['steps'], c_summary['mean'],
            'b-o', label='Curriculum (surrogate-guided)')
    ax.fill_between(c_summary['steps'],
                    c_summary['mean'] - c_summary['std'],
                    c_summary['mean'] + c_summary['std'],
                    alpha=0.2, color='blue')

    ax.plot(b_summary['steps'], b_summary['mean'],
            'r-o', label='Baseline (uniform domain randomization)')
    ax.fill_between(b_summary['steps'],
                    b_summary['mean'] - b_summary['std'],
                    b_summary['mean'] + b_summary['std'],
                    alpha=0.2, color='red')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Test Reward (20 held-out envs)')
    ax.set_title(f'Curriculum vs Baseline ({len(curr_dfs)} seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('curriculum_vs_baseline_multiseed.png', dpi=150)
    print('Plot saved → curriculum_vs_baseline_multiseed.png')

    # Summary stats
    final_c = curr_df.groupby('seed')['mean_test_reward'].last()
    final_b = baseline_df.groupby('seed')['mean_test_reward'].last()
    pct_improvements = (
        (final_c - final_b) / np.maximum(np.abs(final_b), 1e-8) * 100
    )

    print(f"\n=== MULTI-SEED RESULTS ({len(curr_dfs)} seeds) ===")
    for seed in seeds:
        if seed in final_c.index and seed in final_b.index:
            print(f"Seed {seed}: curriculum={final_c[seed]:.1f}  "
                  f"baseline={final_b[seed]:.1f}  "
                  f"improvement={pct_improvements[seed]:+.1f}%")
    print(f"\nMean improvement: {pct_improvements.mean():+.1f}% "
          f"± {pct_improvements.std():.1f}%")


def run_embed_compare_diagnostic(
    csv_path=CSV_PATH,
    expert_ckpt=None,
    curriculum_ckpt=None,
    probe_seed=42,
):
    """
    Compare expert and curriculum behavioral embeddings on the fixed probe set.
    """
    _, _, _, _, _, _, _, emb_mean, emb_std = load_and_preprocess(csv_path)

    if expert_ckpt is None:
        expert_ckpt = os.path.join(CHECKPOINT_DIR, 'ppo_cheetah_1000000_steps.zip')
    if curriculum_ckpt is None:
        for cand in ('curriculum_agent_seed0.zip',):
            if os.path.isfile(cand):
                curriculum_ckpt = cand
                break

    if probe_seed != PROBE_SEED:
        print(
            f'WARNING: probe_seed={probe_seed} differs from surrogate '
            f'PROBE_SEED={PROBE_SEED}; embeddings may not be comparable.'
        )
    probe_params = fixed_probe_params()

    expert_model = PPO.load(expert_ckpt, device='cpu')
    expert_emb   = compute_behavioral_embedding(expert_model, probe_params)
    expert_emb_n = normalize_embedding(expert_emb, emb_mean, emb_std)

    print(f'Expert checkpoint: {expert_ckpt}')
    print('Expert embedding normalised range:')
    print(f'  min: {expert_emb_n.min():.3f}  max: {expert_emb_n.max():.3f}')
    pct_out = float((np.abs(expert_emb_n) > 2).mean() * 100)
    print(f'  % dims outside [-2, 2]: {pct_out:.1f}%')

    if curriculum_ckpt is None:
        print('\n(No curriculum_agent_seed0.zip found — skipping curriculum comparison.)')
        return

    curr_model = PPO.load(curriculum_ckpt, device='cpu')
    curr_emb   = compute_behavioral_embedding(curr_model, probe_params)
    curr_emb_n = normalize_embedding(curr_emb, emb_mean, emb_std)

    print(f'\nCurriculum checkpoint: {curriculum_ckpt}')
    print('Curriculum agent embedding normalised range:')
    print(f'  min: {curr_emb_n.min():.3f}  max: {curr_emb_n.max():.3f}')
    print(f'  % dims outside [-2, 2]: {float((np.abs(curr_emb_n) > 2).mean() * 100):.1f}%')

    print('\nDifference in embedding space:')
    print(f'  L2 distance: {np.linalg.norm(expert_emb_n - curr_emb_n):.3f}')
    print(f'  (Expert L2 from origin: {np.linalg.norm(expert_emb_n):.3f})')


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

PHASES = {
    'train'         : lambda: train_base_agent(),
    'dataset'       : lambda: build_dataset(),
    'surrogate'     : lambda: train_surrogate(),
    'diagnostic'    : lambda: run_diagnostic(),
    'bounds'        : lambda: run_bounds(),
    'embed_compare' : lambda: run_embed_compare_diagnostic(),
    'run'           : lambda: run_multiseed(seeds=(0, 1, 2)),
    'plot'          : lambda: plot_comparison(seeds=(0, 1, 2)),
    'full'          : lambda: run_full_experiment(seeds=(0, 1, 2)),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase', required=True,
        choices=list(PHASES.keys()),
        help='Which phase to run',
    )
    parser.add_argument(
        '--warm-start',
        action='store_true',
        help='(--phase run) Load strongest PPO from checkpoints instead of scratch init.',
    )
    args = parser.parse_args()

    if args.phase == 'run':
        run_multiseed(
            seeds=(0, 1, 2),
            from_scratch=CURRICULUM_FROM_SCRATCH and not args.warm_start,
        )
    elif args.phase == 'full':
        run_full_experiment(
            seeds=(0, 1, 2),
            from_scratch=CURRICULUM_FROM_SCRATCH and not args.warm_start,
        )
    else:
        PHASES[args.phase]()