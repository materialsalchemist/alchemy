"""Build split.npz for original-QM9 PaiNN training.

Replays qm9_baselines.py's split procedure (rng seed=0, permutation, test_frac
~0.0764) against schnetpack's QM9 row order. schnetpack's QM9._download_data
sorts molecule files by (int(digits), filename) -- the same order
sorted(glob("dsgdb9nsd_*.xyz")) gives -- so these indices are index-identical
to the ones qm9_baselines.py and geometry_cell.py already used, not just a
similarly-distributed split.

New choice (undocumented in the original run, disclosed here): a validation
set is carved out of the train pool, sized at 5% of the full dataset, since no
original validation-set size survives.
"""
import numpy as np

N_TOTAL = 133_885
TEST_FRAC = 0.0764  # matches qm9_baselines.py --test-frac default, applied with its exact
                    # rounding rule (int(round(n * test_frac))). This actually gives 10,229,
                    # not the "~10,233" work.md's investigation described -- that figure was
                    # itself an approximate description of this same reconstructed procedure,
                    # not the (different, unrecoverable) Methods-section OpenBabel-filtered
                    # split count. 10,229 is the number that is genuinely index-identical with
                    # qm9_baselines.py and geometry_cell.py's existing published baselines.
VAL_FRAC = 0.05     # new, disclosed choice -- no original value survives
SEED = 0            # matches qm9_baselines.py --seed default

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_TOTAL)
n_test = int(round(N_TOTAL * TEST_FRAC))
n_val = int(round(N_TOTAL * VAL_FRAC))

test_idx = perm[:n_test]
val_idx = perm[n_test : n_test + n_val]
train_idx = perm[n_test + n_val :]

assert len(train_idx) + len(val_idx) + len(test_idx) == N_TOTAL
assert len(test_idx) == 10_229, f"test set size {len(test_idx)} != 10,229 -- check test_frac/rounding"

np.savez("split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
print(f"train {len(train_idx):,}  val {len(val_idx):,}  test {len(test_idx):,}")
