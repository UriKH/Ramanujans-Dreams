"""δ of trajectories with huge exact denominators must not stall in sympy's assumption system.

``sp.Abs(Integer)`` asks whether the integer is non-negative; depending on the process's hash
seed (the order sympy walks its facts in) the inference reaches ``Integer.is_prime`` and runs a
pure-Python primality test on the reduced denominator — ~25 min for the 27,822-digit
denominator below (Zeus run #22, analysis task 257).  The seed is per process, so the
trajectory is evaluated in subprocesses with the seeds that stalled.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Euler–Gompertz in MeijerG(0,2,2,1,1): start (1,1,0), direction (3,3,-5), dual walk, depth 1000.
SCRIPT = """
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf.meijer_g import MeijerG
from dreamer.configs import config
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
config.configure(search={'DEFAULT_USES_INV_T': True})
cmf = MeijerG(0, 2, 2, 1, 1)
keys = list(cmf.matrices.keys())
h = TrajectoryAttributesHandler.from_cmf(
    cmf, Position(dict(zip(keys, (3, 3, -5)))), Position(dict(zip(keys, (1, 1, 0)))),
    constant=sp.E * sp.expint(1, 1), walk_depth=1000, walk_type=1)
print(h.identified(), h.delta())
"""


@pytest.mark.timeout(300)
@pytest.mark.parametrize("hash_seed", ["4", "6"])
def test_stalled_trajectory_delta_is_fast(hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, cwd=ROOT, capture_output=True, text=True,
                         timeout=90)
    identified, delta = out.stdout.split()[-2:]
    assert identified == "True"
    assert abs(float(delta) + 0.99446) < 1e-4
