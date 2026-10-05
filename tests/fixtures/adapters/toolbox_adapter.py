"""Test adapter: a toy LightGBM detector (``toy_v1`` space) with switchable misbehaviour.

malvalid's own sandbox tests use it. The value of the first feature of the first row selects a
behaviour, so one sandboxed worker can serve many tests. Normal toy rows have ``X[0, 0] > 0``.
"""

import os
import pickle
import pwd
import resource
import signal
import socket
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent

MODE_N_BY_2 = -1
MODE_NAN = -2
MODE_OUT_OF_RANGE = -3
MODE_WRONG_LENGTH = -4
MODE_SLOW = -5
MODE_EXIT = -6
MODE_SEGFAULT = -7
MODE_RAISE = -8
MODE_PROBE = -9
MODE_NET_PROBE = -10
MODE_BAD_PREDICT = -11

PROBES = (
    "network_connect",
    "write_adapter_dir",
    "write_site_packages",
    "write_scratch",
    "secret_env_visible",
    "home_listable",
    "pickle_loads_refused",
    "rlimit_as_finite",
)


def _try(fn):
    try:
        fn()
        return 1.0
    except Exception:
        return 0.0


def _connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect(("1.1.1.1", 53))
    finally:
        s.close()


def _probe():
    connect = _connect

    def write(path):
        with open(path, "w") as f:
            f.write("malvalid sandbox test")

    def home_listable():
        home = pwd.getpwuid(os.getuid()).pw_dir
        if not os.listdir(home):
            raise OSError("empty")

    def pickle_refused():
        try:
            pickle.loads(pickle.dumps([1, 2, 3]))
        except Exception as e:  # PickleRefused
            if "allow-pickle" in str(e):
                return
            raise
        raise RuntimeError("pickle.loads worked")

    scratch = os.environ.get("MALVALID_SCRATCH", "/nonexistent")
    out = [
        _try(connect),
        _try(lambda: write(HERE / "sandbox_write_probe.txt")),
        _try(lambda: write(Path(np.__file__).resolve().parent.parent / "malvalid_sandbox_write_probe.txt")),
        _try(lambda: write(os.path.join(scratch, "probe.txt"))),
        1.0 if "MALVALID_TEST_SECRET" in os.environ else 0.0,
        _try(home_listable),
        _try(pickle_refused),
        1.0 if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY else 0.0,
    ]
    return np.asarray(out, dtype=np.float64)


class ToolboxDetector:
    feature_version = "toy_v1"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    training_hashes_path = "train_hashes.txt"
    training_cutoff = "2017-12"
    model_path = "model.txt"

    def __init__(self, booster):
        self.native_model = booster

    @classmethod
    def load(cls):
        return cls(lgb.Booster(model_file=str(HERE / cls.model_path)))

    def predict_proba(self, X):
        X = np.asarray(X)
        mode = float(X[0, 0]) if len(X) else 0.0
        if mode == MODE_PROBE:
            return _probe()[: len(X)]
        if mode == MODE_NET_PROBE:
            return np.full(len(X), _try(_connect))
        p = self.native_model.predict(X)
        if mode == MODE_N_BY_2:
            return np.column_stack([1 - p, p])
        if mode == MODE_NAN:
            p = p.copy()
            p[min(1, len(p) - 1)] = np.nan
            return p
        if mode == MODE_OUT_OF_RANGE:
            return p + 1.5
        if mode == MODE_WRONG_LENGTH:
            return p[:-1]
        if mode == MODE_SLOW:
            time.sleep(60)
        if mode == MODE_EXIT:
            sys.stderr.write("toolbox adapter: exiting on purpose\n")
            sys.stderr.flush()
            os._exit(7)
        if mode == MODE_SEGFAULT:
            os.kill(os.getpid(), signal.SIGSEGV)
        if mode == MODE_RAISE:
            raise ValueError("toolbox adapter raised on purpose")
        return p

    def predict(self, X):
        if len(X) and float(X[0, 0]) == MODE_BAD_PREDICT:
            return np.full(len(X), 2)
        return (self.predict_proba(X) >= self.operating_threshold).astype(np.int8)

    def featurize(self, raw):
        """Toy raw-bytes featurizer: size + 7-bin byte histogram, zero elsewhere (32 dims)."""
        b = np.frombuffer(raw, dtype=np.uint8)
        v = np.zeros(32, dtype=np.float32)
        v[0] = float(len(raw))
        if b.size:
            h = np.bincount(b // 37, minlength=7)[:7].astype(np.float32)
            v[1:8] = h / h.sum()
        return v
