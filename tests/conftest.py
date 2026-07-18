"""Make fisuralab importable whether or not `pip install -e .` has run (belt-and-suspenders for CI/local),
and run each pipeline case ONCE per test session (the ladder is compute-heavy; tests share the manifests)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data-pipeline"))


@pytest.fixture(scope="session")
def battery_manifest():
    from fisuralab import pipeline

    return pipeline.precompute("synthetic_battery")


@pytest.fixture(scope="session")
def bcl_manifest():
    from fisuralab import pipeline

    return pipeline.precompute("bcl_examples")
