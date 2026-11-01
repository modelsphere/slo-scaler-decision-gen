"""`__version__` must follow pyproject.toml — it drifted to 0.1.0 once."""

import pathlib
import re

import decision_gen


def test_version_matches_pyproject():
    text = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert m, "pyproject.toml has no version line"
    assert decision_gen.__version__ == m.group(1)
