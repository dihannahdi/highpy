# Legacy Files

This directory contains files superseded by `pyproject.toml` (PEP 517/518).

- **`setup.py`** — Original setuptools-based build configuration. Retained for reference only. The authoritative build configuration is `../pyproject.toml`.

Do **not** use `python setup.py install`; use `pip install -e ".[dev]"` instead.
