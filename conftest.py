"""Pytest root configuration.

Excludes ``src/_archive/`` from collection. Code there is intentionally
kept for reference but is not part of the active pipeline and shouldn't
block the test run.
"""

collect_ignore_glob = ["src/_archive/*"]
