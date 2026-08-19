"""Test package.

Present so the test modules can import shared helpers from ``conftest`` with a
relative import; without it pytest imports each module standalone and the
relative import fails.
"""
