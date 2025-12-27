"""
Version information for lob-models.

Follows semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible
"""

__version__ = "0.1.0"

VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "alpha",
}


def get_version() -> str:
    """Get the current version string."""
    return __version__

