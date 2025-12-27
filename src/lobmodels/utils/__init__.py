"""
Utility modules for lob-models.

Contains:
- Feature layout transformation utilities
- Shape validation helpers
"""

from lobmodels.utils.feature_layout import (
    rearrange_grouped_to_fi2010,
    rearrange_fi2010_to_grouped,
    FeatureRearrangement,
)

__all__ = [
    "rearrange_grouped_to_fi2010",
    "rearrange_fi2010_to_grouped",
    "FeatureRearrangement",
]

