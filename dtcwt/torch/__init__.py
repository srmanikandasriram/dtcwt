"""
This module contains the PyTorch backend for the DTCWT library.
"""

from __future__ import absolute_import

__all__ = [
    'Transform1d',
    'Pyramid',
]

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

# Only define these if torch is available
if _HAVE_TORCH:
    from dtcwt.torch.transform1d import Transform1d
    from dtcwt.torch.common import Pyramid
else:
    # Dummy classes if torch is not available
    class Transform1d:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch backend is not available. Install torch to use this backend.')
    
    class Pyramid:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch backend is not available. Install torch to use this backend.')

# vim:sw=4:sts=4:et
