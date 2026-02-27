from __future__ import absolute_import

try:
    import torch
except ImportError:
    # The lack of torch will be caught by the low-level routines.
    pass


class Pyramid(object):
    """A PyTorch representation of a transform domain signal.

    An interface-compatible version of
    :py:class:`dtcwt.Pyramid` where the initialiser
    arguments are assumed to be :py:class:`torch.Tensor` instances.

    The attributes defined in :py:class:`dtcwt.Pyramid`
    are implemented via properties. The original torch tensors may be accessed
    via the ``..._op(s)`` attributes.

    .. py:attribute:: lowpass_op

        A PyTorch tensor that can be evaluated to return
        the coarsest scale lowpass signal for the input, X.

    .. py:attribute:: highpasses_ops

        A tuple of PyTorch tensors, where each element is the complex
        subband coefficients for corresponding scales finest to coarsest.

    .. py:attribute:: scales_ops

        *(optional)* A tuple where each element is a PyTorch tensor
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.
    """
    def __init__(self, lowpass, highpasses, scales=None, numpy=False):
        self.lowpass_op = lowpass
        self.highpasses_ops = highpasses
        self.scales_ops = scales
        self.numpy = numpy

    @property
    def lowpass(self):
        if not hasattr(self, '_lowpass'):
            if self.lowpass_op is None:
                self._lowpass = None
            elif self.numpy:
                self._lowpass = self.lowpass_op.detach().cpu().numpy()
            else:
                self._lowpass = self.lowpass_op
        return self._lowpass

    @property
    def highpasses(self):
        if not hasattr(self, '_highpasses'):
            if self.highpasses_ops is None:
                self._highpasses = None
            elif self.numpy:
                self._highpasses = tuple(
                    x.detach().cpu().numpy() if x is not None else None 
                    for x in self.highpasses_ops
                )
            else:
                self._highpasses = self.highpasses_ops
        return self._highpasses

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            if self.scales_ops is None:
                self._scales = None
            elif self.numpy:
                self._scales = tuple(
                    x.detach().cpu().numpy() if x is not None else None 
                    for x in self.scales_ops
                )
            else:
                self._scales = self.scales_ops
        return self._scales

# vim:sw=4:sts=4:et
