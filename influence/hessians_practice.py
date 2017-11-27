import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def hessian_vector_product(ys, xs, v):
    """
    Multiply the Hessian of `ys` w.r.t. `xs` by `v`.
    Args:
        ys: A scalar value, or a tensor or list of tensors to
            be summed to yield a scalar;
        xs: A list of tensors that we wish to construct the
            Hessian over;
        v:  A list of tensors with the same shape as `xs` that
            we wish to multiply by the Hessian;

    Returns:
        A list of tensors the same shape as `xs` and `v`
        containing the product between the Hessian and `v`.
    Raises:
        ValueError: `xs` and `v` are of different length
    """

    # Validate input
    if len(v) != len(xs):
        raise ValueError("xs and v must be of the same length.")

    # First Backprop
    grads = gradients(ys, xs)
    elementwise_products = [math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                            for grad_elem, v_elem in zip(grads,v) if grad_elem is not None]

    # Second Backprop
    grads_with_none = gradients(elementwise_products, xs)
    return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x)
                    for x, grad_elem in zip(xs, grads_with_none)]

    return return_grads

def _AsList(x):
    return x if isinstance(x,(list,tuple)) else [x]

def hessians(ys, xs, name="hessians", colocate_gradients_with_ops=False,
             gate_gradients=False, aggregation_method=None):

    """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.
    Args:
        ys:     A tensor or list of tensors we wish to differentiate;
        xs:     A tensor or list of tensors we wish to differentiate over;
        name:   Optional name to use for grouping all the gradient ops together;
        colocate_gradients_with_ops: See `gradients()` documentation for details;
        gate_gradients: same as above;
        aggregation_method: same as above;

    Returns:
        A list of Hessian matrices of `sum(y)` for each `x` in `xs`.

    Raises:
        LookupError: if one of the operations between `xs` and `ys` does not
                     have a registered gradient function;
        ValueError:  if the arguments are invalid or not supported. (1D)

    The function adds ops to the graph to output the Hessian matrix of `ys` wrt
    `xs`.
    """
    xs = _AsList(xs)
    kwargs = {
        'colocate_gradients_with_ops': colocate_gradients_with_ops,
        'gate_gradients': gate_gradients,
        'aggregation_method': aggregation_method
    }

    # compute a hessian matrix for each x in xs
    hessians = []
    for i, x in enumerate(xs):
        ndims = x.get_shape().ndims
        if ndims is None:
            raise ValueError('Cannot compute Hessian because dimensionality of'
                             ' element number %d of `xs` cannot be determined' % i)
        elif ndims != 1:
            raise ValueError('Computing Hessians is currently only supported for'
                             ' 1-D tensors. Element number %d of `xs` has %d '
                             'dimensions.' % (i, ndims))
        with ops.name_scope(name+'_first_derivative'): # add to tf framework ops
            _gradients = tf.gradients(ys, x, **kwargs)[0]
            _gradients = array_ops.unpack(_gradients)
        with ops.name_scope(name+'_second_derivative'):
            _hess = [tf.gradients(_gradient, x, **kwargs)[0] for _gradient in _gradients]
            hessians.append(array_ops.pack(_hess, name=name))

    return hessians
























