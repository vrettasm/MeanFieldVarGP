import numpy as np
from numba import njit
from numpy import array as array_t
from numpy import asarray
from numpy.linalg import solve, cholesky


def log_det(x: array_t):
    """
    Returns the log(det(x)), but more stable and accurate.

    :param x: input array (dim_x x dim_x).

    :return: log(det(x)) (dim_x x dim_x).

    Note: if the input is 1D-vector, it will return the
    log(det()) on the diagonal matrix.
    """

    # Make sure input is array.
    x = asarray(x)

    # If the input is scalar.
    if x.ndim == 0:
        # Return from here with the log.
        return np.log(x)
    # _end_if_

    # If the input is a 1-D vector.
    if x.ndim == 1:
        # Transform it to diagonal matrix.
        x = np.diag(x)
    else:
        # Get the number of rows/cols.
        rows, cols = x.shape

        # Make sure the array is square.
        if rows != cols:
            raise RuntimeError(" log_det: Rows != Cols.")
        # _end_if_
    # _end_if_

    # More stable than: log(det(x)).
    return 2.0 * np.sum(np.log(cholesky(x).diagonal()))
# _end_def_

@njit(fastmath=True)
def safe_log(x: array_t):
    """
    This function prevents the computation of very small,
    or very large values of logarithms that would lead to
    -/+ inf, by setting predefined LOWER and UPPER bounds.

    The bounds are set as follows:

        - LOWER = 1.0E-300
        - UPPER = 1.0E+300

    It is assumed that the input values lie within this range.

    Example:
        >> numpy.log(1.0E-350)
        >> -inf
        >>
        >> safe_log(1.0E-350)
        >> -690.77552789821368

    :param x: input array (dim_n x dim_m).

    :return: the log(x) after the values of x have been
    filtered (dim_n x dim_m).
    """

    # Make sure input is an array.
    x = asarray(x)

    # Filter out small and large values.
    x = np.maximum(np.minimum(x, 1.0E+300), 1.0E-300)

    # Return the log() of the filtered input.
    return np.log(x)
# _end_def_

@njit(fastmath=True)
def _cholesky_inv_fast(x: array_t):
    """
    Helper function implemented with numba.

    :param x: array to be inverted.
    """

    # Invert the Cholesky decomposition.
    c_inv = solve(cholesky(x), np.eye(x.shape[0]))

    # Invert input matrix.
    x_inv = c_inv.T.dot(c_inv)

    return x_inv, c_inv
# _end_def_

def cholesky_inv(x: array_t):
    """
    Inverts an input array (matrix) using Cholesky
    decomposition.

    :param x: input array (dim_d x dim_d)

    :return: inverted 'x' and inverted Cholesky factor.
    """

    # Make sure input is array.
    x = asarray(x)

    # Check if the input is scalar.
    if x.ndim == 0:
        return 1.0 / x, 1.0 / np.sqrt(x)
    else:

        # Check if the input is vector.
        if x.ndim == 1:

            # Convert it to diagonal matrix.
            x = np.diag(x)
        # _end_if_

        return _cholesky_inv_fast(x)
    # _end_if_

# _end_def_
