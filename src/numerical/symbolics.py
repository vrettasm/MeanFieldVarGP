import sympy as sym
from sympy.utilities.iterables import flatten


def LagrangePolynomial(letter: str = "m", order: int = 3, fp: str = "t"):
    """
    Returns a Lagrange polynomial in symbolic representation.

    https://en.wikipedia.org/wiki/Lagrange_polynomial

    :param letter: this is the letter that will be used as function name.
    We typically use 'm' for the mean functions and 's' for the variance
    functions. Note that in terms of variables the returned function will
    only "know" the names of 'xi' and 'ti', where:

        xi[0] == m0, xi[1] == m1, xi[2] == m2, etc.
        ti[0] == t0, ti[1] == t1, ti[2] == t2, etc.

    :param order: this is the order of the polynomials. Typically, we will
    use 3rd order (order=3) for the mean functions 'mt', and 2nd order
    (order=2) for the variance functions 'st'.

    :param fp: this is the letter to identify the fixed mid-points.

    EXAMPLE(s):
        # Here we define the function m(t).
        mt, t, mk, _ = LagrangePolynomial(letter="m", order=3)

        # Here we take the derivatives w.r.t. time 't'.
        dmt_dt = mt.diff(t)

        # Here we take the derivatives w.r.t. the first points mk0.
        dmt_dm0 = mt.diff(mk[0])

        etc.

    :return: the Lagrange polynomial in symbolic notation. In addition,
    it also returns all the symbolic variables to allow for the gradient
    calculations.
    """

    # Make sure order is integer.
    order = int(order)

    # Sanity check.
    if order < 0:
        raise ValueError(f" LagrangePolynomial({letter}, {order}):"
                          " Order value can't be negative.")
    # _end_if_

    # Define the time variable 't' as symbol.
    t = sym.Symbol("t", real=True, nonnegative=True)

    # Define the fixed time points 'tk' as symbols.
    ti = sym.symbols(f"{fp}:{order + 1}", real=True, positive=True)

    # Define the fixed function points 'xk' as symbols.
    xi = sym.symbols(f"{letter}:{order + 1}", real=True)

    # Declare the return polynomial.
    poly_func = None

    # Construct the Lagrange polynomial iteratively.
    for k in range(0, order + 1):

        # This list will hold the product for each index 'k'.
        partial_k = []

        for l in range(0, order + 1):

            # Make sure we avoid the same index l==k.
            # This will make sure the denominator will
            # not be equal to zero.
            if l == k:
                continue
            # _end_if_

            # Append the product in the list.
            partial_k.append((t - ti[l]) / (ti[k] - ti[l]))
        # _end_for_

        # Add the partial sum scaled with the function
        # value 'xk'.
        if k == 0:
            poly_func = xi[k] * sym.prod(partial_k)
        else:
            poly_func += xi[k] * sym.prod(partial_k)
        # _end_if_

    # _end_for_

    #  Sanity check.
    if poly_func is None:
        raise RuntimeError(f" LagrangePolynomial({letter}, {order}):"
                           " Error while computing the polynomial.")
    # _end_if_

    # Return all the symbolic variables.
    return poly_func, t, xi, ti
# _end_def_

def get_local_polynomials():
    """
    Generates one local polynomial function for the marginal means m(t)
    and one for the marginal variance function s(t). These are computed
    with LagrangePolynomial function, and they are returned as lambdas.

    These are used to reconstruct the final mean and variance functions
    after the minimization of the free energy has occurred. Since the
    order of the polynomials does not change the final expressions will
    be the same for each dimension. So we can call repeatedly these m(t)
    and s(t) for each dimension to reconstruct the whole mean and vars
    function.

    Note:
        The signatures of the generated functions are as follows:

        1) m(t) -> m(t, h0, h1, h2, h3, m0, m1, m2, m3)
        2) s(t) -> s(t, c0, c1, c2, s0, s1, s2)

        where 't' is the time we want to evaluate the functions and:

            [h0, h1, h2, h3] -> [t+h0, t+h1, t+h2, t+h3]
            [c0, c1, c2] -> [t+c0, t+c1, t+c2]

        these are the fixed times of the support points:

            [m0, m1, m2, m3] -> [m(t+h0), m(t+h1), m(t+h2), m(t+h3)]
            [s0, s1, s2] -> [s(t+c0), s(t+c1), s(t+c2)]

        The support points are the optimized variables of the mean
        field algorithm.

    :return: the local polynomials m(t) and s(t) as lambda functions.
    """

    # Create the mean / variance polynomials (with fixed orders).
    mt, t, mk, t_h = LagrangePolynomial(letter="m", order=3, fp="h")
    st, t, sk, t_c = LagrangePolynomial(letter="s", order=2, fp="c")

    # Generate a lambda function for m(t) and s(t).
    local_mt = sym.lambdify([t, *flatten([t_h, mk])], mt,
                            modules=["scipy", "numpy"], cse=True)

    local_st = sym.lambdify([t, *flatten([t_c, sk])], st,
                            modules=["scipy", "numpy"], cse=True)
    # Return the functions.
    return local_mt, local_st
# _end_def_
