import sympy as sym


def LagrangePolynomial(letter: str = "m", order: int = 3):
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

    EXAMPLE(s):
        # Here we define the functions.
        mt = LagrangePolynomial(letter="m", order=3)
        st = LagrangePolynomial(letter="s", order=2)

        # Here we take the derivatives w.r.t. time 't'.
        dm_dt = mt.diff(t)
        ds_dt = st.diff(t)

        # Here we take the derivatives w.r.t. the first
        # points m0, s0.
        dm_dm0 = mt.diff(xi[0])
        ds_ds0 = st.diff(xi[0])

        etc.

    :return: the Lagrange polynomial in symbolic notation.
    """

    # Define the time variable 't' as symbol.
    t = sym.Symbol("t")

    # Define the fixed time points 'tk' as symbols.
    ti = sym.symbols(f"t:{order + 1}")

    # Define the fixed function points 'xk' as symbols.
    xi = sym.symbols(f"{letter}:{order + 1}")

    # Initialize the return polynomial.
    poly_func = sym.Symbol('')

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
        poly_func += (xi[k] * sym.prod(partial_k))

    # _end_for_

    return poly_func
# _end_def_
