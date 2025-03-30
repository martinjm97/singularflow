import math
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import uniform
import numpy as np


def get_average_value(integrand, s, samples, print_variance=False):
    """Get the average value of integrand(s, samples) at the given samples."""
    integrand_at_samples = jax.vmap(lambda samp: integrand(s, samp))(samples)
    # if print_variance:
    #     print(jnp.var(integrand_at_samples))
    return jnp.mean(integrand_at_samples)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def singular_integrate(
    numer: Callable[[jax.Array, jax.Array], jax.Array],
    pow: int,
    bounds: tuple[float, float],
    key: jax.Array,
    num_samples: int,
    theta: jax.Array,
    s: float,
) -> float:
    """Calculate the singular integral of numer(x, theta) / (x - s)^pow over the interval bounds using num_samps samples.

    There are two cases:
    (1) The derivative with respect to parameters, theta, in numer.
    (2) The derivative with respect to s.

    The solution is as follows:
    For (1) the derivative with respect to theta is the singular integral of the derivative of numer with respect to theta
    For (2), the derivative with respect to s is pow times the singular integral at pow + 1
    """
    symm_samples, rest_samples = get_samples(bounds, key, s, num_samples)
    return _singular_integrate(numer, pow, bounds, symm_samples, rest_samples, theta, s)


def singular_integrate_fwd(numer, pow, bounds, key, num_samples, theta, s):
    symm_samples, rest_samples = get_samples(bounds, key, s, num_samples)
    primal = _singular_integrate(numer, pow, bounds, symm_samples, rest_samples, theta, s)
    return primal, (symm_samples, rest_samples, theta, s)


def singular_integrate_bwd(numer, pow, bounds, key, num_samples, fwd_deriv_vals, g):
    symm_samples, rest_samples, theta, s = fwd_deriv_vals

    # Derivative with respect to theta
    deriv = jax.vjp(lambda t: _singular_integrate(numer, pow, bounds, symm_samples, rest_samples, t, s), theta)
    vjp_theta = deriv[1](g)[0]

    # Derivative with respect to s
    vjp_s = pow * _singular_integrate(numer, pow + 1, bounds, symm_samples, rest_samples, theta, s) * g

    return (vjp_theta, vjp_s)


singular_integrate.defvjp(singular_integrate_fwd, singular_integrate_bwd)


def get_samples(bounds: tuple[float, float], key: jax.Array, s: float, num_samples: int) -> tuple[jax.Array, jax.Array]:
    """Get symmetric and non-symmetric samples around the singular point s."""
    a, b = bounds
    assert a < b, f"Lower bound {a} must be less than upper bound {b}."
    symm_middle, symm_upper, rest_lower, rest_upper = jnp.where(
        s - a > b - s, jnp.array([s, b, a, 2 * s - b]), jnp.array([s, 2 * s - a, 2 * s - a, b])
    )

    symm_samples = uniform(key, (num_samples // 4,), minval=symm_middle, maxval=symm_upper)
    rest_samples = uniform(key, (num_samples // 2,), minval=rest_lower, maxval=rest_upper)

    return (symm_samples, rest_samples)


def _singular_integrate(
    numer: Callable[[jax.Array, jax.Array], jax.Array],
    pow: int,
    bounds: tuple[float, float],
    symm_samples: jax.Array,
    rest_samples: jax.Array,
    theta: jax.Array,
    s: float,
) -> float:
    """
    Integrate the singular integral of the form
    int_a^b frac{numer(x, theta)}{(x-s)^pow} dx
    where pow is a positive integer.
    """
    a, b = bounds
    assert a < b, f"Lower bound {a} must be less than upper bound {b}."
    assert int(pow) == pow and pow > 0, f"Power {pow} must be greater than 0."
    symm_middle, symm_upper, rest_lower, rest_upper = jnp.where(
        s - a > b - s, jnp.array([s, b, a, 2 * s - b]), jnp.array([s, 2 * s - a, 2 * s - a, b])
    )

    integrand = lambda s, x: numer(x, theta) / (x - s) ** pow

    match pow:
        case 1:
            # let h(x) = f(x) / (x - s)
            # C int_a^b h(x) dx = int_a^s (f(x) - f(2s - x)) / (x - s) dx + int_{2s - a}^b h(x) dx  (by Proposition 3.6)
            #                   = int_s^{2s - a} (f(x) - f(2s - x)) / (x - s) dx + int_{2s - a}^b h(x) dx  (by symmetry/change of variables)
            # In the following code, symm_int estimates term 1 above and rest_int estimates term 2 above.

            # To understand the estimator for term 1, we can rewrite the integral as:
            # int_s^{2s - a} (f(x) - f(2s - x)) / (x - s) dx
            #  = int_s^{2s - a} f(x) / (x - s) - f(2s - x) / (x - s) dx
            #  = int_s^{2s - a} f(x) / (x - s) + f(2s - x) / ((2s - x) - s) dx
            #  = int_s^{2s - a} h(x) + h(2s - x) dx
            #  = 2 int_s^{2s - a} (h(x) + h(2s - x)) / 2 dx
            # The Monte Carlo estimator for this integral is:
            #  = 2 * ((2s - a) - s) * (h(v) + h(2s - v)) / 2    where v ~ U(s, 2s - a)
            #  = 2 * (s - a) * average(h([v, 2s - v]))
            # This approach allows us to vectorize integrand evaluation over nsymm_samples
            nsymm_samples = jnp.concatenate([symm_samples, 2 * s - symm_samples])
            symm_int = 2 * (symm_upper - symm_middle) * get_average_value(integrand, s, nsymm_samples)

            # We now build a Monte Carlo estimator for term 2: int_{2s - a}^b h(x) dx.
            # We do this directly:
            # (b - (2s - a)) h(v) where v ~ U(2s - a, b)
            rest_int = (rest_upper - rest_lower) * get_average_value(integrand, s, rest_samples)
            return symm_int + rest_int
        case _:
            total = 0.0
            for i in reversed(range(1, pow)):  # NOTE: this could potentially be vectorized for perf
                num_a = numer(jnp.array([a], dtype=jnp.float32), theta) / (a - s) ** i
                num_b = numer(jnp.array([b], dtype=jnp.float32), theta) / (b - s) ** i
                # if num_a and num_b are a 1d array with a single element, then extract the scalar, otherwise, keep the float
                total -= math.factorial(i - 1) * (jnp.squeeze(num_b) - jnp.squeeze(num_a))
                numer = jax.jacfwd(numer)
            total += _singular_integrate(numer, 1, bounds, symm_samples, rest_samples, theta, s)
            return total / math.factorial(pow - 1)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    numer = lambda x, theta: jnp.sum(theta * x)
    pow = 2
    bounds = (0, 1)
    theta = jnp.array([1.0])
    s = 0.5
    num_samples = 1000

    # C int_0^1 theta * x / (x - s) dx
    # at s= 0.5, theta=1
    # = C int_0^1 x / (x - 0.5) dx
    # = C int_0^1 (x - 0.5 + 0.5) / (x - 0.5) dx
    # = 1 + C int_0^1 0.5 / (x - 0.5) dx
    # = 1 + 0.5 C int_0^1 1 / (x - 0.5) dx
    # = 1 + 0.5 (log|x - 0.5||_x=0^1)
    # = 1 + 0.5 (log|1 - 0.5| - log|0 - 0.5|)
    # = 1 + 0.5 (log|0.5| - log|0.5|)
    # = 1
    print(
        singular_integrate(numer, 1, bounds, key, num_samples, theta, s),
        "should be 1.0",
    )

    # H int_0^1 theta * x / (x - s)^2 dx
    # at s= 0.5, theta=1
    # = H int_0^1 x / (x - 0.5)^2 dx
    # = C int_0^1 1 / (x - 0.5) dx - (x/(x - 0.5))|_x=0^1
    # = - (x/(x - 0.5))|_x=0^1
    # = - (1/(1 - 0.5) - 0/(0 - 0.5))
    # = - 1 / 0.5 + 0
    # = -2
    print(
        singular_integrate(numer, pow, bounds, key, num_samples, theta, s),
        "should be -2.0",
    )

    # D_theta H int_0^1 theta * x / (x - s)^2 dx
    # = H int_0^1 x / (x - 0.5)^2 dx
    # = -2
    print(
        jax.grad(singular_integrate, argnums=5)(numer, pow, bounds, key, num_samples, theta, s),
        "should be [-2.]",
    )

    # D_s H int_0^1 theta * x / (x - s)^2 dx
    # = 2 * H int_0^1 theta * x / (x - s)^3 dx
    # by Prop. 3.10 at s=0.5, theta=1
    # = 2 (1/2 C int_0^1 1 / (x - 0.5) dx - 1/2 (1 / (x - 0.5))|_x=0^1 - 1/2 (x / (x - 0.5)^2)|_x=0^1
    # = - (1 / (1 - 0.5) - 1 / (0 - 0.5)) - (1 / (1 - 0.5)^2 - 0 / (0 - 0.5)^2)
    # = - (2 + 2) - (4 - 0)
    # = -8

    # For gradient with respect to s (position 6)
    print(
        jax.grad(singular_integrate, argnums=6)(numer, pow, bounds, key, num_samples, theta, s),
        "should be -8.0",
    )

    # Test multi-dimensional theta
    theta = jnp.array([1.0, 2.0])
    num_samples = 10000
    numer = lambda x, theta: theta[0] * x + theta[1]
    pow = 1
    bounds = (0, 1)
    s = 0.1

    # C int_0^1 (theta[0]x + theta[1]) / (x - 0.1) dx

    # grad_theta C int_0^1 (theta[0]x + theta[1]) / (x - 0.1) dx
    # = (C int_0^1 x / (x - 0.1) dx,  C int_0^1 1 / (x - 0.1) dx)
    # = (C int_0^1 (x - 0.1) / (x - 0.1) dx + C int_0^1 0.1 / (x - 0.1) dx, C int_0^1 1 / (x - 0.1) dx)
    # = (1 + 0.1 * 2.1972, 2.1972) = (1.219, 2.1972)

    print(
        jax.grad(singular_integrate, argnums=5)(numer, pow, bounds, key, num_samples, theta, s),
        "should be [1.219, 2.1972]",
    )
