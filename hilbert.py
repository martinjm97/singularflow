import time
from tap import Tap

import pandas as pd

# Show all columns
pd.set_option("display.max_columns", None)

import jax
import jax.numpy as jnp
from jax import random
from jax.random import uniform

from singular_integrate import singular_integrate, get_average_value, get_samples, _singular_integrate

from utils import round_sigfigs, latexify


def hilbert_transform(numer, key, num_samps):
    """Compute the Hilbert transform of f."""

    def hilbert_fun(y):
        return -(1 / jnp.pi) * singular_integrate(numer, 1, (-1, 1), key, num_samps, jnp.array([0.0]), y)

    return hilbert_fun


def test_hilbert(
    funs,
    exprs,
    ground_truth,
    num_samps: int = 10000,
    bounds: tuple[int, int] = (-1, 1),
    verbose: bool = False,
    suppress: bool = False,
):
    if verbose:
        print("Function", "Hilbert Transform", "Math Approx.")
    htfs_means, htfs_stds = [], []
    jit_funs = [(lambda fun: lambda key: hilbert_transform(fun, key, num_samps))(fun) for fun in funs]
    for fun, expr, truth in zip(jit_funs, exprs, ground_truth):
        htfs = []
        for i in range(10):
            key = random.PRNGKey(i)
            hf_at_half = fun(key)(0.5)
            if verbose:
                print(expr, hf_at_half, truth)
            if not suppress:
                assert jnp.isclose(hf_at_half, truth, atol=1e-2)
            htfs.append(hf_at_half)
        htfs = jnp.array(htfs)
        htfs_means.append(jnp.mean(htfs).item())
        htfs_stds.append(jnp.std(htfs).item())
    return htfs_means, htfs_stds


def standard_hilbert(funs, exprs, num_samps: int = 10000):
    means, stds = [], []
    bounds = (-1, 1)
    for fun, expr in zip(funs, exprs):
        standard, htfs, dhtfs = [], [], []
        for i in range(10):
            key = random.PRNGKey(i)
            samples = uniform(key, (num_samps,), minval=bounds[0], maxval=bounds[1])
            hilbert_at_half = -1 / jnp.pi * (bounds[1] - bounds[0]) * get_average_value(fun, 0.5, samples)
            standard.append(hilbert_at_half.item())
        standard = jnp.array(standard)
        # print("Hilbert transform standard mean and std", expr, jnp.mean(standard), jnp.std(standard))
        means.append(jnp.mean(standard).item())
        stds.append(jnp.std(standard).item())
    return means, stds


# This is actually the deriv of the standard hilbert
# def standard_hilbert_deriv(funs, exprs, num_samps: int = 10000):
#     means, stds = [], []
#     for fun, expr in zip(funs, exprs):
#         standard = []

#         for i in range(10):
#             key = random.PRNGKey(i)
#             samples = uniform(key, (num_samps,), minval=-1, maxval=1)

#             def f(s):
#                 return -1 / jnp.pi * mc_integrate(fun, s, samples)

#             deriv_at_half = jax.grad(f)(0.5)
#             print(f"Iteration {i}, standard deriv at half", deriv_at_half)

#             # def g(s):
#             #     return -1 / jnp.pi * mc_integrate(lambda s, x: fun(s, x) / (x - s), s, samples)

#             # print(f"Iteration {i}, standard deriv at half another way", g(0.5))

#             standard.append(deriv_at_half.item())

#         standard = jnp.array(standard)
#         means.append(jnp.mean(standard).item())
#         stds.append(jnp.std(standard).item())
#     return means, stds


def standard_hilbert_deriv(funs_div, exprs, num_samps: int = 10000):
    shd = []
    key = random.PRNGKey(0)
    bounds = (-1, 1)
    # for fun, expr in zip(funs_div, exprs):
    #     samples = uniform(key, (num_samps,), minval=-1, maxval=1)

    #     def g(s):
    #         return -1 / jnp.pi * mc_integrate(fun, s, samples)

    #     shd.append(jax.grad(g)(0.5).item())
    for fun, expr in zip(funs_div, exprs):
        samples = uniform(key, (num_samps,), minval=bounds[0], maxval=bounds[1])

        def g(s):
            return (
                -1 / jnp.pi * (bounds[1] - bounds[0]) * get_average_value(lambda s, x: fun(s, x) / (x - s), s, samples)
            )

        shd.append(g(0.5).item())

    return shd


def pv_of_deriv(funs, exprs, num_samps: int = 10000):
    derivs_at_half = []
    for fun, expr in zip(funs, exprs):

        def pv(s):
            key = random.PRNGKey(0)
            si = singular_integrate((lambda x, theta: fun(x, 0.0) / (x - theta)), 1, (-1, 1), key, num_samps, s, s)
            return -1 / jnp.pi * si

        deriv_at_half = pv(0.5)
        derivs_at_half.append(deriv_at_half.item())
    return derivs_at_half


def deriv_pv(funs, exprs, num_samps: int = 10000):
    # d/ds (-(1 / jnp.pi) * pv int_-1^1 f(x) / (x - s) dx)
    # = -(1 / jnp.pi) * d/ds (pv int_-1^1 f(x) / (x - s) dx)
    # s - a > b - s, symm_middle, symm_upper, rest_lower, rest_upper = [s, b, a, 2 * s - b]
    #

    # -(1 / jnp.pi) * d/ds(int_-1^1 f(x) / (x - s) dx)
    # d/ds (pv int_-1^1 f(x) / (x - s) dx)
    # = d/ds (pv int_-1^1 f(x) / (x - s) dx)

    # C int_-1^1 f(x) / (x - s)^2 dx != H int_-1^1 f(x) / (s - x)^2 dx
    dpv = []
    for fun, expr in zip(funs, exprs):

        # def pv(s):
        #     key = random.PRNGKey(0)
        #     bounds = (-1, 1)
        #     symm_samples, rest_samples = get_samples(bounds, key, s, num_samps)
        #     return -1 / jnp.pi * _singular_integrate(fun, 1, bounds, symm_samples, rest_samples, 0.0, s)

        def pv(s):
            key = random.PRNGKey(0)
            bounds = (-1, 1)
            symm_samples, rest_samples = get_samples(bounds, key, s, num_samps)
            a, b = bounds
            symm_middle, symm_upper, rest_lower, rest_upper = jnp.where(
                s - a > b - s, jnp.array([s, b, a, 2 * s - b]), jnp.array([s, 2 * s - a, 2 * s - a, b])
            )
            integrand = lambda s, x: fun(x, 0.0) / (x - s)
            symm_samples = jnp.concatenate([symm_samples, 2 * s - symm_samples])
            symm_int = 2 * (symm_upper - symm_middle) * get_average_value(integrand, s, symm_samples)
            rest_int = (rest_upper - rest_lower) * get_average_value(integrand, s, rest_samples)
            return -1 / jnp.pi * (symm_int + rest_int)

        deriv_at_half = jax.grad(pv)(0.5)
        dpv.append(deriv_at_half.item())

    return dpv


def test_deriv_hilbert(funs, exprs, deriv_ground_truth, num_samps: int = 10000, verbose: bool = False):
    if verbose:
        print("Function", "Deriv Hilbert Transform", "Math Approx.")
    grad_funs = [(lambda fun: lambda key: jax.grad(hilbert_transform(fun, key, num_samps)))(fun) for fun in funs]
    our_deriv_means, our_deriv_stds = [], []
    for grad_fun, expr, truth in zip(grad_funs, exprs, deriv_ground_truth):
        our_deriv_runs = []
        for i in range(10):
            key = random.PRNGKey(i)
            deriv_at_half = grad_fun(key)(0.5).item()
            if verbose:
                print(expr, deriv_at_half, truth)
            our_deriv_runs.append(deriv_at_half)
            # if verbose:
            #     print(expr, deriv_at_half, truth)
            # assert (jnp.isclose(deriv_at_half, truth, atol=1e-2)), f"On {expr} expected {truth} but got {deriv_at_half}"
        our_deriv_runs = jnp.array(our_deriv_runs)
        our_deriv_means.append(jnp.mean(our_deriv_runs).item())
        our_deriv_stds.append(jnp.std(our_deriv_runs).item())
    return our_deriv_means, our_deriv_stds


def baseline_funs(funs, key, num_samps: int = 10000):
    baseline_funs = []
    bounds = (-1, 1)
    for fun in funs:

        def closure(fun):
            def baseline_fun(s):
                samples = uniform(key, (num_samps,), minval=bounds[0], maxval=bounds[1])
                funsx = lambda s, x: fun(x, 0.0) / (x - s)
                hilbert_at_half = -1 / jnp.pi * (bounds[1] - bounds[0]) * get_average_value(funsx, s, samples)
                return hilbert_at_half

            return baseline_fun

        baseline_funs.append(closure(fun))
    return baseline_funs


def hilbert_timing_baseline(funs, key, num_samps: int = 10000, use_jit: bool = False):
    times = []
    if use_jit:
        compiled_funs = [jax.jit(baseline_fun) for baseline_fun in baseline_funs(funs, key, num_samps)]
    else:
        compiled_funs = baseline_funs(funs, key, num_samps)
    for i in range(26):
        start = time.time()
        for fun in compiled_funs:
            hilbert_at_half = fun(0.5)
        end = time.time()
        times.append(end - start)
    times = times[1:]
    return jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item()


def deriv_hilbert_timing_baseline(funs, key, num_samps: int = 10000, use_jit: bool = False):
    times = []
    if use_jit:
        compiled_funs = [jax.jit(jax.grad(baseline_fun)) for baseline_fun in baseline_funs(funs, key, num_samps)]
    else:
        compiled_funs = [jax.grad(baseline_fun) for baseline_fun in baseline_funs(funs, key, num_samps)]
    for i in range(26):
        start = time.time()
        for fun in compiled_funs:
            deriv_at_half = fun(0.5)
        end = time.time()
        times.append(end - start)
    times = times[1:]
    return jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item()


def hilbert_timing(funs, key, num_samps: int = 10000, use_jit: bool = False):
    times = []
    if use_jit:
        hfs = [jax.jit(hilbert_transform(fun, key, num_samps)) for fun in funs]
    else:
        hfs = [hilbert_transform(fun, key, num_samps) for fun in funs]
    for i in range(26):
        start = time.time()
        for hf in hfs:
            deriv_at_half = hf(0.5).item()
        end = time.time()
        times.append(end - start)
    times = times[1:]
    return jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item()


def deriv_hilbert_timing(funs, key, num_samps: int = 10000, use_jit: bool = False):
    times = []
    if use_jit:
        grad_funs = [jax.jit(jax.grad(hilbert_transform(fun, key, num_samps))) for fun in funs]
    else:
        grad_funs = [jax.grad(hilbert_transform(fun, key, num_samps)) for fun in funs]

    for i in range(26):
        start = time.time()
        for grad_fun in grad_funs:
            deriv_at_half = grad_fun(0.5)
        end = time.time()
        times.append(end - start)
    times = times[1:]
    return jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item()


def hilbert_timing_baseline_each_benchmark(funs, key, num_samps: int = 10000, use_jit: bool = True):
    all_times = []
    for fun in baseline_funs(funs, key, num_samps):
        times = []

        f = jax.jit(fun) if use_jit else fun
        for i in range(26):
            start = time.time()
            deriv_at_half = f(0.5).block_until_ready()
            end = time.time()
            times.append(end - start)
        # drop the first to get rid of the warmup time
        times = times[1:]
        all_times.append(
            (jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item())
        )
    return all_times


def deriv_hilbert_timing_baseline_each_benchmark(funs, key, num_samps: int = 10000, use_jit: bool = True):
    all_times = []
    for fun in baseline_funs(funs, key, num_samps):
        times = []
        f = jax.grad(fun)
        if use_jit:
            f = jax.jit(f)
        for i in range(26):
            start = time.time()
            deriv_at_half = f(0.5).block_until_ready()
            end = time.time()
            times.append(end - start)
        # drop the first to get rid of the warmup time
        times = times[1:]
        all_times.append(
            (jnp.median(jnp.array(times)).item(), jnp.mean(jnp.array(times)).item(), jnp.std(jnp.array(times)).item())
        )
    return all_times


def hilbert_timing_each_benchmark(funs, key, num_samps: int = 10000, use_jit: bool = True):
    hfs = [hilbert_transform(fun, key, num_samps) for fun in funs]
    if use_jit:
        hfs = [jax.jit(hf) for hf in hfs]

    all_times = []
    for hf in hfs:
        times = []
        for i in range(26):
            start = time.time()
            deriv_at_half = hf(0.5).block_until_ready()
            end = time.time()
            times.append(end - start)
        # drop the first to get rid of the warmup time
        times = times[1:]
        median, mean, std = (
            jnp.median(jnp.array(times)).item(),
            jnp.mean(jnp.array(times)).item(),
            jnp.std(jnp.array(times)).item(),
        )
        all_times.append((median, mean, std))
    return all_times


def deriv_hilbert_timing_each_benchmark(funs, key, num_samps: int = 10000, use_jit: bool = True):
    all_times = []
    for fun in funs:
        times = []
        hilbert_transform_deriv = jax.grad(hilbert_transform(fun, key, num_samps))
        if use_jit:
            hilbert_transform_deriv = jax.jit(hilbert_transform_deriv)
        for i in range(26):
            start = time.time()
            deriv_at_half = hilbert_transform_deriv(0.5).block_until_ready()
            end = time.time()
            times.append(end - start)
        # drop the first to get rid of the warmup time
        times = times[1:]
        median, mean, std = (
            jnp.median(jnp.array(times)).item(),
            jnp.mean(jnp.array(times)).item(),
            jnp.std(jnp.array(times)).item(),
        )
        all_times.append((median, mean, std))
    return all_times


class Args(Tap):
    seed: int = 0
    table1: bool = False
    table2: bool = False
    table3: bool = False
    appx_computation_time: bool = False
    num_samples: int = 10000


def table1(exprs, funs, funs_divs, args: Args):
    ground_truth = [-0.462, -0.231, -0.291, -0.894, -0.409, 0.458]
    ours_mean, ours_std = test_hilbert(funs, exprs, ground_truth, args.num_samples, verbose=False)
    latex = False
    p = 1
    print()

    standard_mean, standard_std = standard_hilbert(funs_divs, exprs, num_samps=args.num_samples)

    df = pd.DataFrame(
        {
            "f(u)": latexify(exprs) if latex else exprs,
            "Ground Truth": round_sigfigs(ground_truth, p + 1, latex),
            "Ours": round_sigfigs(ours_mean, p, latex),
            "Ours Std": round_sigfigs(ours_std, p, latex),
            "Standard": round_sigfigs(standard_mean, p, latex),
            "Standard Std": round_sigfigs(standard_std, p, latex),
            "Mathematica": ["-0.46", "-0.23", "-0.29", "-0.89", "-0.41", "0.46"],
        }
    )
    print(df)
    # print(df.to_latex(index=False, escape=False))


def table2(exprs, funs, funs_divs, args: Args):
    deriv_ground_truth = [0.774, -0.074714029, 1.51771, 0.467989, 0.815591, 0.8675]
    ours_deriv, ours_deriv_std = test_deriv_hilbert(funs, exprs, deriv_ground_truth, args.num_samples, verbose=False)

    pv_of_deriv_val = pv_of_deriv(funs, exprs, args.num_samples)
    standard_deriv = standard_hilbert_deriv(funs_divs, exprs, args.num_samples)

    # pv_of_deriv_val = deriv_pv(funs, exprs, args.num_samples)

    latex = False
    p = 1
    df = pd.DataFrame(
        {
            "f(u)": latexify(exprs) if latex else exprs,
            "Ground Truth": round_sigfigs(deriv_ground_truth, p + 1, latex),
            "Ours": round_sigfigs(ours_deriv, p, latex),
            "Ours Std": round_sigfigs(ours_deriv_std, p, latex),
            "PV": round_sigfigs(pv_of_deriv_val, p, latex),
            "Standard": round_sigfigs(standard_deriv, p, latex),
            "Mathematica": ["0.77", "-0.0747", "-", "-", "0.82", "0.87"],
        }
    )
    print(df)
    # print(df.to_latex(index=False, escape=False))


def table3(exprs, funs, key, args: Args):
    num_samples = args.num_samples

    # Get times for all of the benchmarks
    use_jit = False
    hilbert_baseline = hilbert_timing_baseline(funs, key, num_samples, use_jit)
    hilbert = hilbert_timing(funs, key, num_samples, use_jit)
    dhilbert_baseline = deriv_hilbert_timing_baseline(funs, key, num_samples, use_jit)
    dhilbert = deriv_hilbert_timing(funs, key, num_samples, use_jit)

    use_jit = True
    hilbert_baseline_jit = hilbert_timing_baseline(funs, key, num_samples, use_jit)
    hilbert_jit = hilbert_timing(funs, key, num_samples, use_jit)
    dhilbert_baseline_jit = deriv_hilbert_timing_baseline(funs, key, num_samples, use_jit)
    dhilbert_jit = deriv_hilbert_timing(funs, key, num_samples, use_jit)

    latex = False
    p = 1
    scale = 1000  # Convert from seconds to milliseconds
    benchmark_data = [
        {
            "Benchmark": "Hilbert Transform",
            "Baseline": round_sigfigs(scale * hilbert_baseline[1], p, latex),
            "Baseline Std": round_sigfigs(scale * hilbert_baseline[2], p, latex),
            "Ours": round_sigfigs(scale * hilbert[1], p, latex),
            "Ours Std": round_sigfigs(scale * hilbert[2], p, latex),
            "Baseline (JIT)": round_sigfigs(scale * hilbert_baseline_jit[1], p, latex),
            "Baseline Std (JIT)": round_sigfigs(scale * hilbert_baseline_jit[2], p, latex),
            "Ours (JIT)": round_sigfigs(scale * hilbert_jit[1], p, latex),
            "Ours Std (JIT)": round_sigfigs(scale * hilbert_jit[2], p, latex),
        },
        {
            "Benchmark": "Deriv Hilbert Transform",
            "Baseline": round_sigfigs(scale * dhilbert_baseline[1], p, latex),
            "Baseline Std": round_sigfigs(scale * dhilbert_baseline[2], p, latex),
            "Ours": round_sigfigs(scale * dhilbert[1], p, latex),
            "Ours Std": round_sigfigs(scale * dhilbert[2], p, latex),
            "Baseline (JIT)": round_sigfigs(scale * dhilbert_baseline_jit[1], p, latex),
            "Baseline Std (JIT)": round_sigfigs(scale * dhilbert_baseline_jit[2], p, latex),
            "Ours (JIT)": round_sigfigs(scale * dhilbert_jit[1], p, latex),
            "Ours Std (JIT)": round_sigfigs(scale * dhilbert_jit[2], p, latex),
        },
    ]
    df = pd.DataFrame(benchmark_data)

    print("Timing for all benchmarks in milliseconds.")
    print(df)


def appx_computation_time(exprs, funs, key, args: Args):
    num_samples = args.num_samples

    # Get times for each benchmark
    use_jit = False

    hilbert_baseline = hilbert_timing_baseline_each_benchmark(funs, key, num_samples, use_jit)
    hilbert = hilbert_timing_each_benchmark(funs, key, num_samples, use_jit)

    dhilbert = deriv_hilbert_timing_each_benchmark(funs, key, num_samples, use_jit)
    dhilbert_baseline = deriv_hilbert_timing_baseline_each_benchmark(funs, key, num_samples, use_jit)

    use_jit = True

    hilbert_baseline_jit = hilbert_timing_baseline_each_benchmark(funs, key, num_samples, use_jit)
    hilbert_jit = hilbert_timing_each_benchmark(funs, key, num_samples, use_jit)

    dhilbert_jit = deriv_hilbert_timing_each_benchmark(funs, key, num_samples, use_jit)
    dhilbert_baseline_jit = deriv_hilbert_timing_baseline_each_benchmark(funs, key, num_samples, use_jit)

    latex = False

    print("Timing for each benchmark in milliseconds.")
    print()
    scale = 1000
    exprs_name = "$f(u)$" if latex else "f(u)"
    data = {
        exprs_name: exprs,
        "Baseline": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in hilbert_baseline],
        "Ours": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in hilbert],
        "Deriv. Baseline": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in dhilbert_baseline],
        "Deriv Ours": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in dhilbert],
        "Baseline (JIT)": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in hilbert_baseline_jit],
        "Ours (JIT)": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in hilbert_jit],
        "Deriv. Baseline (JIT)": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in dhilbert_baseline_jit],
        "Deriv Ours (JIT)": [f"{median*scale:.2f} ± {std*scale:.2f}" for median, _, std in dhilbert_jit],
    }
    df = pd.DataFrame(data)
    print(df)

    # print(df.to_latex(index=False, escape=False))


if __name__ == "__main__":
    args = Args().parse_args()
    exprs = ["u", "u^2", "e^u", "ue^u", "sin(u)", "cos(u)"]
    funs = [
        lambda x, theta: x,
        lambda x, theta: x**2,
        lambda x, theta: jnp.exp(x),
        lambda x, theta: x * jnp.exp(x),
        lambda x, theta: jnp.sin(x),
        lambda x, theta: jnp.cos(x),
    ]
    key = random.PRNGKey(args.seed)

    funs_divs = [
        lambda s, x: x / (x - s),
        lambda s, x: x**2 / (x - s),
        lambda s, x: jnp.exp(x) / (x - s),
        lambda s, x: x * jnp.exp(x) / (x - s),
        lambda s, x: jnp.sin(x) / (x - s),
        lambda s, x: jnp.cos(x) / (x - s),
    ]

    if args.table1:
        table1(exprs, funs, funs_divs, args)
        print()

    if args.table2:
        table2(exprs, funs, funs_divs, args)
        print()

    if args.table3:
        table3(exprs, funs, key, args)
        print()

    if args.appx_computation_time:
        appx_computation_time(exprs, funs, key, args)
        print()
