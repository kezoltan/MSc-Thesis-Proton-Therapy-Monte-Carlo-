import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _log2_positive(values):
    """Return log2(values), masking zero, negative and non-finite entries."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    result[valid] = np.log2(values[valid])
    return result


def _finite_argmax(values, description):
    """Return the largest finite entry and ignore inactive-output NaNs."""
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        raise ValueError(f"no finite {description} values found")
    return int(np.nanargmax(values))


def mlmc_plot(filename, nvert=3, error_bars=True, mode="all"):
    """
    Generate MLMC plots from the text output produced by mlmc_testv.

    mode can be:
      "all"              plot every output
      "worst_variance"   plot the output with the largest correction
                         variance on the finest convergence-test level
      "worst_kurtosis"   plot the output with the largest kurtosis on
                         the finest convergence-test level
      "primal"           plot only the first output
    """
    plt.close("all")
    print("Calling MLMC plot...")

    if nvert not in (1, 3):
        raise ValueError("nvert must be 1 or 3")

    if mode not in ("all", "worst_variance", "worst_kurtosis", "primal"):
        raise ValueError(
            "mode must be 'all', 'worst_variance', 'worst_kurtosis' or 'primal'"
        )

    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    # Read N from the convergence-test heading.
    N = 0
    for line in lines:
        if "*** using N" in line:
            match = re.search(r"N\s*=\s*(\d+)", line)
            if match:
                N = int(match.group(1))
            break

    # Read every convergence table.  This is the only substantive extension
    # to the original single-output parser.
    names = []
    convergence_data = []

    for i, line in enumerate(lines):
        if "ave(Pf-Pc)" not in line:
            continue

        name = "Node"
        for previous in reversed(lines[max(0, i - 6):i]):
            if "Convergence tests" in previous:
                match = re.search(r"\*\*\*\s*(.*?)\s*Convergence tests", previous)
                if match and match.group(1):
                    name = match.group(1).strip()
                break

        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("-")):
            j += 1

        table = []
        while j < len(lines):
            try:
                data = [float(value) for value in lines[j].split()]
            except ValueError:
                break

            if len(data) < 8:
                break

            table.append(data[:8])
            j += 1

        if table:
            names.append(name)
            convergence_data.append(np.asarray(table))

    if not convergence_data:
        raise ValueError("no convergence-test table found in " + filename)

    # Read the complexity table once; it is common to all outputs.
    Eps = []
    mlmc_cost = []
    std_cost = []
    Nls_rows = []
    num_q = len(convergence_data)

    for i, line in enumerate(lines):
        if "mlmc_cost" not in line or "std_cost" not in line:
            continue

        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("-")):
            j += 1

        while j < len(lines):
            stripped = lines[j].strip()
            if stripped.startswith("===") or stripped.startswith(">>>"):
                break

            # A failed weak-convergence warning can be printed immediately
            # before each numerical complexity row.  It is not table data.
            if not stripped or stripped.startswith("-") or stripped.startswith("***"):
                j += 1
                continue

            try:
                data = [float(value) for value in lines[j].split()]
            except ValueError:
                j += 1
                continue

            if len(data) < num_q + 5:
                j += 1
                continue

            Eps.append(data[0])
            mlmc_cost.append(data[num_q + 1])
            std_cost.append(data[num_q + 2])
            Nls_rows.append(data[num_q + 4:])
            j += 1
        break

    Eps = np.asarray(Eps)
    mlmc_cost = np.asarray(mlmc_cost)
    std_cost = np.asarray(std_cost)

    if Nls_rows:
        max_levels = max(len(row) for row in Nls_rows)
        Nls = np.full((max_levels, len(Nls_rows)), np.nan)
        for i, row in enumerate(Nls_rows):
            Nls[:len(row), i] = row
    else:
        Nls = np.empty((0, 0))

    if mode == "all":
        selected = range(len(convergence_data))
    elif mode == "worst_variance":
        selected = [_finite_argmax(
            [table[-1, 3] for table in convergence_data],
            "finest-level correction variance",
        )]
    elif mode == "worst_kurtosis":
        selected = [_finite_argmax(
            [table[-1, 5] for table in convergence_data],
            "finest-level kurtosis",
        )]
    else:
        selected = [0]

    base = os.path.splitext(filename)[0]

    for output in selected:
        table = convergence_data[output]
        del1 = table[:, 1]
        del2 = table[:, 2]
        var1 = table[:, 3]
        var2 = table[:, 4]
        kur1 = table[:, 5]
        cost = table[:, 7]

        with np.errstate(invalid="ignore"):
            vvr1 = var1**2 * (kur1 - 1.0)
        L = len(del1) - 1

        if nvert == 3:
            fig = plt.figure(figsize=(8, 12))
        else:
            fig = plt.figure(figsize=(10, 4))

        plt.subplot(nvert, 2, 1)
        plt.plot(range(L + 1), _log2_positive(var2), "-*", label=r"$P_\ell$")
        plt.plot(range(1, L + 1), _log2_positive(np.abs(var1[1:])), ":*",
                 label=r"$P_\ell-P_{\ell-1}$")
        plt.xlabel(r"level $\ell$")
        plt.ylabel(r"$\log_2$ variance")
        plt.xlim(0, L)
        plt.legend(loc="lower left")

        if error_bars and N > 0:
            lower = _log2_positive(np.maximum(
                np.abs(var1[1:]) - 3.0 * np.sqrt(np.maximum(vvr1[1:], 0.0) / N),
                1.0e-10,
            ))
            upper = _log2_positive(
                np.abs(var1[1:]) + 3.0 * np.sqrt(np.maximum(vvr1[1:], 0.0) / N)
            )
            for level, low, high in zip(range(1, L + 1), lower, upper):
                if np.isfinite(low) and np.isfinite(high):
                    plt.plot([level, level], [low, high], "-r.")

        plt.subplot(nvert, 2, 2)
        plt.plot(range(L + 1), _log2_positive(np.abs(del2)), "-*", label=r"$P_\ell$")
        plt.plot(range(1, L + 1), _log2_positive(np.abs(del1[1:])), ":*",
                 label=r"$P_\ell-P_{\ell-1}$")
        plt.xlabel(r"level $\ell$")
        plt.ylabel(r"$\log_2 |\mathrm{mean}|$")
        plt.xlim(0, L)
        plt.legend(loc="lower left")

        if error_bars and N > 0:
            standard_error = np.sqrt(np.maximum(var1[1:], 0.0) / N)
            lower = _log2_positive(np.maximum(
                np.abs(del1[1:]) - 3.0 * standard_error, 1.0e-10
            ))
            upper = _log2_positive(
                np.abs(del1[1:]) + 3.0 * standard_error
            )
            for level, low, high in zip(range(1, L + 1), lower, upper):
                if np.isfinite(low) and np.isfinite(high):
                    plt.plot([level, level], [low, high], "-r.")

        if nvert == 3:
            plt.subplot(3, 2, 3)
            plt.plot(range(L + 1), _log2_positive(cost), "--*")
            plt.xlabel(r"level $\ell$")
            plt.ylabel(r"$\log_2$ cost per sample")
            plt.xlim(0, L)

            plt.subplot(3, 2, 4)
            plt.plot(range(1, L + 1), kur1[1:], "--*")
            plt.xlabel(r"level $\ell$")
            plt.ylabel("kurtosis")
            plt.xlim(0, L)

            plt.subplot(3, 2, 5)
            if Nls.size:
                plt.semilogy(np.arange(Nls.shape[0]), Nls)
            plt.xlabel(r"level $\ell$")
            plt.ylabel(r"$N_\ell$")
            if len(Eps):
                plt.legend([str(eps) for eps in Eps], loc="upper right")

            plt.subplot(3, 2, 6)
            if len(Eps):
                plt.loglog(Eps, Eps**2 * std_cost, "-*", label="Std MC")
                plt.loglog(Eps, Eps**2 * mlmc_cost, ":*", label="MLMC")
            plt.xlabel(r"accuracy $\varepsilon$")
            plt.ylabel(r"$\varepsilon^2$ Cost")
            if len(Eps):
                plt.legend()

        fig.suptitle(names[output])
        fig.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", names[output]).strip("_").lower()
        if len(convergence_data) == 1 or mode == "primal":
            plot_filename = base + ".png"
        elif mode == "all":
            plot_filename = f"{base}_{safe_name}.png"
        else:
            plot_filename = f"{base}_{mode}_{safe_name}.png"

        fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_filename}")

    # With nvert=1, retain the original second figure for the common
    # allocation and complexity plots.
    if nvert == 1:
        fig = plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        if Nls.size:
            plt.semilogy(np.arange(Nls.shape[0]), Nls)
        plt.xlabel(r"level $\ell$")
        plt.ylabel(r"$N_\ell$")
        if len(Eps):
            plt.legend([str(eps) for eps in Eps], loc="upper right")

        plt.subplot(1, 2, 2)
        if len(Eps):
            plt.loglog(Eps, Eps**2 * std_cost, "-*", label="Std MC")
            plt.loglog(Eps, Eps**2 * mlmc_cost, ":*", label="MLMC")
        plt.xlabel(r"accuracy $\varepsilon$")
        plt.ylabel(r"$\varepsilon^2$ Cost")
        if len(Eps):
            plt.legend()

        fig.tight_layout()
        plot_filename = base + "_complexity.png"
        fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 mlmc_plot.py <filename> [nvert] [mode]")
    else:
        filename = sys.argv[1]
        nvert = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        mode = sys.argv[3] if len(sys.argv) > 3 else "all"
        mlmc_plot(filename, nvert=nvert, mode=mode)
