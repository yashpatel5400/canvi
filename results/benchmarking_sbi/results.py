from pathlib import Path

import pandas as pd  # noqa
from omegaconf import OmegaConf

from utils import apply_df, clean, compile_df, get_df

basepath_runs = Path(__file__).parent.absolute() / "runs"
basepath = Path(__file__).parent.absolute() / "results"


def main_paper(row):
    """Main paper"""
    if row["algorithm"] == "REJ-ABC 100 KDE":
        row["algorithm"] = "REJ-ABC"

    if row["algorithm"] == "SMC-ABC (ours) 100 .5 .2 KDE" and (
        row["num_simulations"] == "10³" or row["num_simulations"] == "10⁴"
    ):
        row["algorithm"] = "SMC-ABC"
    if (
        row["algorithm"] == "SMC-ABC (ours) 1000 .5 .2 KDE"
        and row["num_simulations"] == "10⁵"
    ):
        row["algorithm"] = "SMC-ABC"

    if row["algorithm"] == "NLE-MAF":
        row["algorithm"] = "NLE"

    if row["algorithm"] == "NPE-NSF":
        row["algorithm"] = "NPE"

    if row["algorithm"] == "NRE-RES":
        row["algorithm"] = "NRE"

    if row["algorithm"] == "SNLE-MAF":
        row["algorithm"] = "SNLE"

    if row["algorithm"] == "SNPE-NSF":
        row["algorithm"] = "SNPE"

    if row["algorithm"] == "SNRE-RES":
        row["algorithm"] = "SNRE"

    if row["algorithm"] not in [
        "REJ-ABC",
        "SMC-ABC",
        "NLE",
        "NPE",
        "NRE",
        "SNLE",
        "SNPE",
        "SNRE",
    ]:
        return None

    return row


def abc_lra_sass(row):
    """ABC LRA and SASS results"""
    if row["algorithm"] == "REJ-ABC 100 KDE":
        row["algorithm"] = "REJ-ABC"

    if row["algorithm"] == "REJ-ABC 100 LRA KDE":
        row["algorithm"] = "REJ-ABC+LRA"

    if row["algorithm"] == "REJ-ABC 100 SASS KDE":
        row["algorithm"] = "REJ-ABC+SASS"

    if row["algorithm"] == "SMC-ABC (ours) 100 .5 .2 KDE" and (
        row["num_simulations"] == "10³" or row["num_simulations"] == "10⁴"
    ):
        row["algorithm"] = "SMC-ABC"
    if (
        row["algorithm"] == "SMC-ABC (ours) 1000 .5 .2 KDE"
        and row["num_simulations"] == "10⁵"
    ):
        row["algorithm"] = "SMC-ABC"

    if row["algorithm"] == "SMC-ABC (ours) .5 .2 LRA KDE":
        row["algorithm"] = "SMC-ABC+LRA"

    if row["algorithm"] == "SMC-ABC (ours) .5 .2 SASS KDE":
        row["algorithm"] = "SMC-ABC+SASS"

    if row["algorithm"] not in [
        "REJ-ABC",
        "REJ-ABC+LRA",
        "REJ-ABC+SASS",
        "SMC-ABC",
        "SMC-ABC+LRA",
        "SMC-ABC+SASS",
    ]:
        return None

    return row


def rf_abc(row):
    """RF-ABC versus REJ-ABC, SMC-ABC"""
    if row["algorithm"] == "REJ-ABC 100 KDE":
        row["algorithm"] = "REJ-ABC"

    if row["algorithm"] == "SMC-ABC (ours) 100 .5 .2 KDE" and (
        row["num_simulations"] == "10³" or row["num_simulations"] == "10⁴"
    ):
        row["algorithm"] = "SMC-ABC"
    if (
        row["algorithm"] == "SMC-ABC (ours) 1000 .5 .2 KDE"
        and row["num_simulations"] == "10⁵"
    ):
        row["algorithm"] = "SMC-ABC"

    if row["algorithm"] not in ["RF-ABC", "REJ-ABC", "SMC-ABC"]:
        return None

    return row


def sl(row):
    """SL versus NLE-MAF, SNLE-MAF"""
    if row["task"] in ["lotka_volterra", "slcp_distractors"]:
        return None

    if row["algorithm"] == "NLE-MAF":
        row["algorithm"] = "NLE"

    if row["algorithm"] == "SNLE-MAF":
        row["algorithm"] = "SNLE"

    if row["algorithm"] not in [
        "SL",
        "NLE",
        "SNLE",
    ]:
        return None

    if row["num_observation"] > 10:
        return None

    return row


def rej_abc_sweep(row):
    """REJ-ABC hyperparameters"""
    if "REJ-ABC" not in row["algorithm"]:
        return None

    if "SASS" in row["algorithm"]:
        return None

    if "LRA" in row["algorithm"]:
        return None

    return row


def smc_abc_sweep_ours(row):
    """SMC-ABC hyperparams"""
    if "SMC-ABC (ours)" not in row["algorithm"]:
        return None

    if "SASS" in row["algorithm"]:
        return None

    if "LRA" in row["algorithm"]:
        return None

    row["algorithm"] = row["algorithm"][len("SMC-ABC (ours) ") :]

    return row


def smc_abc_sweep_pyabc(row):
    """SMC-ABC hyperparams"""
    if "SMC-ABC (pyabc)" not in row["algorithm"]:
        return None

    if "SASS" in row["algorithm"]:
        return None

    if "LRA" in row["algorithm"]:
        return None

    row["algorithm"] = row["algorithm"][len("SMC-ABC (pyabc) ") :]

    return row


def snle_maf_nsf(row):
    """(S)NLE-MAF versus -NSF"""
    if row["algorithm"] not in [
        "NLE-MAF",
        "SNLE-MAF",
        "NLE-NSF",
        "SNLE-NSF",
    ]:
        return None

    if row["num_observation"] > 10:
        return None

    return row


def snpe_maf_nsf(row):
    """(S)NPE-MAF versus -NSF"""
    if row["algorithm"] not in [
        "NPE-MAF",
        "SNPE-MAF",
        "NPE-NSF",
        "SNPE-NSF",
    ]:
        return None

    if row["num_observation"] > 10:
        return None

    return row


def snre_mlp_res(row):
    """(S)NRE-MLP versus -RES"""
    if row["algorithm"] not in [
        "NRE-MLP",
        "SNRE-MLP",
        "NRE-RES",
        "SNRE-RES",
    ]:
        return None

    if row["num_observation"] > 10:
        return None

    return row


def prior(row):
    """Prior"""
    if "Prior" not in row["algorithm"]:
        return None

    return row


def posterior(row):
    """Posterior"""
    if "Posterior" not in row["algorithm"]:
        return None

    row["algorithm"] = "Ref. Posterior rerun"
    row["num_simulations"] = ">10⁶"

    return row


if __name__ == "__main__":
    df = compile_df(basepath=f"{basepath_runs}")
    df.to_csv(f"{basepath}/df_raw.csv", index=False)
    df = apply_df(df=df, row_fn=clean)
    df.to_csv(f"{basepath}/df.csv", index=False)
    print(len(df))

    print(main_paper.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=main_paper)
    df.to_csv(f"{basepath}/main_paper.csv", index=False)
    assert len(df) == 2400

    print(abc_lra_sass.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=abc_lra_sass)
    df.to_csv(f"{basepath}/supplement_abc_lra_sass.csv", index=False)
    assert len(df) == 1800

    print(rf_abc.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=rf_abc)
    df.to_csv(f"{basepath}/supplement_rf_abc.csv", index=False)
    assert len(df) == 900

    print(sl.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=sl)
    df.to_csv(f"{basepath}/supplement_sl.csv", index=False)
    assert len(df) == 560

    print(rej_abc_sweep.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=rej_abc_sweep)
    df.to_csv(f"{basepath}/supplement_hyperparameters_rej_abc.csv", index=False)
    assert len(df) == 1500

    print(smc_abc_sweep_ours.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=smc_abc_sweep_ours)
    df.to_csv(f"{basepath}/supplement_hyperparameters_smc_abc_ours.csv", index=False)
    assert len(df) == 7200

    print(smc_abc_sweep_pyabc.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=smc_abc_sweep_pyabc)
    df.to_csv(f"{basepath}/supplement_hyperparameters_smc_abc_pyabc.csv", index=False)
    assert len(df) == 7200

    print(snle_maf_nsf.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=snle_maf_nsf)
    df.to_csv(f"{basepath}/supplement_hyperparameters_snle.csv", index=False)
    assert len(df) == 1200

    print(snpe_maf_nsf.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=snpe_maf_nsf)
    df.to_csv(f"{basepath}/supplement_hyperparameters_snpe.csv", index=False)
    assert len(df) == 1200

    print(snre_mlp_res.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=snre_mlp_res)
    df.to_csv(f"{basepath}/supplement_hyperparameters_snre.csv", index=False)
    assert len(df) == 1200

    print(prior.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=prior)
    df.to_csv(f"{basepath}/extra_prior.csv", index=False)
    assert len(df) == 100

    print(posterior.__doc__)
    df = apply_df(df=get_df(path=f"{basepath}/df.csv"), row_fn=posterior)
    df.to_csv(f"{basepath}/extra_posterior.csv", index=False)
    assert len(df) == 80  # 2 tasks share the same posterior
