import glob
import os
from pathlib import Path
from shutil import copy

import altair as alt
import altair_saver
import deneb as den
import pandas as pd
import sbibm
from sbibm.visualisation import fig_correlation, fig_metric, fig_posterior
from tqdm.auto import tqdm

from utils import get_colors, get_df

basepath_dfs = Path(__file__).parent.absolute() / "results"
basepath_figs = Path(__file__).parent.absolute() / "figures"

all_tasks = [
    ["gaussian_linear", "gaussian_linear_uniform"],
    ["slcp", "slcp_distractors"],
    ["bernoulli_glm", "bernoulli_glm_raw"],
    ["gaussian_linear"],
    ["gaussian_linear_uniform"],
    ["slcp"],
    ["slcp_distractors"],
    ["bernoulli_glm"],
    ["bernoulli_glm_raw"],
    ["gaussian_mixture"],
    ["two_moons"],
    ["sir"],
    ["lotka_volterra"],
]

all_metrics = ["C2ST", "MMD", "KSD", "MEDDIST", "RT"]


def figs_main_paper():
    df = get_df(path=f"{basepath_dfs}/main_paper.csv",)
    df.loc[df["algorithm"] == "REJ-ABC", "algorithm"] = " REJ-ABC"
    for tasks in tqdm(all_tasks):
        for metric in all_metrics:
            plot_task_metric(df, tasks, metric, subfolder="main_paper")
            plot_task_metric(df, tasks, metric, subfolder="main_paper", labels=False)


def figs_abc_lra_sass():
    df = get_df(path=f"{basepath_dfs}/supplement_abc_lra_sass.csv",)

    for tasks in tqdm(all_tasks):
        for metric in ["C2ST"]:
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="abc_lra_sass",
                default_color=get_colors(df=df),
            )
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="abc_lra_sass",
                labels=False,
                default_color=get_colors(df=df),
            )


def figs_rf_abc():
    df = get_df(path=f"{basepath_dfs}/supplement_rf_abc.csv",)
    subfolder = "rf_abc"
    df.loc[df["algorithm"] == "RF-ABC", "algorithm"] = " RF-ABC"
    for tasks in tqdm(all_tasks):
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder)
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder, labels=False)


def figs_sl():
    df = get_df(path=f"{basepath_dfs}/supplement_sl.csv",)
    subfolder = "sl"
    df.loc[df["algorithm"] == "SL", "algorithm"] = " SL"
    df.loc[df["algorithm"] == "NLE-MAF", "algorithm"] = "NLE"
    df.loc[df["algorithm"] == "SNLE-MAF", "algorithm"] = "SNLE"
    for tasks in tqdm(all_tasks):
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder)
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder, labels=False)


def figs_hyperparameters_rej_abc():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_rej_abc.csv",)
    for tasks in tqdm(all_tasks):
        for metric in ["C2ST"]:
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_rej_abc",
                default_color=get_colors(df=df, include_defaults=True)["REJ"],
            )
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_rej_abc",
                labels=False,
                default_color=get_colors(df=df, include_defaults=True)["REJ"],
            )


def figs_hyperparameters_smc_abc_ours():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_smc_abc_ours.csv",)

    for tasks in tqdm(all_tasks):
        for metric in ["C2ST"]:
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_smc_abc_ours",
                default_color=get_colors(df=df, include_defaults=True)["SMC"],
                rotate_labels=True,
            )
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_smc_abc_ours",
                labels=False,
                default_color=get_colors(df=df, include_defaults=True)["SMC"],
                rotate_labels=True,
            )


def figs_hyperparameters_smc_abc_pyabc():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_smc_abc_pyabc.csv",)

    for tasks in tqdm(all_tasks):
        for metric in ["C2ST"]:
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_smc_abc_pyabc",
                default_color=get_colors(df=df, include_defaults=True)["SMC"],
                rotate_labels=True,
            )
            plot_task_metric(
                df,
                tasks,
                metric,
                subfolder="hyperparameters_smc_abc_pyabc",
                labels=False,
                default_color=get_colors(df=df, include_defaults=True)["SMC"],
                rotate_labels=True,
            )


def figs_hyperparameters_snle():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_snle.csv",)
    for tasks in tqdm(all_tasks):
        for metric in ["C2ST", "RT"]:
            plot_task_metric(df, tasks, metric, subfolder="hyperparameters_snle")
            plot_task_metric(
                df, tasks, metric, subfolder="hyperparameters_snle", labels=False
            )


def figs_hyperparameters_snpe():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_snpe.csv",)
    for tasks in tqdm(all_tasks):
        for metric in ["C2ST", "RT"]:
            plot_task_metric(df, tasks, metric, subfolder="hyperparameters_snpe")
            plot_task_metric(
                df, tasks, metric, subfolder="hyperparameters_snpe", labels=False
            )


def figs_hyperparameters_snre():
    df = get_df(path=f"{basepath_dfs}/supplement_hyperparameters_snre.csv",)
    for tasks in tqdm(all_tasks):
        for metric in ["C2ST", "RT"]:
            plot_task_metric(df, tasks, metric, subfolder="hyperparameters_snre")
            plot_task_metric(
                df, tasks, metric, subfolder="hyperparameters_snre", labels=False
            )


def figs_posterior():
    df = get_df(path=f"{basepath_dfs}/extra_posterior.csv",)
    subfolder = "extra_posterior"
    for tasks in tqdm(all_tasks):
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder)
        plot_task_metric(df, tasks, "C2ST", subfolder=subfolder, labels=False)


def figs_correlations():
    df = get_df(path=f"{basepath_dfs}/main_paper.csv",)
    subfolder = "correlations"
    for task in tqdm(df.task.unique()):
        plot_task_correlation(df, task, subfolder=subfolder)


def figs_mmd():
    subfolder = "mmd"
    if not os.path.exists(basepath_figs / subfolder):
        os.makedirs(basepath_figs / subfolder)

    chart = fig_posterior(
        task_name="two_moons",
        width=200,
        height=200,
        samples_path="runs/8ad671f1-1e7b-4af8-b0e9-ec35a99d354f/posterior_samples.csv.bz2",
        samples_name="REJ",
        title="REJ-ABC on Two Moons",
        legend=False,
        colors_dict=get_colors(include_defaults=True),
        config="manuscript",
    )

    path = str(basepath_figs / subfolder / f"rej_abc.svg")
    chart.save(path)
    for ff in ["pdf", "png"]:
        den.convert_file(path, to=ff, debug=0)

    chart = fig_posterior(
        task_name="two_moons",
        width=200,
        height=200,
        samples_path="runs/e37c16cf-3dda-4e26-ac3e-781fd27971a8/posterior_samples.csv.bz2",
        samples_name="SNLE",
        title="SNLE on Two Moons",
        legend=False,
        colors_dict=get_colors(include_defaults=True),
        config="manuscript",
    )

    path = str(basepath_figs / subfolder / f"snle.svg")
    chart.save(path)
    for ff in ["pdf", "png"]:
        den.convert_file(path, to=ff, debug=0)


def plot_task_correlation(df, task, subfolder=""):
    df_ = df[df["task"] == task]

    if not os.path.exists(basepath_figs / subfolder):
        os.makedirs(basepath_figs / subfolder)

    chart = fig_correlation(
        df=df_,
        metrics=["C2ST", "MMD", "MEDDIST", "KSD", "NLTP"],
        config="manuscript",
        title=sbibm.get_task_name_display(task),
    )
    path = str(basepath_figs / subfolder / f"{task}.svg")

    chart.save(path)

    for ff in ["pdf", "png"]:
        den.convert_file(path, to=ff, debug=0)

    return chart


def plot_task_metric(
    df,
    tasks,
    metric,
    subfolder="",
    labels=True,
    default_color="#000000",
    rotate_labels=False,
):
    if type(tasks) == str:
        tasks = list(tasks)

    if not os.path.exists(basepath_figs / subfolder):
        os.makedirs(basepath_figs / subfolder)

    keywords = {}

    if len(tasks) == 1:
        task = tasks[0]
        df_ = df[df["task"] == task]
        title = sbibm.get_task_name_display(task)
    else:
        df_ = df.query("task in (" + ", ".join([f"'{s}'" for s in tasks]) + ")")
        task = "_".join(tasks)
        title = " / ".join([sbibm.get_task_name_display(task) for task in tasks])
        keywords["shape"] = alt.Shape(
            "task:N",
            scale=alt.Scale(
                range=["circle", "square", "triangle-up", "cross", "diamond"]
            ),
            legend=None,
        )
        keywords["detail"] = ("task:N",)

        # ● ■ ▲
        if len(tasks) == 2:
            title = "● " + str(
                sbibm.get_task_name_display(tasks[0])
            ) + "  /  " "■ " + str(sbibm.get_task_name_display(tasks[1]))

    if rotate_labels:
        keywords["column_keywords"] = {"labelAngle": 270, "labelAlign": "right"}

    if len(df_) == 0:
        print("Empty df")
        return None

    chart = fig_metric(
        df=df_.sort_values("algorithm"),
        metric=metric,
        config="manuscript",
        title=title,
        title_dx=20,
        labels=labels,
        keywords=keywords,
        default_color=default_color,
        colors_dict=get_colors(df=df_),
    )

    if labels:
        path = str(basepath_figs / subfolder / f"{task}_{metric}.svg")
    else:
        path = str(basepath_figs / subfolder / f"{task}_{metric}_no_labels.svg")
    chart.save(path)

    fin = open(path, "rt")
    data = fin.read()
    data = data.replace(
        '\'Number of Simulations\'" style="pointer-events: none;"><text text-anchor="middle" transform="translate(0,9)"',
        '\'Number of Simulations\'" style="pointer-events: none;"><text text-anchor="middle" transform="translate(0,-10)"',
    )  # Move x-axis label
    data = data.replace(
        'rotate(-90) translate(0,-3)" style="font-family: Inter;',
        'rotate(-90) translate(0,25)" style="font-family: Inter;',
    )  # Move y-axis label
    if not labels:
        data = data.replace(
            'transform="translate(20,9)', 'transform="translate(20,15)',
        )  # Move title
    fin.close()

    fin = open(path, "wt")
    fin.write(data)
    fin.close()

    for ff in ["pdf", "png"]:
        den.convert_file(path, to=ff, debug=0)

    return chart


def update_figs(logfile, latex_dir):
    pdfs = []
    with open(logfile) as fp:
        for line in fp:
            line = line.strip("\n")
            if line[-4:] == ".pdf":
                pdfs.append("/".join(line.split("/")[-2:]))

    for i in range(len(pdfs)):
        source = Path("figures/" + pdfs[i])
        target = Path(latex_dir).expanduser() / Path(pdfs[i])
        if source.exists():
            copy(source, target)


if __name__ == "__main__":
    figs_main_paper()
    figs_abc_lra_sass()
    figs_rf_abc()
    figs_sl()
    figs_hyperparameters_rej_abc()
    figs_hyperparameters_smc_abc_ours()
    figs_hyperparameters_smc_abc_pyabc()
    figs_hyperparameters_snle()
    figs_hyperparameters_snpe()
    figs_hyperparameters_snre()
    figs_correlations()
    figs_mmd()
    figs_posterior()
