import sbibm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def coverage_trial(test_theta, test_x):
    test_scores = 1 / encoder.log_prob(test_theta.to(device), test_x.to(device)).detach().cpu().exp().numpy()
    return [np.sum(test_scores < conformal_quantile) / trial_sims for conformal_quantile in conformal_quantiles]

task_names = [
    'two_moons',
    'slcp',
    'gaussian_linear_uniform',
    'bernoulli_glm',
    'gaussian_mixture',
    'gaussian_linear',
    'slcp_distractors',
    'bernoulli_glm_raw'
]

task_name_titles = [
    'Two Moons',
    'SLCP',
    'Gaussian Linear Uniform',
    'Bernoulli GLM',
    'Gaussian Mixture',
    'Gaussian Linear',
    'SLCP Distractors',
    'Bernoulli GLM Raw'
]

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(24,24))
sns.set_theme()

for task_idx, task_name in enumerate(task_names):
    print(f"Plotting {task_name}")
    ax = axs[task_idx // 2, task_idx % 2]
    
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()   

    device = "cuda:0"
        
    fn = f"{task_name}"
    cached_fn = f"{fn}.nf"
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    total_trials = 5

    trial_sims = 10 # same number for both test and calibration
    sims = (total_trials + 1) * trial_sims

    theta = prior.sample((sims,))
    x = simulator(theta)

    # very weird, but something odd happens on certain simulation runs if we generate test data at
    # test time -- just generate all data (both test and calibration) ahead of time to avoid this
    thetas = np.split(theta, sims // trial_sims)
    xs = np.split(x, sims // trial_sims)

    calibration_theta = thetas[0]
    calibration_x = xs[0]

    test_thetas = thetas[1:]
    test_xs = xs[1:]

    num_coverage_pts = 20
    desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]

    cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
    conformal_quantiles = np.array([np.quantile(cal_scores, q = coverage) for coverage in desired_coverages])

    dfs = []
    for i in range(total_trials):
        df = pd.DataFrame(columns=["confidence", "coverages"])
        df["confidence"] = desired_coverages
        df["coverages"] = coverage_trial(test_thetas[i], test_xs[i])
        dfs.append(df)
    df = pd.concat(dfs)

    sns.lineplot(data=df, x="confidence", y="coverages", legend="full", ax=ax)
    sns.lineplot(data=df, x="confidence", y="confidence", c="black", linestyle='--', ax=ax)
        
    task_name_title = task_name_titles[task_idx]
    ax.set_title(task_name_titles[task_idx], fontsize=18)

    if task_idx // 2 != 3:
        ax.set_xlabel("")
    if task_idx % 2 == 1:
        ax.set_ylabel("")
    
plt.tight_layout()
plt.savefig(f"coverages/complete.png")