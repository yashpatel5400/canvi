from sklearn.cluster import KMeans
import numpy as np
import time
import torch
import sbibm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

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

task_name = "two_moons"

device = "cpu"
fn = f"{task_name}"
cached_fn = f"{task_name}_marg_epoch=5000.nf"
with open(cached_fn, "rb") as f:
    encoder = pickle.load(f)
encoder.to(device)

overall_covs = []
specific_covs = []

trials = 1
for trial in range(trials):
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    calibration_sims = 500_000
    calibration_theta = prior.sample((calibration_sims,))
    calibration_x = simulator(calibration_theta)
    calibration_theta = calibration_theta[...,:2]

    test_sim = 10_000
    test_theta = prior.sample((test_sim,))
    test_x = simulator(test_theta)
    test_theta = test_theta[...,:2]

    K = 4
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(calibration_x)
    
    radii = []
    for k in range(K):
        dists = np.linalg.norm(kmeans.cluster_centers_[k] - kmeans.cluster_centers_, axis=1) / 2
        radii.append(np.min(dists[dists != 0]))

    cal_labels = np.ones(calibration_x.shape[0]) * -1
    test_labels = np.ones(test_x.shape[0]) * -1
    for k in range(K):
        test_dists = np.linalg.norm(test_x - kmeans.cluster_centers_[k], axis=1)
        test_labels[test_dists < radii[k]] = k

        cal_dists = np.linalg.norm(calibration_x - kmeans.cluster_centers_[k], axis=1)
        cal_labels[cal_dists < radii[k]] = k
    
    cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
    test_scores = 1 / encoder.log_prob(test_theta.to(device), test_x.to(device)).detach().cpu().exp().numpy()
    
    desired_coverages = np.arange(0.10, 1.0, 0.05)
    overall_quantiles = [np.quantile(cal_scores, q = desired_coverage) for desired_coverage in desired_coverages]
    
    overall_covs = []
    specific_covs = []

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24,12))
    
    overall_coverages_dfs = []
    specific_coverages_dfs = []
    for k in range(K):
        print(f"Computing coverage for region: {k}")
        overall_coverages_df = pd.DataFrame(columns=["k", "confidence", "coverage"])
        overall_coverages_df["confidence"] = desired_coverages
        overall_coverages_df["k"] = k
        specific_coverages_df = overall_coverages_df.copy()
        
        specific_quantiles = [np.quantile(cal_scores[cal_labels == k], q = desired_coverage) for desired_coverage in desired_coverages]
        
        overall_coverages = []
        specific_coverages = []
        for i, desired_coverage in enumerate(desired_coverages):
            test_region_scores = test_scores[test_labels == k]
            covered_overall_quantile = test_region_scores < overall_quantiles[i]
            covered_specific_quantile = test_region_scores < specific_quantiles[i]

            overall_coverages.append(np.sum(covered_overall_quantile) / len(test_region_scores))
            specific_coverages.append(np.sum(covered_specific_quantile) / len(test_region_scores))
        overall_coverages_df["coverage"] = overall_coverages
        specific_coverages_df["coverage"] = specific_coverages

        overall_coverages_dfs.append(overall_coverages_df)
        specific_coverages_dfs.append(specific_coverages_df)

    overall_coverages = pd.concat(overall_coverages_dfs)
    specific_coverages = pd.concat(specific_coverages_dfs)

    axs[0].set_title("Overall Quantile", fontsize=24)
    axs[1].set_title("Specific Quantile", fontsize=24)
    sns.lineplot(data=overall_coverages, x="confidence", y="confidence", ax=axs[0], color="black", lw=4)
    sns.lineplot(data=overall_coverages, x="confidence", y="coverage", ax=axs[0], hue="k", legend="auto", palette="flare", lw=4)
    
    sns.lineplot(data=specific_coverages, x="confidence", y="confidence", ax=axs[1], color="black", lw=4)
    sns.lineplot(data=specific_coverages, x="confidence", y="coverage", ax=axs[1], hue="k", legend="auto", palette="flare", lw=4)
    
    plt.savefig("coverage.png")
    # print(f"Overall: {np.unique(label_overall_covs)[1:]}")
    # print(f"Specific: {np.unique(label_specific_covs)[1:]}")