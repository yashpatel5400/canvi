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

task_name = "two_moons"

device = "cpu"
fn = f"{task_name}"
cached_fn = f"{task_name}_marg_epoch=5000.nf"
with open(cached_fn, "rb") as f:
    encoder = pickle.load(f)
encoder.to(device)

overall_covs = []
specific_covs = []

trials = 5
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
        dists = np.linalg.norm(test_x - kmeans.cluster_centers_[k], axis=1)
        test_labels[dists < radii[k]] = k

        cal_dists = np.linalg.norm(calibration_x - kmeans.cluster_centers_[k], axis=1)
        cal_labels[cal_dists < radii[k]] = k

    plt.scatter(test_x[:,0],test_x[:,1],c=test_labels.astype(float))
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="r")
    plt.xlim(-2.0, 1.0)
    plt.ylim(-1.5, 1.5)

    cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
    desired_coverage = 0.95
    conformal_quantile = np.quantile(cal_scores, q = desired_coverage)
    probs = encoder.log_prob(test_theta.to(device), test_x.to(device)).detach().cpu().exp().numpy()

    label_overall_covs = test_labels.copy().astype(float)
    label_specific_covs = test_labels.copy().astype(float)

    for k in range(K):
        covered_overall_quantile = (1 / probs[test_labels == k]) < conformal_quantile
        label_overall_covs[test_labels == k] = np.sum(covered_overall_quantile) / len(covered_overall_quantile)

        conformal_quantile_k = np.quantile(cal_scores[cal_labels == k], q = desired_coverage)
        covered_specific_quantile = (1 / probs[test_labels == k]) < conformal_quantile_k
        label_specific_covs[test_labels == k] = np.sum(covered_specific_quantile) / len(covered_specific_quantile)
        print(f"{k} -> {len(cal_scores[cal_labels == k])}")

    print(f"Overall: {np.unique(label_overall_covs)[1:]}")
    print(f"Specific: {np.unique(label_specific_covs)[1:]}")