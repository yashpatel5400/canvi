import time
import os
import torch
import sbibm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

device = "cpu"

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

def get_thetas_grid(mins, maxs):
    theta1 = np.linspace(mins[0], maxs[0], 200)
    theta2 = np.linspace(mins[1], maxs[1], 200)
    dA = (theta1[1] - theta1[0]) * (theta2[1] - theta2[0]) # area element (for approximating integral)
    thetas_unflat = np.meshgrid(theta1, theta2)
    return np.vstack((thetas_unflat[0].flatten(), thetas_unflat[1].flatten())).T.astype(np.float32), dA

def get_exact_vol(encoder, test_x, theta_mins, theta_maxs, conformal_quantile):
    theta_grid, dA = get_thetas_grid(theta_mins, theta_maxs)
    test_x_tiled = np.transpose(np.tile(test_x, (theta_grid.shape[0],1,1)), (1, 0, 2))
    theta_grid_tiled = np.tile(theta_grid, (test_x_tiled.shape[0],1,1))

    flat_x = test_x_tiled.reshape(-1, test_x_tiled.shape[-1])
    flat_theta_grid = theta_grid_tiled.reshape(-1, theta_grid_tiled.shape[-1])
    probs = encoder.log_prob(flat_theta_grid, flat_x).detach().cpu().exp().numpy()
    exact_vol = (np.sum((1 / probs) < conformal_quantile) * dA) / len(test_x)
    print(f"Exact: {exact_vol}")
    return exact_vol

def get_mc_vol_est(prior, encoder, test_x, conformal_quantile, K = 10, S = 1_000):
    mc_set_size_est_ks = []
    for k in np.linspace(0, 1, K):
        # start = time.time() 
        lambda_k = 1 - k / K
        zs = np.random.random((len(test_x), S)) < lambda_k # indicators for mixture draw

        prior_theta_dist = prior.sample((len(test_x), S)).detach().cpu().numpy()
        empirical_theta_dist = encoder.sample((S), test_x).detach().cpu().numpy()
        sample_x = np.transpose(np.tile(test_x, (S,1,1)), (1, 0, 2))

        mixed_theta_dist = np.zeros(empirical_theta_dist.shape)
        mixed_theta_dist[np.where(zs) == 0] = prior_theta_dist[np.where(zs) == 0]
        mixed_theta_dist[np.where(zs) == 1] = empirical_theta_dist[np.where(zs) == 1]

        flat_mixed_theta_dist = mixed_theta_dist.reshape(-1, mixed_theta_dist.shape[-1])
        flat_sample_x = sample_x.reshape(-1, sample_x.shape[-1])
        flat_mixed_theta_dist = torch.Tensor(flat_mixed_theta_dist)
        
        prior_probs = prior.log_prob(flat_mixed_theta_dist).detach().cpu().exp().numpy()
        var_probs = encoder.log_prob(flat_mixed_theta_dist, flat_sample_x).detach().cpu().exp().numpy()
        mixed_probs = lambda_k * prior_probs + (1 - lambda_k) * var_probs

        mc_set_size_est_k = np.mean((1 / var_probs < conformal_quantile).astype(float) / mixed_probs)
        mc_set_size_est_ks.append(mc_set_size_est_k)
    mc_est = np.mean(mc_set_size_est_ks)
    print(f"MC: {mc_est}")
    return mc_est

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    task_name = args.task

    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    total_trials = 5

    trial_sims = 100 # same number for both test and calibration
    sims = (total_trials + 1) * trial_sims

    theta = prior.sample((sims,))
    x = simulator(theta)
    theta = theta[...,:2]

    # very weird, but something odd happens on certain simulation runs if we generate test data at
    # test time -- just generate all data (both test and calibration) ahead of time to avoid this
    thetas = torch.split(theta, trial_sims)
    xs = torch.split(x, trial_sims)

    calibration_theta = thetas[0]
    calibration_x = xs[0]

    test_thetas = thetas[1:]
    test_xs = xs[1:]

    num_coverage_pts = 20
    desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]

    epochs = [100 * i for i in range(26)]

    with open(os.path.join("minmax", "two_moons.pkl"), "rb") as f:
        mins, maxs = pickle.load(f)

    dfs = []
    for epoch in epochs:
        print(f"Computing epoch={epoch}")
        cached_fn = os.path.join("trained", f"{task_name}_iter={epoch}.nf")
        with open(cached_fn, "rb") as f:
            encoder = pickle.load(f)
        encoder.to(device)

        cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
        conformal_quantiles = np.array([np.quantile(cal_scores, q = coverage) for coverage in desired_coverages])
        conformal_quantile = conformal_quantiles[-1] # only consider alpha = 0.05 for now

        set_sizes_exact = []
        mc_set_sizes_est = []
        for batch_idx, test_x in enumerate(test_xs):
            set_sizes_exact.append(get_exact_vol(encoder, test_x, mins, maxs, conformal_quantile))
            mc_set_sizes_est.append(get_mc_vol_est(prior, encoder, test_x, conformal_quantile))
            print(f"Completed batch = {batch_idx}")
        
        df = pd.DataFrame(columns=["sizes", "epoch"])
        df["exact_sizes"] = set_sizes_exact
        df["mc_sizes"] = mc_set_sizes_est
        df["epoch"] = epoch
        df.to_csv(f'{task_name}_sizes.csv', mode='a', header=False)