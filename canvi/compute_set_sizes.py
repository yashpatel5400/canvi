import time
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

    dfs = []
    for epoch in epochs:
        print(f"Computing epoch={epoch}")
        fn = f"{task_name}"
        cached_fn = f"{task_name}_marg_epoch={epoch}.nf"
        with open(cached_fn, "rb") as f:
            encoder = pickle.load(f)
        encoder.to(device)

        cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
        conformal_quantiles = np.array([np.quantile(cal_scores, q = coverage) for coverage in desired_coverages])
        conformal_quantile = conformal_quantiles[-1] # only consider alpha = 0.05 for now

        mc_set_size_ests = []
        for test_x in test_xs:
            start = time.time() 
            variational_dist_samples = 100
            empirical_theta_dist = encoder.sample((variational_dist_samples), test_x)
            sample_x = np.transpose(np.tile(test_x, (variational_dist_samples,1,1)), (1, 0, 2))

            flat_empirical_theta_dist = empirical_theta_dist.reshape(-1, empirical_theta_dist.shape[-1])
            flat_sample_x = sample_x.reshape(-1, sample_x.shape[-1])
            var_probs = encoder.log_prob(flat_empirical_theta_dist, flat_sample_x).detach()
            var_log_probs = var_probs.reshape((trial_sims, -1))
            unnorm_probabilities = var_log_probs.cpu().exp().numpy()

            mc_set_size_est_all = np.sum((1 / unnorm_probabilities < conformal_quantile).astype(float) / unnorm_probabilities, axis=1) / variational_dist_samples
            mc_set_size_est = np.mean(mc_set_size_est_all)
            mc_set_size_ests.append(mc_set_size_est)
            print(f"Completed in {time.time()  - start}")
        
        df = pd.DataFrame(columns=["sizes", "epoch"])
        df["sizes"] = mc_set_size_ests
        df["epoch"] = epoch
        # dfs.append(df)
        # df = pd.concat(dfs)
        df.to_csv(f'{task_name}_sizes.csv', mode='a', header=False)