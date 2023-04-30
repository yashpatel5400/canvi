import copy
import argparse
import os
import pickle
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import time

import sbibm
from sbi.inference import BNRE, SNRE_A, SNRE_B, SNRE_C
from sbi.inference import SNPE_C
from sbi.inference import MNLE, SNLE_A

plt.rcParams['text.usetex'] = True

# rounds is for refinement (if using sequential alg; if not, set = 0)
def assess_coverage(task, amortized_posterior, fn, rounds = 0, device = "cpu", coverage_trials = 25, num_coverage_pts = 20):
    calibration_prior = task.get_prior()
    calibration_simulator = task.get_simulator()

    variational_coverages = np.zeros(num_coverage_pts)
    desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]
    
    sample_sizes = [1024 * 2 ** j for j in range(8)]
    variational_coverages_per_size = [np.zeros(num_coverage_pts) for _ in sample_sizes]
    times = [[] for _ in sample_sizes]

    for j in range(coverage_trials):
        test_theta = calibration_prior(num_samples=1)
        test_x = calibration_simulator(test_theta)

        posterior = copy.deepcopy(amortized_posterior)

        training_time_start = time.time()
        proposal = posterior
        inference = SNLE_A(task.get_prior_dist())
        for _ in range(rounds):
            num_sims = 100
            theta = proposal.sample((num_sims,))
            x = calibration_simulator(theta)
            _ = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior().set_default_x(test_x[0])
            proposal = posterior
        training_time = time.time() - training_time_start

        posterior.set_default_x(test_x[0])
        predicted_lps = posterior.log_prob(test_theta[0].view(1,-1).to(device)).detach()
        predicted_prob = predicted_lps.cpu().exp().numpy()
        
        for i, sample_size in enumerate(sample_sizes):
            calibration_time_start = time.time()
            empirical_theta_dist = posterior.sample((sample_size,))
            predicted_lps = posterior.log_prob(empirical_theta_dist).detach()
            unnorm_probabilities = predicted_lps.cpu().exp().numpy()

            var_quantiles = np.zeros(len(desired_coverages))
            for k, desired_coverage in enumerate(desired_coverages):
                var_quantiles[k] = np.quantile(unnorm_probabilities, q = 1 - desired_coverage, method="inverted_cdf")
            variational_coverages_per_size[i] += predicted_prob > var_quantiles
            
            calibration_time = time.time() - calibration_time_start
            times[i].append(training_time + calibration_time)
    
    plt.close()
    
    dfs = []
    for i in range(len(variational_coverages_per_size)):
        df = pd.DataFrame(columns=["confidence", "coverages", "times", "sample_sizes",])
        df["confidence"] = desired_coverages
        df["coverages"] = variational_coverages_per_size[i] / coverage_trials
        df["times"] = np.mean(times[i])
        df["sample_sizes"] = sample_sizes[i]
        dfs.append(df)
    df = pd.concat(dfs)    

    sns.set_theme()
    sns.lineplot(data=df, x="confidence", y="coverages", hue="sample_sizes", palette="flare", legend="full")
        
    plt.xlabel("$\\mathrm{Confidence Level}$")
    plt.ylabel("$\\mathrm{Empirical Coverage}$")
    plt.legend()
    plt.title("$\\mathrm{Conformal vs. Variational Coverage}$")
    
    plt.tight_layout()
    plt.savefig(f"{fn}-coverage.png")

    plt.close()
    sns.set_theme()
    sns.lineplot(data=df, x="sample_sizes", y="times")
        
    plt.xlabel("$\\mathrm{Sample Size}$")
    plt.ylabel("$\\mathrm{Time (s)}$")
    plt.title("$\\mathrm{Sample Size vs. Runtime}$")
    
    plt.tight_layout()
    plt.savefig(f"{fn}-time.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--alg")
    parser.add_argument("--rounds", type=int, default=0)
    args = parser.parse_args()

    alg_name = "SNPE_A"
    task = sbibm.get_task(args.task)
    
    fn = f"{args.task}_{args.alg}"
    cache_fn = f"{fn}.pkl"

    if os.path.exists(cache_fn):
        with open(cache_fn, "rb") as f:
            posterior = pickle.load(f)
    else:
        prior = task.get_prior_dist()
        simulator = task.get_simulator()
        observation = task.get_observation(num_observation=1)

        rounds = 1
        num_sims = 100

        inference = eval(args.alg)(prior)
        proposal = prior
        for _ in range(1):
            theta = proposal.sample((num_sims,))
            x = simulator(theta)
            _ = inference.append_simulations(theta, x).train()
            posterior = inference.build_posterior().set_default_x(observation)
            proposal = posterior

        with open(cache_fn, "wb") as f:
            pickle.dump(posterior, f)

    assess_coverage(task, posterior, fn, rounds=args.rounds)

    observation = task.get_observation(num_observation=1)
    posterior.set_default_x(observation)
    samples = posterior.sample((1000,))
    columns = [f"$\\theta_{i}$" for i in range(samples.shape[-1])]
    posterior_df = pd.DataFrame(columns=columns, data=samples)
    posterior_df["label"] = "approximation"

    sns.set_theme()
    sns.pairplot(posterior_df, hue="label", kind="kde", corner=True)
    plt.title("Approximate Posteriors")
    plt.savefig(f"{fn}-posterior.png")
    plt.close()
