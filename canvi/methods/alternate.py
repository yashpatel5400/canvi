import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sbibm
from sbi.inference import BNRE, SNRE_A, SNRE_B, SNRE_C
from sbi.inference import SNPE_A, SNPE_C
from sbi.inference import MNLE, SNLE_A

plt.rcParams['text.usetex'] = True

def assess_coverage(task, posterior, fn, device = "cpu", coverage_trials = 25, num_coverage_pts = 20):
    calibration_prior = task.get_prior()
    calibration_simulator = task.get_simulator()

    variational_coverages = np.zeros(num_coverage_pts)
    desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]
    
    for j in range(coverage_trials):
        test_theta = calibration_prior(num_samples=1)
        test_x = calibration_simulator(test_theta)

        posterior.set_default_x(test_x[0])

        predicted_lps = posterior.log_prob(test_theta[0].view(1,-1).to(device)).detach()
        predicted_prob = predicted_lps.cpu().exp().numpy()
        
        empirical_theta_dist = posterior.sample((10_000,))
        predicted_lps = posterior.log_prob(empirical_theta_dist).detach()
        unnorm_probabilities = predicted_lps.cpu().exp().numpy()

        var_quantiles = np.zeros(len(desired_coverages))
        for k, desired_coverage in enumerate(desired_coverages):
            var_quantiles[k] = np.quantile(unnorm_probabilities, q = 1 - desired_coverage, method="inverted_cdf")
        variational_coverages += predicted_prob > var_quantiles
    variational_coverages /= coverage_trials

    plt.close()
    plt.plot(desired_coverages, variational_coverages, label="$\\mathrm{Variational}$")
    plt.plot(desired_coverages, desired_coverages, label="$\\mathrm{Desired}$")
    plt.legend()
    plt.title("$\\mathrm{Conformal vs. Variational Coverage}$")
    
    plt.tight_layout()
    plt.savefig(f"{fn}-coverage.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--alg")
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

        # inference = SNRE_C(prior)
        # proposal = prior
        # for _ in range(rounds):
        #     theta = proposal.sample((num_sims,))
        #     x = simulator(theta)
        #     _ = inference.append_simulations(theta, x).train()
        #     posterior = inference.build_posterior().set_default_x(observation)
        #     proposal = posterior


        # inference = MNLE(prior)
        # theta = prior.sample((num_sims,))
        # x = simulator(theta)
        # _ = inference.append_simulations(theta, x).train()
        # posterior = inference.build_posterior().set_default_x(observation)

        inference = eval(args.alg)(prior)
        proposal = prior
        for _ in range(rounds):
            theta = proposal.sample((num_sims,))
            x = simulator(theta)
            _ = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior().set_default_x(observation)
            proposal = posterior

        with open(cache_fn, "wb") as f:
            pickle.dump(posterior, f)

    assess_coverage(task, posterior, fn)

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
