"""
To launch all the tasks, create tmux sessions (separately for each of the following) 
and run (for instance):

python conformalize_sbibm.py --task two_moons --cuda_idx 0
python conformalize_sbibm.py --task slcp --cuda_idx 1
python conformalize_sbibm.py --task gaussian_linear_uniform --cuda_idx 2
python conformalize_sbibm.py --task bernoulli_glm --cuda_idx 3
python conformalize_sbibm.py --task gaussian_mixture --cuda_idx 4
python conformalize_sbibm.py --task gaussian_linear --cuda_idx 5
python conformalize_sbibm.py --task slcp_distractors --cuda_idx 6
python conformalize_sbibm.py --task bernoulli_glm_raw --cuda_idx 7
"""

import sbibm
import torch
import math
import os
import pickle

import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
import argparse

def generate_data(n_pts, return_theta=False):
    theta = prior(num_samples=n_pts)
    x = simulator(theta)

    if return_theta: 
        return theta, x
    else:
        return x

# Code to plot the true posterior density
def plot(j, x, theta, encoder, device):
    # Plot exact density
    nrows = 4
    ncols = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24,24))

    discretization = .01
    vals = torch.arange(-1., 1., discretization)
    eval_pts = torch.cartesian_prod(vals, vals)
    lps = encoder.log_prob(eval_pts.to(device), x[j].view(1,-1).repeat(eval_pts.shape[0],1).to(device)).detach()
    X, Y = torch.meshgrid(vals, vals)
    Z = lps.view(X.shape).cpu().exp().numpy()

    probabilities = Z.flatten()
    total_mass = np.sum(probabilities)
    sorted_indices = np.argsort(probabilities)[::-1]
    probabilities = probabilities[sorted_indices] / total_mass
    cdf = np.cumsum(probabilities)
            
    ax[0,1].plot(theta[j][0], theta[j][1], marker="x", color="r")
    ax[0,1].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), Z)
    ax[0,1].set_title('Approximate Posterior Flow')

    remaining_spots = nrows * ncols - 2
    for k in range(remaining_spots):
        # can either plot the conformalized posterior regions (with the marginal coverage guarantees)
        coverage_guarantee = 0.05 + 0.1 * k
        qhat = np.quantile(cal_scores, q = coverage_guarantee)
        prob_min = 1 / qhat
        prediction_interval = (Z > prob_min).astype("bool")

        graphic_idx = k + 2
        row_idx = graphic_idx // ncols
        col_idx = graphic_idx - row_idx * ncols

        # find corresponding indices of matrix for lookup: have to invert y convention, since down is positive in index
        ax[row_idx,col_idx].plot(theta[j][0], theta[j][1], marker="x", color="r")
        ax[row_idx,col_idx].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), prediction_interval)
        ax[row_idx,col_idx].set_title(f'Conformalized Posterior: q={coverage_guarantee:.2f}')

def assess_coverage(coverage_trials = 1000, num_coverage_pts = 20):
    test_theta, test_x = generate_data(coverage_trials, return_theta=True)
    
    conformal_coverages = np.zeros(num_coverage_pts)
    variational_coverages = np.zeros(num_coverage_pts)
    desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]
    conformal_quantiles = np.array([1 / np.quantile(cal_scores, q = coverage) for coverage in desired_coverages])
    
    for j in range(coverage_trials):
        predicted_lps = encoder.log_prob(test_theta[j].view(1,-1).to(device), test_x[j].view(1,-1).to(device)).detach()
        predicted_prob = predicted_lps.cpu().exp().numpy()
        conformal_coverages += predicted_prob > conformal_quantiles

        empirical_theta_dist = encoder.sample(50_000, test_x[j].view(1,-1).to(device))
        predicted_lps = encoder.log_prob(empirical_theta_dist[0], test_x[j].repeat(empirical_theta_dist[0].shape[0],1).to(device)).detach()
        unnorm_probabilities = predicted_lps.cpu().exp().numpy()

        var_quantiles = np.zeros(len(desired_coverages))
        for k, desired_coverage in enumerate(desired_coverages):
            var_quantiles[k] = np.quantile(unnorm_probabilities, q = 1 - desired_coverage, method="inverted_cdf")
        variational_coverages += predicted_prob > var_quantiles

    return conformal_coverages / coverage_trials, variational_coverages / coverage_trials, desired_coverages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--cuda_idx")
    args = parser.parse_args()

    task = sbibm.get_task(args.task)
    prior = task.get_prior()
    simulator = task.get_simulator()

    device = f"cuda:{args.cuda_idx}"
    cached_fn = f"sbibm_canvi/{args.task}.nf"
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    # plot sample posterior
    # test_theta, test_x = generate_data(10, return_theta=True)
    # if test_theta.shape[-1] == 2:
    #     plot(j=8, x=test_x, theta=test_theta, encoder=encoder, device=device)

    # perform calibration
    print("Calibrating...")
    calibration_theta, calibration_x = generate_data(10_000, return_theta=True)
    cal_scores = []
    for calibration_theta_pt, calibration_x_pt in zip(calibration_theta, calibration_x):
        log_prob = encoder.log_prob(calibration_theta_pt.view(1,-1).to(device), calibration_x_pt.view(1,-1).to(device)).detach()
        prob = log_prob.cpu().exp().numpy()
        cal_scores.append(1 / prob)
    cal_scores = np.array(cal_scores)

    # assess conformalization
    print("Assessing coverage...")
    conformal_coverages, variational_coverages, desired_coverages = assess_coverage()

    plt.plot(desired_coverages, conformal_coverages, label="Conformalized")
    plt.plot(desired_coverages, variational_coverages, label="Variational")
    plt.plot(desired_coverages, desired_coverages, label="Desired")
    plt.legend()
    plt.title("Conformal vs. Variational Coverage")
    plt.savefig(f"results/{args.task}.png")