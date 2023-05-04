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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['text.usetex'] = True

def generate_data(n_pts, return_theta=False):
    theta = prior(num_samples=n_pts)
    x = simulator(theta)

    if return_theta: 
        return theta, x
    else:
        return x

# Code to plot the true posterior density
def plot(encoders, device, cal_score_per_encoder):
    theta, x = generate_data(1, return_theta=True)
    j = 0

    # Plot exact density
    nrows = 4
    ncols = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24,24))

    labels = [
        "Untrained\ } q_\\varphi(\\cdot) \\mathrm{",
        "Semi-trained\ } q_\\varphi(\\cdot) \\mathrm{",
        "Trained\ } q_\\varphi(\\cdot) \\mathrm{",
    ]

    for encoder_idx, encoder in enumerate(encoders):
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
                
        sns.set_theme()
        ax[0,encoder_idx].plot(theta[j][0], theta[j][1], marker="x", color="r")
        ax[0,encoder_idx].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), Z)
        ax[0,encoder_idx].set_title('$\\mathrm{' + labels[encoder_idx] + '}$')
        ax[0,encoder_idx].set_xticks([])
        ax[0,encoder_idx].set_yticks([])

        desired_coverages = [.80, .90, 0.95]
        for k, desired_coverage in enumerate(desired_coverages):
            # can either plot the conformalized posterior regions (with the marginal coverage guarantees)
            qhat = np.quantile(cal_score_per_encoder[encoder_idx], q = desired_coverage)
            prob_min = 1 / qhat
            prediction_interval = (Z > prob_min).astype("bool")

            # find corresponding indices of matrix for lookup: have to invert y convention, since down is positive in index
            ax[k + 1,encoder_idx].plot(theta[j][0], theta[j][1], marker="x", color="r")
            ax[k + 1,encoder_idx].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), prediction_interval)
            ax[k + 1,encoder_idx].set_title('$\\mathrm{Conformalized\ Posterior:\ q=' + f"{desired_coverage:.2f}" + '}$')
            ax[k + 1,encoder_idx].set_xticks([])
            ax[k + 1,encoder_idx].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"results/{args.task}/credible_regions.png")

def assess_coverage(encoders, cal_score_per_encoder, coverage_trials = 1_000, num_coverage_pts = 20):
    test_theta, test_x = generate_data(coverage_trials, return_theta=True)
    
    labels = [
        "Untrained\ } q_\\varphi(\\cdot) \\mathrm{",
        "Semi-trained\ } q_\\varphi(\\cdot) \\mathrm{",
        "Trained\ } q_\\varphi(\\cdot) \\mathrm{",
    ]
    colors = [
        '#64c987',
        '#089f8f',
        '#215d6e',
    ]

    sns.set_theme()
    
    for encoder_idx, encoder in enumerate(encoders):
        conformal_coverages = np.zeros(num_coverage_pts)
        variational_coverages = np.zeros(num_coverage_pts)
        desired_coverages = [(1 / num_coverage_pts) * k for k in range(num_coverage_pts)]
        conformal_coverages = np.zeros(num_coverage_pts)
        conformal_quantiles = np.array([1 / np.quantile(cal_score_per_encoder[encoder_idx], q = coverage) for coverage in desired_coverages])
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

        sns.lineplot(x=desired_coverages, y=variational_coverages / coverage_trials, color=colors[encoder_idx], label="$\\mathrm{" + labels[encoder_idx] + "}$")
        sns.lineplot(x=desired_coverages, y=conformal_coverages / coverage_trials, color=colors[encoder_idx], linestyle='--')
    sns.lineplot(x=desired_coverages, y=desired_coverages, c="black", linestyle='--', label="$\\mathrm{Desired}$")
    
    plt.legend()
    plt.title("$\\mathrm{Conformal vs. Variational Coverage}$")
        
    plt.tight_layout()
    plt.savefig(f"results/{args.task}/coverage.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--cuda_idx")
    args = parser.parse_args()

    task = sbibm.get_task(args.task)
    prior = task.get_prior()
    simulator = task.get_simulator()

    device = f"cuda:{args.cuda_idx}"
    # cached_fn = f"sbibm_canvi/{args.task}.nf"
    encoders = []
    cached_fns = [
        # f"untrained_{args.task}.nf",
        # f"semitrained_{args.task}.nf",
        f"trained_{args.task}.nf",
    ]
    for cached_fn in cached_fns:
        with open(cached_fn, "rb") as f:
            encoder = pickle.load(f)
        encoders.append(encoder.to(device))

    # visualize posterior
    output_dir = f"results/{args.task}"
    os.makedirs(output_dir, exist_ok=True)

    print("Plotting posteriors...")
    observation_idx = 1
    observation = task.get_observation(num_observation=observation_idx)
    reference_samples_all = task.get_reference_posterior_samples(num_observation=observation_idx)
    reference_samples = reference_samples_all[np.random.choice(len(reference_samples_all), 1_000, replace=False)]
    posterior_samples_all = encoder.sample(50_000, observation[0].view(1,-1).to(device)).cpu().detach()[0]
    posterior_samples = posterior_samples_all[np.random.choice(len(posterior_samples_all), 1_000, replace=False)]

    plt.rcParams['text.usetex'] = True
    # columns = [f"$\\theta_{i}$" for i in range(posterior_samples.shape[-1])]
    # ref_df = pd.DataFrame(columns=columns, data=reference_samples)
    # ref_df["label"] = "$\\mathrm{exact}$"
    # posterior_df = pd.DataFrame(columns=columns, data=posterior_samples)
    # # posterior_df["label"] = "$\\mathrm{approximation}$"
    # df = pd.concat([ref_df, posterior_df])

    # plt.title("$\\mathrm{Reference vs. Approximate Posteriors}$")
    # sns.set_theme()
    # sns.pairplot(posterior_df, kind="kde", corner=True)
    # # plt.tight_layout()
    # plt.savefig(f"results/{args.task}/posterior.png")
    # plt.close()
    
    # perform calibration
    print("Calibrating...")
    cal_score_per_encoder = []
    for encoder in encoders:
        calibration_theta, calibration_x = generate_data(1_000, return_theta=True)
        cal_scores = []
        for calibration_theta_pt, calibration_x_pt in zip(calibration_theta, calibration_x):
            log_prob = encoder.log_prob(calibration_theta_pt.view(1,-1).to(device), calibration_x_pt.view(1,-1).to(device)).detach()
            prob = log_prob.cpu().exp().numpy()
            cal_scores.append(1 / prob)
        cal_scores = np.array(cal_scores).flatten()
        score_quantile = np.quantile(cal_scores, q = 0.95)
        
        sns.histplot(x=cal_scores)
        plt.axvline(score_quantile, color="r")
        plt.text(score_quantile, -10.0,'$\\widehat{q}$')

        plt.xlabel("$1 / q_\\varphi(\\cdot)$")
        plt.ylabel("$\mathcal{P}(\\cdot)$")
        plt.xlim([0, .25])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"results/{args.task}/quantile.png")

        cal_score_per_encoder.append(np.array(cal_scores))
    plt.close()
    plot(encoders, device, cal_score_per_encoder)
    plt.close()

    # assess conformalization
    # print("Assessing coverage...")
    # assess_coverage(encoders, cal_score_per_encoder)