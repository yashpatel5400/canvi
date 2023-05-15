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

def coverage_trial(encoder, test_theta, test_x):
    test_scores = 1 / encoder.log_prob(test_theta.to(device), test_x.to(device)).detach().cpu().exp().numpy()
    return [np.sum(test_scores < conformal_quantile) / trial_sims for conformal_quantile in conformal_quantiles]

def var_coverage_trial(encoder, test_theta, test_x):
    variational_dist_samples = 100
    empirical_theta_dist = encoder.sample((variational_dist_samples), test_x)
    sample_x = np.transpose(np.tile(test_x, (variational_dist_samples,1,1)), (1, 0, 2))

    flat_empirical_theta_dist = empirical_theta_dist.reshape(-1, empirical_theta_dist.shape[-1])
    flat_sample_x = sample_x.reshape(-1, sample_x.shape[-1])
    var_probs = encoder.log_prob(flat_empirical_theta_dist, flat_sample_x).detach()
    var_log_probs = var_probs.reshape((trial_sims, -1))
    unnorm_probabilities = var_log_probs.cpu().exp().numpy()

    var_quantiles = []
    for k, desired_coverage in enumerate(desired_coverages):
        var_quantiles.append(np.quantile(unnorm_probabilities, q = 1 - desired_coverage, method="inverted_cdf", axis=1))
    var_quantiles = np.transpose(np.array(var_quantiles), (1, 0))

    predicted_prob = encoder.log_prob(test_theta, test_x).cpu().exp().detach().numpy()
    tiled_predicted_probs = np.tile(predicted_prob.reshape(-1,1), (1,20))
    return np.sum(tiled_predicted_probs > var_quantiles, axis=0) / trial_sims

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

colors = [
    "#92dc7e",
    # "#39b48e",
    "#00898a",
    # "#08737f",
    "#2a4858",
]

iterates = [0, 2000, 4000]
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(24,24))
sns.set_theme()

# for task_idx, task_name in enumerate(task_names):

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    task_name = args.task

    for iterate_idx, iterate in enumerate(iterates):
        print(f"Plotting {task_name} -- iterate {iterate}")
        # ax = axs[task_idx // 2, task_idx % 2]
        
        task = sbibm.get_task(task_name)
        prior = task.get_prior_dist()
        simulator = task.get_simulator()   

        fn = f"{task_name}"
        cached_fn = f"{task_name}_epoch={iterate}.nf"
        with open(cached_fn, "rb") as f:
            encoder = pickle.load(f)
        encoder.to(device)

        total_trials = 5

        trial_sims = 10_000 # same number for both test and calibration
        sims = (total_trials + 1) * trial_sims

        theta = prior.sample((sims,))
        x = simulator(theta)

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

        cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
        conformal_quantiles = np.array([np.quantile(cal_scores, q = coverage) for coverage in desired_coverages])

        dfs = []
        for i in range(total_trials):
            # yikes, not a fan of this try/catch but sometimes have weird sampling crashes in nflows? not exactly
            # sure why, but sporadically happens so just ignore for now
            try:
                df = pd.DataFrame(columns=["confidence", "coverages"])
                df["confidence"] = desired_coverages
                print(f"CP Trial {i}")
                df["coverages"] = coverage_trial(encoder, test_thetas[i], test_xs[i])
                print(f"Var Trial {i}")
                df["var_coverages"] = var_coverage_trial(encoder, test_thetas[i], test_xs[i])
                df.to_csv(f'{task_name}_epoch={iterate}.csv', mode='a', header=False)

                # dfs.append(df)
            except:
                continue
        # df = pd.concat(dfs)

#         sns.lineplot(data=df, x="confidence", y="coverages", c=colors[iterate_idx], linestyle='--', ax=ax)
#         if task_idx == 0:
#             gfg = sns.lineplot(data=df, x="confidence", y="var_coverages", c=colors[iterate_idx], legend="full", label=f"Epoch {iterate}", ax=ax)
#             plt.setp(gfg.get_legend().get_texts(), fontsize='20') 
#         else:
#             sns.lineplot(data=df, x="confidence", y="var_coverages", c=colors[iterate_idx], ax=ax)

#         if iterate_idx == 0:
#             sns.lineplot(data=df, x="confidence", y="confidence", c="black", ax=ax)
            
#         task_name_title = task_name_titles[task_idx]
#         ax.set_title(task_name_titles[task_idx], fontsize=24)

#         if task_idx // 2 != 3:
#             ax.set_xlabel("")
#         else:
#             ax.set_xlabel("Confidence",fontsize=20)
        
#         if task_idx % 2 == 1:
#             ax.set_ylabel("")
#         else:
#             ax.set_ylabel("Coverage",fontsize=20)
    
# plt.suptitle('Confidence vs. Coverage (SBI Benchmarks)', fontsize=28)
# plt.tight_layout()
# plt.subplots_adjust(top=0.94)
# plt.savefig(f"coverages/complete.png")