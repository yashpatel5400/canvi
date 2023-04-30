# Benchmarking simulation-based inference

This repository contains the results as well as the scripts that generated them for the [manuscript "Benchmarking simulation-based inference"](http://proceedings.mlr.press/v130/lueckmann21a.html). It also includes Ansible playbooks to recreate the exact server setup on which our experiments were run, as detailed under "Reproducing results" below.

The results were generated using our benchmark framework, `sbibm`, which can be found at: https://github.com/sbi-benchmark/sbibm.

## Results as dataframes

The subfolder `results/` contains csv files with results for the manuscript *Benchmarking simulation-based inference*. They can, for example, be read with [`pandas`](https://pandas.pydata.org) as dataframes for comparisons and further analyses.

| **Filename**                                  | **Description**                                     |
| --------------------------------------------- | --------------------------------------------------- |
| `main_paper.csv`                              | All results for main part of the manuscript         |
| `supplement_rf_abc.csv`                       | RF-ABC results, as in Appendix F                    |
| `supplement_sl.csv`                           | Synthetic Likelihood (SL) results, as in Appendix F |
| `supplement_hyperparameters_rej_abc.csv`      | Hyperparameters for REJ-ABC, as in Appendix H       |
| `supplement_hyperparameters_smc_abc_pyabc.csv`      | Hyperparameters for SMC-ABC (with `pyabc`)          |
| `supplement_hyperparameters_smc_abc_ours.csv` | Hyperparameters for SMC-ABC (our implementation)    |
| `supplement_hyperparameters_snle.csv`         | Hyperparameters for (S)NLE                          |
| `supplement_hyperparameters_snpe.csv`         | Hyperparameters for (S)NPE                          |
| `supplement_hyperparameters_snre.csv`         | Hyperparameters for (S)NRE                          |

## Raw results for each run

Raw results for each single run from which these dataframes were compiled—including samples from all approximate posteriors—are also available. For space reasons, they are hosted in a separately using LFS:

> https://github.com/mackelab/benchmarking_sbi_runs

## Reproducing results

If you want to reproduce the experiments of the benchmark, or want to adapt our code to run your own experiments using the same analysis pipeline, start by cloning this repository:

```commandline
$ git clone git@github.com:sbi-benchmark/results.git
```

Next, install the neccessary dependencies via:

```commandline
$ cd benchmarking_sbi
$ pip install -r requirements.txt
```

Besides the [benchmark framework `sbibm`](https://github.com/sbi-benchmark/sbibm), this will install [Hydra](https://hydra.cc) and [Ansible](https://www.ansible.com). In order to run each algorithm on different tasks, observations and simulation budgets while potentially exploring different hyperparameters, we used [Hydra, a framework for elegantly configuring complex applications](https://hydra.cc) including plugins which we contributed to parallelize workloads ([Joblib launcher](https://hydra.cc/docs/plugins/joblib_launcher) and [RQ launcher](https://hydra.cc/docs/plugins/rq_launcher), to parallize on a single and multiple machines, respectively). Most runs for the benchmark were run on [AWS C5-instances](https://aws.amazon.com/en/ec2/instance-types/): In order to deploy this infrastructure, we used [Ansible](https://www.ansible.com).

The main script for running algorithms is in `run.py`, and configuration files are in `config/`. The following command would run REJ-ABC on `gaussian_linear`, for the first of ten observations, using a budget of 10k simulations:

```commandline
$ python run.py task=gaussian_linear task.num_simulations=10000 algorithm=rej-abc
```

The configuration for this run is the composition of the default configs in `config/config.yaml`, `config/task/gaussian_linear.yaml`, and `config/algorithm/rej-abc.yaml` and the overrides provided in the commandline call. Here, an override `task.num_simulation=10000` is used, which would have been `1000` by default, as defined in `config/task/gaussian_linear.yaml`. This composition is managed by Hydra: To better understand how Hydra works, we would recommend taking a moment to read the [Getting started guide of Hydra](https://hydra.cc/docs/intro), and possibly even the full tutorial at a later point. All other algorithms/tasks/simulation budgets can be launched by modifying the command above.

Results will be stored in a subfolder in `outputs/`. Similar to the raw results included in the repository, you will find the full config, posterior and posterior predictive samples and metrics (along with any additional artefacts the run produced).

It is also possible to launch multiple runs with a single command, when using `--multirun`:

```bash
$ python run.py --multirun task=gaussian_linear task.num_observation=1,2,3 task.num_simulations=1000,10000 algorithm=rej-abc,smc-abc
```

The above command would run Rejection ABC and SMC-ABC for the three observations, for 1k, and 10k simulations each. The 6 simulations would be launched sequentially, which is inefficient given that most workstation have multiple CPUs/GPUs available. We wrote two Hydra plugins to parallelize the workload:

1. Joblib launcher: The Joblib launcher parallelises runs on a single machine. Per default, it uses all available cores. To use it, simply add `hydra/launcher=joblib` to the command. [You can find more information about the plugin here](https://hydra.cc/docs/plugins/joblib_launcher).
2. Redis Queue launcher: [Redis Queue (RQ)](https://python-rq.org) is a "simple Python library for queueing jobs and processing them in the background with workers". The advantage over Joblib is that workloads can be distributed across multiple machines. [You can find more information about the plugin here](https://hydra.cc/docs/plugins/rq_launcher).

As explained in the plugin documentation, you will need to setup a [Redis server](https://redis.io) managing the queue in order to use this plugin. We used [Ansible](https://www.ansible.com) to set up headnode, worker instances and enqueuing runs when running the benchmark. Additional information including the Ansible playbooks used can be found in the `ansible/` subfolder of this repository.

We used the script `results.py` to compile the dataframes from raw results of runs. If you are adopting this code to run your own experiments, the following snippet will create a dataframe for raw results in subfolders:

```python
from utils import compile_df
df = compile_df(basepath="/path/to/raw/results/")
```

One useful and quick way to explore the results of runs is using [HiPlot](https://github.com/facebookresearch/hiplot). To use it, simply pass the analysis dataframe as obtained above in a notebook session:

```python
import hiplot as hip
hip.Experiment.from_dataframe(df).display()
```

## Citation

The manuscript is [available through PMLR](http://proceedings.mlr.press/v130/lueckmann21a.html):

```bibtex
@InProceedings{lueckmann2021benchmarking, 
  title     = {Benchmarking Simulation-Based Inference},
  author    = {Lueckmann, Jan-Matthis and Boelts, Jan and Greenberg, David and Goncalves, Pedro and Macke, Jakob}, 
  booktitle = {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics}, 
  pages     = {343--351}, 
  year      = {2021}, 
  editor    = {Banerjee, Arindam and Fukumizu, Kenji}, 
  volume    = {130}, 
  series    = {Proceedings of Machine Learning Research}, 
  month     = {13--15 Apr}, 
  publisher = {PMLR}
}
```

## License

MIT
