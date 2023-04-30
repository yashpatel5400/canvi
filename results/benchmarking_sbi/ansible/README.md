# Ansible

[Ansible](https://www.ansible.com/) is an open-source software for provisioning infrastructure through code. We ran experiments on cloud infrastructure, in particular using Amazon AWS. This repository contains the playbooks that were used to set up servers on AWS and to run the experiments. If you are planning to use another cloud provider or a self-hosted cluster, playbooks would need to be adapted (should be easy to do). In the following, basic familiarity with `ansible` will be assumed, see [Ansible's User Guide](https://docs.ansible.com/ansible/latest/user_guide/index.html) for resources and a brief introduction. 


## Disclaimer

See [Amazon EC2 pricing](https://aws.amazon.com/ec2/pricing/) for costs of AWS EC2 instances. Note that there are associated charges, e.g., for storage on [EBS](https://aws.amazon.com/ebs) and [EFS](https://aws.amazon.com/efs). In order to monitor accumulating costs and monitor server load, we would recommend setting up Amazon CloudFront alarms through the web interface. **Using these playbooks is entirely at your own risk. We do not take any responsibility for unwanted costs or damages.**


## Installation

Assuming you have cloned repository and installed the dependencies, as described in the [README of the top folder](../README.md), install additional ansible role dependencies via:

```commandline
$ ansible-galaxy install -r requirements.yml -f
```

Create a user through IAM via the AWS online console. Use `Programmatic access` and make sure the user has the following rights: `AmazonEC2FullAccess` and `AmazonElasticFileSystemFullAccess` and take note of the Access key ID and Secret access key. Install [AWS Command Line Interface (CLI)](https://aws.amazon.com/en/cli/) and set it up via `aws configure` for that user. Your credentials will be stored in `~/.aws` from which `ansible`, or rather the `boto` package that `ansible` uses internally, will access them (see [`boto` documentation](https://boto.readthedocs.io/en/latest/boto_config_tut.html) for more information). If `aws sts get-caller-identity` executes without an error, the credentials are successfully set up. 

Next, set the environment variable `AWS_REGION` to the region you want to use, e.g. `export AWS_REGION=us-east-1`, as some ansible tasks will make use of it. You may also consider setting `AWS_DEFAULT_REGION` to this value for usage with AWS CLI. 

Finally, you will need to add a [dynamic inventory](https://docs.ansible.com/ansible/latest/user_guide/intro_dynamic_inventory.html#intro-dynamic-inventory). The location of the inventory folder depends on the operating system, on macOS, copy the file `inventory/aws_ec2.yml` to `~/.ansible/inventory/aws_ec2.yml`. Make sure to adjust the region inside the file if you are not planning to use `us-east-1`. With that in place, any AWS EC2 machines you will launch should show up as part of `ansible-inventory --graph` under the prefix `aws_ec2`.


## Server

The playbook `aws_server.yml` sets up the head node which will run the Redis server used with [Hydra's RQ launcher plugin](https://hydra.cc/docs/plugins/rq_launcher), as described in the [README of the top folder](../README.md). When running the experiments for the paper, we used a `t3.medium` instance for the head node using the `Deep Learning Base AMI (Ubuntu 18.04) Version 29.0` (ami-07a0b410c24ce1b0f). 

Before executing the playbook deploying the server, set the environment variable `REDIS_PASSWORD`. To run the playbook:
```commandline
$ ansible-playbook aws_server.yml
```

Take note of the server IP and set the environment variable `REDIS_HOST` before executing other playbooks.


## Workers

The playbook `aws_workers.yml` sets up worker instances which will execute runs through [Hydra's RQ launcher plugin](https://hydra.cc/docs/plugins/rq_launcher). When running the experiments for the paper, we used multiple `c5.24xlarge` (48 CPUs, not to be confused with vCPUs), using the same AMI as for the server. 

Worker can be launched via:
```commandline
$ ansible-playbook aws_workers.yml --extra-vars "aws_ec2_tag_name=worker_1"
```

Note that each worker should receive a unique tag.


## Queue jobs

The playbooks in `enqueue_runs/` can be used to enqueue runs for the benchmark, e.g.:
```commandline
$ ansible-playbook enqueue_runs/main_paper.yml
```

Note that enqueuing might take a while. 
