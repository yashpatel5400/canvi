# VERY hacky script but hey, gets the job done
import libtmux

task_names = [
    # "two_moons",
    # "slcp",
    # "gaussian_linear_uniform",
    # "bernoulli_glm",
    # "gaussian_mixture",
    "gaussian_linear",
    # "slcp_distractors",
    # "bernoulli_glm_raw",
]

server = libtmux.Server()

cuda_gpus = [1,2,3,5,6,7]
for cuda_gpu in cuda_gpus:
    for task_idx, task_name in enumerate(task_names):
        server.new_session(attach=False)
        session = server.sessions[-1]
        p = session.attached_pane
        p.send_keys("conda activate chig", enter=True)
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_gpu} python canvi_sbibm.py --task {task_name}"
        p.send_keys(cmd, enter=True)
        print(f"Launched: {cmd}")