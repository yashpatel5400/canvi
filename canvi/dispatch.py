# VERY hacky script but hey, gets the job done
import libtmux

task_name = [
    "two_moons",
    "slcp",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "gaussian_mixture",
    "gaussian_linear",
    "slcp_distractors",
    "bernoulli_glm_raw",
]

server = libtmux.Server()

for task_idx, task_name in enumerate(task_name):
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate chig", enter=True)
    cmd = f"CUDA_VISIBLE_DEVICES={task_idx % 4} python canvi_sbibm.py --task {task_name}"
    p.send_keys(cmd, enter=True)
    print(f"Launched: {cmd}")