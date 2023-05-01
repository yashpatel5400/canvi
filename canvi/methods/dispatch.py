# VERY hacky script but hey, gets the job done
import libtmux
import sbibm

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

algs = [
    "SNRE", 
    "SNLE", 
    "SNPE", 
    "BNRE",
]

server = libtmux.Server()

for task_name in task_name:
    for alg in algs:
        server.new_session(attach=False)
        session = server.sessions[-1]
        p = session.attached_pane
        p.send_keys("conda activate chig", enter=True)
        cmd = f"python alternate.py --task {task_name} --alg {alg} --rounds 0"
        p.send_keys(cmd, enter=True)
        print(f"Launched: {cmd}")

        # for any algorithm with sequential variant, also launch corresponding job
        # if alg.startswith("S"):
        #     cmd = f"python alternate.py --task {task_name} --alg {alg} --rounds 5"
        #     p.send_keys(cmd, enter=True)
        #     print(f"Launched: {cmd}")