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
        sh_job = f"""#!/bin/bash                                                                                                                                                                                                                     
# The interpreter used to execute the script                                                                                                                                                                                    

#"#SBATCH" directives that convey submission options:                                                                                                                                                                           

#SBATCH --job-name=train_idp                                                                                                                                                                                                    
#SBATCH --mail-type=BEGIN,END                                                                                                                                                                                                   
#SBATCH --nodes=1                                                                                                                                                                                                               
#SBATCH --ntasks-per-node=1                                                                                                                                                                                                     
#SBATCH --mem-per-cpu=8g                                                                                                                                                                                                        
#SBATCH --time=10:00:00                                                                                                                                                                                                            
#SBATCH --account=tewaria0                                                                                                                                                                                                      
#SBATCH --partition=standard

cd ../methods/
python alternate.py --task {task_name} --alg {alg} --rounds 0
"""

        sh_fn = f"../dispatches/{task_name}_{alg}.sh"
        with open(sh_fn, "w") as f:
            f.write(sh_job)