python main.py --multirun training.loss=iwbo,elbo,favi training.log_eff=true training.device='cpu' training.epochs=25000
python plot.py --multirun training.loss=iwbo,elbo,favi
python calib.py
python canvi_calib.py
python plot_canvi.py --multirun training.loss=iwbo,elbo,favi
python plot_efficiency.py
