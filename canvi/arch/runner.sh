python main.py --multirun training.loss=iwbo,elbo,favi training.log_eff=true training.epochs=10000
python main.py --multirun training.loss=iwbo,elbo,favi training.log_eff=false
python plot.py --multirun training.loss=iwbo,elbo,favi
python calib.py
python canvi_calib.py
python plot_canvi.py --multirun training.loss=iwbo,elbo,favi
python plot_efficiency.py
