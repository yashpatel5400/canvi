python main.py --multirun training.loss=iwbo,elbo,favi
python plot.py --multirun training.loss=iwbo,elbo,favi
python calib.py
python canvi_calib.py
python plot_canvi.py --multirun training.loss=iwbo,elbo,favi
