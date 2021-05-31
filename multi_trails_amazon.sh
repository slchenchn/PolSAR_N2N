cd /home/csl/code/PolSAR_N2N

# python train.py --data.log False --train.optimizer.lr 1e-5
# python train.py --data.log False --train.optimizer.lr 1e-6
# python train.py --data.log False --train.optimizer.lr 1e-7

# python train.py --data.log True --train.optimizer.weight_decay 1e-7
# python train.py --data.log True --train.optimizer.weight_decay 1e-6
# python train.py --data.log True --train.optimizer.weight_decay 1e-5
# python train.py --data.log True --train.optimizer.weight_decay 1e-4

# python train.py --data.log True --data.ENL 10 --train.epoch 150

python train.py --data.log True --data.ENL 10 --config_file configs/hoekman_unetpp_simulate_step.yml

# python train.py --train.optimizer.lr 1e-5
# python train.py --train.optimizer.lr 1e-7
# python train.py --data.ENL 3
# python train.py --data.ENL 5