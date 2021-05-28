cd /home/csl/code/PolSAR_N2N

# python train.py --data.log False --train.optimizer.lr 1e-5
# python train.py --data.log False --train.optimizer.lr 1e-6
# python train.py --data.log False --train.optimizer.lr 1e-7

python train.py --data.log True --train.optimizer.weight_decay 1e-7
python train.py --data.log True --train.optimizer.weight_decay 1e-6
python train.py --data.log True --train.optimizer.weight_decay 1e-5
python train.py --data.log True --train.optimizer.weight_decay 1e-4


# python train.py --train.optimizer.lr 1e-5
# python train.py --train.optimizer.lr 1e-7
# python train.py --data.ENL 3
# python train.py --data.ENL 5