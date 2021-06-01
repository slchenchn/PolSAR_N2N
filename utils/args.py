'''
Author: Shuailin Chen
Created Date: 2021-04-24
Last Modified: 2021-06-01
	content: 
'''
from mylib import nestargs
import yaml
from mylib import types
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_argparser(config_file=None)->nestargs.NestedArgumentParser:
    ''' get a nested argument parser '''

    parser = nestargs.NestedArgumentParser(description="config")

    parser.add_argument(
        "--config_file",
        nargs='?',
        type=str,
        # default="configs/siamdiff.yml",
        help="Configuration file to use"
    )    
    parser.add_argument(
        "--gpu",
        nargs='?',
        type=str,
        default='0',
        help="gpu indices"
    )
    parser.add_argument(
    "--model.arch",
    nargs='?',
    type=str,
    # default='clipped_resnet50',
    help="backbone"
    )
    parser.add_argument(
    "--data.tile_size",
    nargs='?',
    type=int,
    # default=32,
    help="tile size"
    )
    parser.add_argument(
    "--train.epoch",
    nargs='?',
    type=int,
    # default=1000,
    help="epoch"
    )
    parser.add_argument(
    "--train.batch_size",
    nargs='?',
    type=int,
    # default=1280,
    help="batch size"
    )
    parser.add_argument(
    "--train.n_workers",
    nargs='?',
    type=int,
    # default=320,
    help="number of workers"
    )
    parser.add_argument(
    "--train.loss.weight",
    nargs='?',
    type=str,
    # default='0.01,2.0',
    help="weights of classess"
    )
    parser.add_argument(
    "--train.resume",
    nargs='?',
    type=str,
    # default='0.01,2.0',
    help="weights of classess"
    )
    parser.add_argument(
    "--train.optimizer.lr",
    nargs='?',
    type=float,
    # default='0.01,2.0',
    help="learning rate"
    )
    parser.add_argument(
    "--train.optimizer.weight_decay",
    nargs='?',
    type=float,
    # default='0.01,2.0',
    help="weight_decay"
    )
    parser.add_argument(
    "--data.ENL",
    nargs='?',
    type=int,
    # default='0.01,2.0',
    help="equivalent number of looks"
    )
    parser.add_argument(
    "--data.log",
    nargs='?',
    type=str2bool,
    # default='0.01,2.0',
    help="whether to log transform the data"
    )
    parser.add_argument(
    "--train.lr.step_size",
    nargs='?',
    type=int,
    # default='0.01,2.0',
    help="lr step size"
    )
    parser.add_argument(
    "--train.lr.gamma",
    nargs='?',
    type=float,
    # default='0.01,2.0',
    help="lr gamma"
    )


    args = parser.parse_args()
    args.config_file = args.config_file if args.config_file else config_file
    # args.config_file = config_file if config_file else args.config_file
    # print(args.to_flatten_dict())
    
    # str to list, cause the argparser doesn't support well for list-like command-line params
    if args.gpu:
        args.gpu = [int(idx) for idx in args.gpu.split(',')] 
    if args.train.loss.weight:
        args.train.loss.weight = [float(w) for w in args.train.loss.weight.split(',')] 

    with open(args.config_file) as fp:
        args_from_file = yaml.load(fp, Loader=yaml.FullLoader)
    args_from_file = types.flatten_dict(args_from_file)
    args_from_file = nestargs.NestedNamespace(**args_from_file)
    
    args = transfer_attributes(args, args_from_file)
    return args


def transfer_attributes(slave, master):
    ''' tranasfer the not None attributes of slave to master '''
    if type(slave) != type(master):
        raise ValueError('types of master and slave should be the same')
    for k, v in vars(slave).items():
        if isinstance(v, nestargs.NestedNamespace):
            nxt_slave = getattr(slave, k)
            nxt_master = getattr(master, k)
            setattr(master, k, transfer_attributes(nxt_slave, nxt_master))
            # transfer_attributes(nxt_slave, nxt_master)
        elif v is not None:
            setattr(master, k, v)
    return master


if __name__=='__main__':

    a = get_argparser()
    b = vars(a)
    print(b)
    print(a)