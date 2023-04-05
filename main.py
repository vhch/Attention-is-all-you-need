from model import *
from train import train
from test import test

SEED = 1234

random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=100, type=int,
                        help='number of each process batch number')
    args = parser.parse_args()
    ###########################################################
    args.world_size = args.gpus * args.nodes                  #
    os.environ['MASTER_ADDR'] = '127.0.0.1'                   #
    os.environ['MASTER_PORT'] = '8888'                        #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)#
    ###########################################################

if __name__ == "__main__":
    #main()
    test()
