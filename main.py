import time
import argparse
import logging
parser = argparse.ArgumentParser()
parser.add_argument('--just_val', type=bool, default=False)
parser.add_argument('--running_time', type=bool, default=False) #                                                     
parser.add_argument('--lr', type=float, nargs='+', default=[6e-6, 5e-3, 8e-4, 3e-3, 4e-4]) # [8e-4, 3e-4, 8e-4, 3e-3, 4e-4]
parser.add_argument('--lr_SPL', type=float, default=5e-4)
parser.add_argument('--wselect', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--output_shape', type=int, default=2048)
parser.add_argument('--alpha', type=float, default=0.1) # discrimination_loss
parser.add_argument('--beta', type=float, default=0.1) # MSE
parser.add_argument('--eta', type=float, default=1) # label loss
parser.add_argument('--gamma', type=float, default=0.9) # feature augmentation
parser.add_argument('--datasets', type=str, default='wiki')  # xmedia, xmedianet, wiki, nus, INRIA-Websearch
parser.add_argument('--view_id', type=int, default=-1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_interval', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)   # note: 100 for websearch, xmedianet, nuswide 200 for wiki, 300 for xmedia 
parser.add_argument('--seed', type=int, default=1)


print("current local time: ", time.asctime(time.localtime(time.time())))
import os
args = parser.parse_args()
seed = 1000
from to_seed import to_seed
to_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

logger_name = args.datasets
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logging/' + logger_name + '.log'),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

def main():
    logger.info(args)
    from DRCL import Solver
    solver = Solver(args, logger)

    solver.train()
    exit()

if __name__ == '__main__':
    main()