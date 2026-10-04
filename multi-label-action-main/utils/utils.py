import os
import socket
import configparser as cp

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def read_config(section=None):
    """
    Reads configuration from location.cfg with robust fallbacks:
    1. Checks environment variable MSQNET_DATA_DIR or DATASET_PATH
    2. Matches hostname in location.cfg
    3. Falls back to [DEFAULT] section in location.cfg
    4. Falls back to default paths (./datasets, ./checkpoints)
    """
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = os.path.join(cur_path, 'location.cfg')
    if not os.path.exists(cfg_file):
        example_cfg = os.path.join(cur_path, 'location.cfg.example')
        if os.path.exists(example_cfg):
            cfg_file = example_cfg

    config = cp.ConfigParser()
    if os.path.exists(cfg_file):
        config.read(cfg_file)

    host = section or socket.gethostname()
    if host[:3] == 'dgk':
        host = 'jade2'

    resolved = {}
    if host in config:
        resolved = dict(config[host])
    elif 'DEFAULT' in config and config['DEFAULT']:
        resolved = dict(config['DEFAULT'])
    else:
        resolved = {
            'path_dataset': os.path.join(cur_path, 'datasets'),
            'path_aux': os.path.join(cur_path, 'checkpoints')
        }

    # Environment variables take precedence if present
    env_data_dir = os.environ.get('MSQNET_DATA_DIR') or os.environ.get('DATASET_PATH')
    if env_data_dir:
        resolved['path_dataset'] = env_data_dir

    env_aux_dir = os.environ.get('MSQNET_AUX_DIR') or os.environ.get('CHECKPOINT_PATH')
    if env_aux_dir:
        resolved['path_aux'] = env_aux_dir

    return resolved