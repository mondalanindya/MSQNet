import os
import sys
import string
import random
import argparse
from utils.utils import read_config

try:
    import torch
    import numpy as np
    import torch.multiprocessing as mp
    from torch.distributed import init_process_group, destroy_process_group
except ImportError:
    torch = None
    np = None
    mp = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")
    init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        if rank == 0:
            print(f"[INFO] Setting SEED: {args.seed}", flush=True)   
    else:
        if rank == 0:
            print("[INFO] Setting SEED: None", flush=True)

    if not torch.cuda.is_available():
        if rank == 0:
            print("[WARNING] CUDA is not available.", flush=True)

    if rank == 0:
        print(f"[INFO] Found {torch.cuda.device_count()} GPU(s) available.", flush=True)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"[INFO] Primary device type: {device}", flush=True)
    
    # Resolve dataset folder name
    if args.dataset.lower() == "animalkingdom":
        dataset_folder = 'AnimalKingdom'
    elif args.dataset.lower() == "ava":
        dataset_folder = 'AVA'
    elif args.dataset.lower() == "thumos14":
        dataset_folder = 'THUMOS14'
    else:
        dataset_folder = string.capwords(args.dataset)

    # Resolve dataset path
    if args.data_dir:
        norm_dir = os.path.normpath(args.data_dir)
        if os.path.basename(norm_dir).lower() == dataset_folder.lower():
            path_data = norm_dir
        else:
            path_data = os.path.join(norm_dir, dataset_folder)
    else:
        config = read_config()
        path_data = os.path.join(config.get('path_dataset', './datasets'), dataset_folder)

    if rank == 0:
        print(f"[INFO] Dataset path: {path_data}", flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)
    if rank == 0:
        print(f"[INFO] Train size: {len(train_loader.dataset)}", flush=True)

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
    if rank == 0:
        print(f"[INFO] Test size: {len(val_loader.dataset)}", flush=True)

    # criterion or loss
    import torch.nn as nn
    dataset_key = args.dataset.lower()
    if dataset_key in ['animalkingdom', 'charades', 'hockey', 'volleyball']:
        criterion = nn.BCEWithLogitsLoss()
    elif dataset_key in ['thumos14', 'hmdb51']:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # evaluation metric
    if dataset_key in ['animalkingdom', 'charades']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision (mAP)'
    elif dataset_key in ['hockey', 'volleyball']:
        from torchmetrics.classification import MultilabelAccuracy
        eval_metric = MultilabelAccuracy(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Accuracy'
    elif dataset_key in ['thumos14', 'hmdb51']:
        from torchmetrics.classification import MulticlassAccuracy
        eval_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
        eval_metric_string = 'Multiclass Accuracy'
    else:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Evaluation Metric'

    # model instantiation
    model_args = (train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed, rank)
    model_name = args.model.lower()

    if model_name in ['msqnet', 'timesformerclipinitvideoguide', 'msqnet_video']:
        from models.timesformerclipinitvideoguide import TimeSformerCLIPInitVideoGuideExecutor
        executor = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
    elif model_name in ['convit']:
        from models.convit import ConViTExecutor
        executor = ConViTExecutor(*model_args)
    elif model_name in ['query2label']:
        from models.query2label import Query2LabelExecutor
        executor = Query2LabelExecutor(*model_args)
    elif model_name in ['query2labelclipinit', 'msqnet_q2l']:
        from models.query2labelclipinit import Query2LabelCLIPInitExecutor
        executor = Query2LabelCLIPInitExecutor(*model_args)
    elif model_name in ['query2labelclip']:
        from models.query2labelclip import Query2LabelCLIPExecutor
        executor = Query2LabelCLIPExecutor(*model_args)
    elif model_name in ['timesformer']:
        from models.timesformer import TimeSformerExecutor
        executor = TimeSformerExecutor(*model_args)
    elif model_name in ['timesformerclipinit', 'msqnet_text_only']:
        from models.timesformerclipinit import TimeSformerCLIPInitExecutor
        executor = TimeSformerCLIPInitExecutor(*model_args)
    elif model_name in ['timesformerresidualclipinit']:
        from models.timesformerresidualclipinit import TimeSformerResidualCLIPInitExecutor
        executor = TimeSformerResidualCLIPInitExecutor(*model_args)
    elif model_name in ['videomae']:
        from models.videomae import VideoMAEExecutor
        executor = VideoMAEExecutor(*model_args)
    elif model_name in ['videomaeclipinit']:
        from models.videomaeclipinit import VideoMAECLIPInitExecutor
        executor = VideoMAECLIPInitExecutor(*model_args)
    elif model_name in ['videomaeclipinitvideoguide']:
        from models.videomaeclipinitvideoguide import VideoMAECLIPInitVideoGuideExecutor
        executor = VideoMAECLIPInitVideoGuideExecutor(*model_args)
    elif model_name in ['adaptformer']:
        from models.adaptformerm import AdaptFormermExecutor
        executor = AdaptFormermExecutor(*model_args)
    elif model_name in ['adaptformerclipinit']:
        from models.adaptformerclipinit import AdaptFormerCLIPInitExecutor
        executor = AdaptFormerCLIPInitExecutor(*model_args)
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            if rank == 0:
                print(f"[INFO] Loading checkpoint from: {args.checkpoint}", flush=True)
            executor.load(args.checkpoint)
        else:
            if rank == 0:
                print(f"[WARNING] Checkpoint file not found: {args.checkpoint}", flush=True)

    if args.train:
        executor.train(args.epoch_start, args.epochs)
        if rank == 0:
            config = read_config()
            save_dir = args.save_dir or config.get('path_aux', './checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            ckpt_filename = f"msqnet_dist_{args.model}_{args.dataset}{('_' + args.id) if args.id else ''}.pth"
            save_path = os.path.join(save_dir, ckpt_filename)
            executor.save(save_path)
            print(f"[INFO] Checkpoint saved successfully to: {save_path}", flush=True)

    eval_score = executor.test()
    if rank == 0:
        print(f"[INFO] Result: {eval_metric_string} = {eval_score * 100:.2f}%", flush=True)

    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed training script for MSQNet")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: 1")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="animalkingdom", help="Dataset: animalkingdom, charades, hockey, thumos14, volleyball")
    parser.add_argument("--data_dir", default=None, type=str, help="Path to datasets root (overrides location.cfg)")
    parser.add_argument("--model", default="msqnet", help="Model: msqnet, timesformerclipinitvideoguide, timesformerclipinit, timesformer, query2labelclipinit, videomae, adaptformer")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video clip (default: 16)")
    parser.add_argument("--batch_size", default=16, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="Location of checkpoint file, used to resume or evaluate")
    parser.add_argument("--save_dir", default=None, help="Directory to save checkpoints")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of worker processes for DataLoader")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=True, type=str2bool, help="Distributed training flag")
    parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    parser.add_argument("--zero_shot", default=False, type=str2bool, help="Zero-shot or Fully supervised")
    parser.add_argument("--train", default=True, type=str2bool, help="Train model before test")
    args = parser.parse_args()

    if torch is None:
        print("[ERROR] PyTorch is required to run distributed training. Please run: pip install -r requirements.txt")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("[WARNING] torch.cuda.device_count() <= 1. Falling back to single process execution.", flush=True)
        main(0, 1, args)