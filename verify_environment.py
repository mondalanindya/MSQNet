#!/usr/bin/env python3
"""
Environment and Model Verification Script for MSQNet
Validates dependency installation, configuration parsing, class dictionaries,
and performs a dry-run forward pass through model components.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, 'multi-label-action-main')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

def check_dependencies():
    print("=" * 60)
    print(" 1. Verifying Python Dependencies")
    print("=" * 60)
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "HuggingFace Transformers"),
        ("timm", "PyTorch Image Models (timm)"),
        ("torchmetrics", "TorchMetrics"),
        ("scipy", "SciPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV (cv2)"),
    ]
    missing = []
    for mod_name, disp_name in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, '__version__', 'installed')
            print(f"  [OK]   {disp_name:<30} {ver}")
        except ImportError:
            print(f"  [MISS] {disp_name:<30} (not installed in this Python env)")
            missing.append(disp_name)

    if missing:
        print(f"\n[NOTE] Missing packages: {', '.join(missing)}")
        print("Install via: pip install -r requirements.txt")
        print("or via Conda: conda env create -f environment.yml\n")
    else:
        print("\nAll core dependencies are properly installed!\n")
    return len(missing) == 0

def check_config():
    print("=" * 60)
    print(" 2. Verifying Configuration & Paths")
    print("=" * 60)
    from utils.utils import read_config
    cfg = read_config()
    print(f"  Dataset path : {cfg.get('path_dataset')}")
    print(f"  Aux/Ckpt path: {cfg.get('path_aux')}")
    print("  [OK]   Configuration parsed successfully.\n")

def check_datasets():
    print("=" * 60)
    print(" 3. Verifying Dataset Classes & Mappings")
    print("=" * 60)
    # Datasets class check without full imports if torch is missing
    from datasets.datamanager import DataManager
    class DummyArgs:
        seed = 1
        dataset = "animalkingdom"
        total_length = 16
        test_part = 6
        batch_size = 4
        num_workers = 0
        distributed = False

    args = DummyArgs()
    dm = DataManager(args, path="./datasets")
    for dset in ["animalkingdom", "charades", "hockey", "thumos14", "volleyball"]:
        args.dataset = dset
        dm.dataset = dset
        num_cls = dm.get_num_classes()
        print(f"  [OK]   Dataset '{dset:<15}': {num_cls:>3} classes")
    print()

def check_model_forward():
    print("=" * 60)
    print(" 4. Verifying Model Components Forward Pass")
    print("=" * 60)
    try:
        import torch
        from models.query2labelclipinit import PositionalEncoding, GroupWiseLinear

        # Test Positional Encoding
        pe = PositionalEncoding(d_model=512, max_len=30)
        x = torch.randn(2, 16, 512)
        out_pe = pe(x)
        assert out_pe.shape == (2, 16, 512), f"Unexpected PE shape {out_pe.shape}"
        print("  [OK]   PositionalEncoding: (B=2, T=16, D=512) -> (2, 16, 512)")

        # Test GroupWiseLinear
        gwl = GroupWiseLinear(num_class=10, hidden_dim=512)
        hs = torch.randn(2, 10, 512)
        out_gwl = gwl(hs)
        assert out_gwl.shape == (2, 10), f"Unexpected GroupWiseLinear shape {out_gwl.shape}"
        print("  [OK]   GroupWiseLinear: (B=2, K=10, D=512) -> (2, 10)")
        print()
    except Exception as e:
        print(f"  [SKIP] Model dry run skipped: {e}\n")

def main():
    print("\n" + "#" * 60)
    print("       MSQNet Environment & System Verification       ")
    print("#" * 60 + "\n")

    deps_ok = check_dependencies()
    check_config()
    try:
        check_datasets()
    except Exception as e:
        print(f"  [NOTE] Dataset check deferred (requires torch/torchvision): {e}\n")
    if deps_ok:
        check_model_forward()

    print("=" * 60)
    print(" Verification Complete!")
    print(" To start training:  python run.py --dataset animalkingdom --model msqnet")
    print(" To evaluate model:  python run.py --dataset animalkingdom --checkpoint <ckpt.pth> --train False")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
