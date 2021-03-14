import os
import argparse
import torch

parser = argparse.ArgumentParser("ckpt surgery tool")

parser.add_argument("--ckpt-path", type=str, required=True)
parser.add_argument("--method")
parser.add_argument("--output-dir", type=str, default=".")


def reset_ckpt(ckpt):
    if 'meta' in ckpt:
        del ckpt['meta']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, "cpu")
    print(f"load ckpt from {ckpt_path}")
    return ckpt


def ckpt_surgery(args):
    ckpt = load_ckpt(args.ckpt_path)
    reset_ckpt(ckpt)
    method = args.method
    if method == "reset":
        pass
    elif method == "remove":
        to_removed_keys = []
        for key in ckpt["state_dict"].keys():
            if "fc_cls" in key:
                to_removed_keys.append(key)
        for key in to_removed_keys:
            ckpt["state_dict"].pop(key)
            print(f"remove weights: {key}")
    else:
        raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)
    save_name = "model_reset_" + method + ".pth"
    save_path = os.path.join(args.output_dir, save_name)
    save_ckpt(ckpt, save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Called with Args: {}".format(args))
    ckpt_surgery(args)
