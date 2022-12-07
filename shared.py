import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    default="./ckpt",
    help="path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded",
)
parser.add_argument("--device", type=str, help="device placement, defaults to 'cuda'", default="cuda")
parser.add_argument("--ip", type=str, help="server ip", default="127.0.0.1")
parser.add_argument(
    "--port",
    type=int,
    help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available",
    default=None,
)
parser.add_argument(
    "--ui-debug-mode", action="store_true", help="Don't load model to quickly launch UI"
)
cmd_opts = parser.parse_args()

if __name__ == "__main__":
    print(cmd_opts)
