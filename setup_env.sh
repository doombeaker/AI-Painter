set -ex
python3 -m pip install -r requirements.txt \
    && python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117 \
    && python3 -m pip install "transformers>=4.26" "diffusers[torch]==0.14.0" \
    && python3 -m pip uninstall accelerate -y \
    && python3 -m pip install onediff==0.7.0

git clone https://github.com/lllyasviel/ControlNet.git \
    && cd ControlNet \
    && git checkout f4748e3630d8141d7765e2bd9b1e348f47847707 \
    && mv annotator/ ../ \
    && cd .. \
    && rm -rf ControlNet

BASE_REPO_DIR="./annotator/ckpts"
cd $BASE_REPO_DIR

wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth -O body_pose_model.pth \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt -O dpt_hybrid-midas-501f0c75.pt \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth -O hand_pose_model.pth \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth -O mlsd_large_512_fp32.pth \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth -O mlsd_tiny_512_fp32.pth \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth -O network-bsds500.pth \
    && wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth -O upernet_global_small.pth \
    && cd ../.. 