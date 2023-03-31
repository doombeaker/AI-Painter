set -ex
python3 -m pip install -r requirements.txt \
    && python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117 \
    && python3 -m pip install "transformers>=4.26" "diffusers[torch]==0.14.0" \
    && python3 -m pip uninstall accelerate -y \
    && python3 -m pip install onediff==0.7.0

