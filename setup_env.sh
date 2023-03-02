set -ex
python3 -m pip install -r requirements.txt \
    && python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117 \
    && python3 -m pip install "transformers>=4.26" "diffusers[torch]==0.12.1" \
    && python3 -m pip uninstall accelerate -y \
    && python3 -m pip install onediff


BASE_REPO_DIR="./repositories"
mkdir -p $BASE_REPO_DIR

DIFFUSERS_REPO="https://github.com/Oneflow-Inc/diffusers.git"
git clone $DIFFUSERS_REPO $BASE_REPO_DIR/diffusers \
 && cd $BASE_REPO_DIR/diffusers \
 && python3 -m pip install -e .[oneflow] \
 && git checkout oneflow-fork \
 && cd ../.. 

# TRANSFORMERS_REPO="https://github.com/Oneflow-Inc/transformers.git"
# git clone $TRANSFORMERS_REPO $BASE_REPO_DIR/transformers \
#  && cd $BASE_REPO_DIR/transformers \
#  && python3 -m pip install -e . \
#  && cd ../..
