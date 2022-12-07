set -ex
python3 -m pip install -r requirements.txt \
    && python3 -m pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow

BASE_REPO_DIR="./repositories"
mkdir -p $BASE_REPO_DIR

DIFFUSERS_REPO="https://github.com/Oneflow-Inc/diffusers.git"
git clone $DIFFUSERS_REPO $BASE_REPO_DIR/diffusers \
 && cd $BASE_REPO_DIR/diffusers \
 && python3 -m pip install -e .[oneflow] \
 && cd ../.. 

TRANSFORMERS_REPO="https://github.com/Oneflow-Inc/transformers.git"
git clone $TRANSFORMERS_REPO $BASE_REPO_DIR/transformers \
 && cd $BASE_REPO_DIR/transformers \
 && python3 -m pip install -e . \
 && cd ../..
