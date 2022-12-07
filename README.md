
## How to Run

### With Docker

```bash
docker run -it --rm \
  --gpus all --ipc=host -p 7860:7860 --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${PWD}:/workspace \
  -v ${PWD}/.cache:/root/.cache \
  -w /workspace \
  oneflowinc/ai-painter:cu112 \
  python3 launch.py --ip 0.0.0.0 --port 7860
```

### Without Docker

Install depencies:

```bash
sh setup_env.sh
```

Launch the server:

```bash
python3 launch.py --ip 0.0.0.0 --port 7860
```

### Launch Options

There are other options besides `ip` and `port` mentioned above.

- `--ui-debug-mode`: launch without loading model
- `--graph-mode`: use OneFlow graph mode which will accelerate the inference (but limited by fixed tensor shape)
- `--device`: Target a specific device, eg: `cuda:0` means the first GPU and `cuda:1` means the second GPU

### Using Jupyter