# Instructions

Prediction only:
```sh
uv run pawnet/main.py --no-train-model
```

Train a specific version BX of efficientnet (0-7, default 0)
```sh
uv run pawnet/main.py --train_model efficientnet_bX
```

Train for 37 categories:
```sh
uv run pawnet/main.py --target-types category
```

Train for X epochs:
```sh
uv run pawnet/main.py --epochs X
```

Train X number of layers:
```sh
uv run pawnet/main.py --num-layers X
```

Train with X% of train-val data float in range (0.0,1.0) exclusive:
```sh
uv run pawnet/main.py --train-size 0.X
```

Train with X batch size (power of 2):
```sh
uv run pawnet/main.py --batch-size X
```

Train with gradual unfreezing of X number of layers:
```sh
uv run pawnet/main.py --num-layers X --gradual-unfreezing
```

Train with X% labelled training data float in range (0.0,1.0] (rest becomes unlabelled):
```sh
uv run pawnet/main.py --labeled-fraction 0.X
```

Train with pseudo-labeling, keeping X% of labels:
```sh
uv run pawnet/main.py --labeled-fraction 0.X --use-pseudolabels
```

Train with pseudo-label confidence threshold in [0,1]:
```sh
uv run pawnet/main.py --use-pseudolabels --pseudolabel-threshold 0.95
```

Train with pseudo-label loss weight:
```sh
uv run pawnet/main.py --use-pseudolabels --pseudolabel-weight 0.5
```

Train with pseudo-labeling after N epochs (1-based):
```sh
uv run pawnet/main.py --use-pseudolabels --pseudolabel-start-epoch 2
```
