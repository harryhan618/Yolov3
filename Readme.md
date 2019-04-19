# YOLO object detection in Pytorch

YOLO is a single stage object detection algorithm described in the paper [v2](https://pjreddie.com/media/files/papers/YOLO9000.pdf) and [v3](https://arxiv.org/abs/1612.08242). In this repo, it is re-implemented in Pytorch.



## Demo Test

```
python demo_test.py -i path/to/img -w path/to/mode_weight
```



# Train

1. Specify an experiment directory, e.g. `experiments/exp0`.  Assign the path to variable `exp_dir` in `train.py`.

2. General config is specified in `config.py`. Hyperparameters are specified in `experiments/exp0/cfg.json`.

3. Start training:

   ```python
   python train.py [-r]
   ```

4. Monitor on tensorboard:

   ```
   tensorboard --logdir='experiments/exp0' > experiments/exp0/board.log 2>&1 &
   ```

