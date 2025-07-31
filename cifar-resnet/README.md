# Daunce on CIFAR-ResNet

This directory contains the implementation of "DAUNCE: Data Attribution through Uncertainty Estimation" on the CIFAR-10 dataset with ResNet-9 model. This readme demonstrates how to performe data attribution with Daunce.

1. **Train the ResNet-9 model (theta_0)**:
    - `python train_resnet.py --lr 0.1`
    - pick one of the checkpoints from `./checkpoint-resnet9/` directory.

2. **Run Daunce**:
    - set `output_dir`, `theta_0_model` and then `./run.sh`
    - `run.sh` train multiple models sequentially, each with a different random seed. Parallelization with multiple GPU is needed if you want to speed up the process.
    - `run.sh` will save the model checkpoints as well as the signals (e.g. loss, margin, etc.) in `output_dir`

3. **Visualize the results**:
    - use visualize.ipynb to visualize the results.