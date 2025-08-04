# Daunce on CIFAR-ResNet

This directory contains the implementation of "DAUNCE: Data Attribution through Uncertainty Estimation" on the CIFAR-10 dataset with ResNet-9 model. This readme demonstrates how to performe training data attribution (TDA) with Daunce.

1. **Train a ResNet-9 model (theta_0), on which TDA will be applied**:
    - `python train_resnet.py --lr 0.1`
    - Pick one of the checkpoints from `./checkpoint-resnet9/` directory with good classification accuracy.

2. **Run Daunce**:
    - Set `output_dir`, `theta_0_model` and then `./run.sh`.
    - `run.sh` trains multiple models sequentially, each with a different random seed. Parallelization with multiple GPU is needed for speeding up the process.
    - `run.sh` will save the model checkpoints as well as the signals (e.g. loss, margin, etc.) in `output_dir`.

3. **Visualize the results**:
    - Use `visualize.ipynb` to visualize the results.