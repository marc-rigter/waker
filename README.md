
# Reward-Free Curricula for Training Robust World Models
![](https://github.com/marc-rigter/waker-dev/tree/clean/terrain_walker.gif)

Official code to reproduce experiments from the ICLR 2024 paper. Proposes the algorithm *WAKER: Weighted Acquisition of Knowledge across Environments for Robustness*.

## Setup

Install dependencies via pip:

```
cd waker
pip3 install -r requirements.txt
```

You must also install MuJoCo 210 to use the SafetyGym environments:
```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C ~/.mujoco
```

## Running the code

To reproduce the experiments in the paper, run the code using:
```
python3 waker/train.py --logdir ~/log_dir --configs domain alg expl_policy 
```

Where:
- domain is TerrainWalker, TerrainHopper, CleanUp, or CarCleanUp.
- alg is WAKER-M, WAKER-R, DR, HardestEnvOracle, ReweightingOracle, or GradualExpansion.
- expl_policy is Plan2Explore or RandomExploration.

Example:
```
python3 waker/train.py --logdir ~/log_dir --configs TerrainWalker WAKER-M Plan2Explore
```

## Citing WAKER

```
@article{rigter2024waker,
  title={Reward-Free Curricula for Training Robust World Models},
  author={Rigter, Marc and Jiang, Minqi and Posner, Ingmar},
  journal={International Conference on Learning Representations},
  year={2024}
}
```