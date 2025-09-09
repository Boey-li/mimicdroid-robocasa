# MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos

<img src="docs/images/mimicdroid-pullfigure.png" width="100%" />

[Home page](https://ut-austin-rpl.github.io/MimicDroid/) · [Paper (TODO)](https://ut-austin-rpl.github.io/MimicDroid/)

This project builds on [**RoboCasa**](https://robocasa.ai), a large-scale simulation framework for training generally capable robots to perform everyday tasks. Please cite RoboCasa if you use this codebase.  

---
We introduce a benchmark built on RoboCasa, spanning **30 objects, 8 kitchen environments, and 8 hours of human play data** for training. All the training environments are shown below:

<img src="docs/images/mimicdroid-train-all.png" width="100%" />

</br>
</br>
Evaluation is structured into **three levels** with increasing difficulty and **4 tasks** in each level.

| Level | Task Name | Abstract Embodiment | Humanoid Embodiment |
|-------|-----------|---------------|------------|
| **L1 (Seen Objects, Seen Environment)** | PnPSinkToRightCounterPlate | <img src="docs/images/tasks/PnPSinkToRightCounterPlate.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToRightCounterPlate_eval.png" width="200"/> |
|       | PnPSinkToCabinet | <img src="docs/images/tasks/PnPSinkToCabinet.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToCabinet_eval.png" width="200"/> |
|       | TurnOnFaucet | <img src="docs/images/tasks/TurnOnFaucet.png" width="200"/> | <img src="docs/images/tasks/TurnOnFaucet_eval.png" width="200"/> |
|       | CloseLeftCabinetDoor | <img src="docs/images/tasks/CloseLeftCabinetDoor.png" width="200"/> | <img src="docs/images/tasks/CloseLeftCabinetDoor_eval.png" width="200"/> |
| **L2 (Unseen Objects, Seen Environment)** | PnPSinkToRightCounterPlateL2 | <img src="docs/images/tasks/PnPSinkToRightCounterPlateL2.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToRightCounterPlateL2_eval.png" width="200"/> |
|       | PnPSinkToCabinetL2 | <img src="docs/images/tasks/PnPSinkToCabinetL2.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToCabinetL2_eval.png" width="200"/> |
|       | CloseRightCabinetDoorL2 | <img src="docs/images/tasks/CloseRightCabinetDoorL2.png" width="200"/> | <img src="docs/images/tasks/CloseRightCabinetDoorL2_eval.png" width="200"/> |
|       | CloseLeftCabinetDoorL2 | <img src="docs/images/tasks/CloseLeftCabinetDoorL2.png" width="200"/> | <img src="docs/images/tasks/CloseLeftCabinetDoorL2_eval.png" width="200"/> |
| **L3 (Unseen Objects, Unseen Environment)** | CloseLeftCabinetDoorL3 | <img src="docs/images/tasks/CloseLeftCabinetDoorL3.png" width="200"/> | <img src="docs/images/tasks/CloseLeftCabinetDoorL3_eval.png" width="200"/> |
|       | PnPSinkToRightCounterPlateL3 | <img src="docs/images/tasks/PnPSinkToRightCounterPlateL3.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToRightCounterPlateL3_eval.png" width="200"/> |
|       | PnPSinkToMicrowaveTopL3 | <img src="docs/images/tasks/PnPSinkToMicrowaveTopL3.png" width="200"/> | <img src="docs/images/tasks/PnPSinkToMicrowaveTopL3_eval.png" width="200"/> |
|       | TurnOnFaucetL3 | <img src="docs/images/tasks/TurnOnFaucetL3.png" width="200"/> | <img src="docs/images/tasks/TurnOnFaucetL3_eval.png" width="200"/> |

---

## Installation
MimicDroid builds on RoboCasa and works across major platforms. The easiest way to set up is via Anaconda. Follow the steps below.

- **Set up conda environment**
```bash
conda create -c conda-forge -n mimicdroid python=3.10
conda activate mimicdroid
```

- **Clone and set up robosuite (use the master branch)**
```bash
git clone --branch=abs_robot https://github.com/ShahRutav/robosuite
cd robosuite
pip install -e .
cd ..
```

- **Clone and set up RoboCasa**
```bash
git clone --branch=latest https://github.com/UT-Austin-RPL/mimicdroid-robocasa
cd robocasa
pip install -e .
pip install pre-commit; pre-commit install    # Optional: code formatter
```

- **(Optional) If you run into numba/numpy issues**
```bash
conda install -c numba numba=0.56.4 -y
```

- **Install the package and download assets**
```bash
python robocasa/scripts/download_kitchen_assets.py   # ~5GB download
python robocasa/scripts/setup_macros.py              # Set up system variables
```
---
 
## Dataset
Please see [DATASET.md](DATASET.md) for dataset installation and visualization instructions.
 
## Citation
```bibtex
@inproceedings{mimicdroid2025,
}

@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```