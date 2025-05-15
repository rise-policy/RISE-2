# 🛠️ Installation

## 💻 Environments

Please follow the instructions to install the conda environments and the dependencies of the codebase. We recommend using CUDA 12.1 during installations to avoid compatibility issues. 

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n rise2 python=3.10
    conda activate rise2
    ```

2. Manually install cudatoolkit, then install necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. Install MinkowskiEngine. We have modified MinkowskiEngine for better adpatation.
    ```bash
    mkdir dependencies && cd dependencies
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/path/to/cuda
    git clone git@github.com:chenxi-wang/MinkowskiEngine.git
    cd MinkowskiEngine
    git checkout cuda-12-1
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas
    cd ../..
    ```

4. Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) manually.
    ```bash
    cd dependencies
    git clone git@github.com:facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install -e .
    cd ../..
    ```
5. Prepare the `weights` folder to store the pre-trained models.
    ```
    mkdir weights
    ```


## 🦾 Real Robot

We have tested the RISE-2 policy on both single-arm and dual-arm robot platforms. 

The single-arm platform consists of:
- Flexiv Rizon 4 Robotic Arm
- Dahuan AG-95 Gripper
- Intel RealSense D435 RGB-D Camera

The dual-arm platform consists of:
- Flexiv Rizon 4 Robotic Arms (Dual-Arm)
- Robotiq 2F-85 Grippers
- Intel RealSense D415 RGB-D Camera 

**Additional Dependencies** (tested on Ubuntu 20.04)
- If you are using Flexiv Rizon robotic arm, install the [Flexiv RDK](https://rdk.flexiv.com/manual/getting_started.html) to allow the remote control of the arm. Specifically, download [FlexivRDK v1.5.1](https://github.com/flexivrobotics/flexiv_rdk/releases/tag/v1.5.1) and copy `lib_py/flexivrdk.cpython-38-[arch].so` to the `device/` directory. Please specify `[arch]` according to your settings. For our platform, `[arch]` is `x86_64-linux-gnu`.
- If you are using Robotiq 2F-85 grippers, install the following python packages for communications.
  ```
  pip install pyserial==3.5
  ```
- If you are using Intel RealSense RGB-D camera, install the python wrapper `pyrealsense2` of `librealsense` according to [the official installation instructions](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation).