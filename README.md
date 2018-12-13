# BINet: Multivariate Business Process Anomaly Detection Using Deep Learning

## Setup
The easiest way to setup an environment is to use Miniconda.

### Using Miniconda
1. Install [Miniconda](https://conda.io/miniconda.html) (make sure to use a Python 3 version)
2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)
3. We suggest that you set up a dedicated environment for this project by running `conda env create -f environment.yml`
    * This will setup a virtual conda environment with all necessary dependencies.
    * If your device does not have a GPU replace `tensorflow-gpu` with `tensorflow` in the `environement.yml`
4. Depending on your operating system you can activate the virtual environment by running `source activate ad` 
on Linux and macOS, and `activate ad` on Windows (`cmd` only).
5. If you want to make use of a GPU, you must install the CUDA Toolkit. To install the CUDA Toolkit on your computer refer to the [TensorFlow installation guide](https://www.tensorflow.org/install/install_windows).

### Generating Event Logs
You can use the files in the `examples` directory to generate example event logs. Refer to `generate_event_logs.py`.

### Training BInet
Use the `train.py` in the `examples` directory for reference.