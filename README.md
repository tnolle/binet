# BINet: Multi-perspective Business Process Anomaly Classification
This is the complementary repository for BINet, a neural network architecture for multi-perspective anomaly detection 
and classification in business process event logs.
BINet was originally proposed in [3] and then extended in [4].
The repository also contains implementations of all methods mentioned in [3, 4].
Specifically, it also contains the implementations for the DAE method from [1, 2].

All results can be reproduced using the notebooks in the `notebooks` directory.

## Additional Material
To illustrate the findings in [4], this repository contains Jupyter notebooks.
The notebooks are named according to the sections in the paper.
Notebooks with `A` in the name contain additional material which is not included in the papers.
The code to reproduce the figures in the paper can be found inside the notebooks.
All necessary files to reproduce the results are also included in the repository.

### Notebooks
1. Introduction
2. Related Work
3. Datasets
    * [3.1 Example Process](notebooks/3.1%20Example%20Process.ipynb)
    * [3.2 Dataset Information](notebooks/3.2%20Dataset%20Information.ipynb)
    * [3.A1 Generation Algorithm](notebooks/3.A1%20Generation%20Algorithm.ipynb)
        * Describes how the likelihood graph generation algorithm works and how it can be used.
    * [3.A2 Dataset Generation](notebooks/3.A2%20Dataset%20Generation.ipynb)
        * Generates the same data corpus as used in the paper.
    * [3.A3 BPIC Datasets](notebooks/3.A3%20BPIC%20Datasets.ipynb)
        * Adds artifial anomalies to the BPIC datasets. Result will be the same as the ones used in the paper.
4. Method
    * [4.1 Heuristics](notebooks/4.1%20Heuristics.ipynb)
    * [4.A1 Training](notebooks/4.A1%20Training.ipynb)
        * Will train and save the anomaly detection models as used in the paper. For non-deterministic anomaly detectors, 
        results might differ from the ones in the paper.
5. Evaluation
    * [5.1 Best Strategy](notebooks/5.1%20Best%20Strategy.ipynb)
    * [5.2 Best Heuristic](notebooks/5.2%20Best%20Heuristic.ipynb)
    * [5.3 Overall Evaluation](notebooks/5.3%20Evaluation.ipynb)
    * [5.A1 Evaluation Script](notebooks/5.A1%20Evaluation%20Script.ipynb)
        * Will evaluate all trained models and save the results to a SQLite database.
    * [5.A2 Additional Evaluations](notebooks/5.A2%20Additional%20Evaluations.ipynb)
        * Misc. evaluations, e.g., per perspective, per event attribute, etc.
    * [5.A3 ROC](notebooks/5.A3%20ROC.ipynb)
        * Analysis of ROC and AUC
    * [5.A3 Hyperparameters](notebooks/5.A4%20Hyperparameters.ipynb)
        * Test of different hyperparameters for BINet and t-STIDE+
6. Classifying Anomalies
    * [6. Classification](notebooks/6.%20Classification.ipynb)
        * Produces the heatmap visualization featured in the paper. 
        Additionally, demonstrates how to use the `plot_heatmap` method.
7. Conclusion

## Setup
The easiest way to setup an environment is to use Miniconda.

### Using Miniconda
1. Install [Miniconda](https://conda.io/miniconda.html) (make sure to use a Python 3 version)
2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)
3. We suggest that you set up a dedicated environment for this project by running `conda env create -f environment.yml`
    * This will setup a virtual conda environment with all necessary dependencies.
    * If your device does have a GPU replace `tensorflow` with `tensorflow-gpu` in the `environement.yml`
4. Depending on your operating system you can activate the virtual environment with `conda activate binet` 
on Linux and macOS, and `activate ad` on Windows (`cmd` only).
5. If you want to make use of a GPU, you must install the CUDA Toolkit. To install the CUDA Toolkit on your computer refer to the [TensorFlow installation guide](https://www.tensorflow.org/install/install_windows).
6. If you want to quickly install the `april` package, run `pip install -e .` inside the root directory.
7. Now you can start the notebook server by `jupyter notebook notebooks`.

Note: To use the graph plotting methods, you will have to install Graphviz.

## References
1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: Unsupervised anomaly detection in noisy business process event logs using 
    denoising autoencoders, 2016](https://link.springer.com/chapter/10.1007/978-3-319-46307-0_28)
2. [Nolle, T., Luettgen, S., Seeliger A., Mühlhäuser, M.: Analyzing business process anomalies using autoencoders, 
    2018](https://link.springer.com/article/10.1007%2Fs10994-018-5702-8)
3. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning,
    2018](https://link.springer.com/chapter/10.1007/978-3-319-98648-7_16)
4. [Nolle, T., Luettgen, S., Seeliger, A., Mühlhäuser, M.: BINet: Multi-perspective Business Process Anomaly Classification,
   2019, (under review)](https://arxiv.org/abs/1902.03155)
