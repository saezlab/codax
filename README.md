# CODAX: Differentiable cellular logic-ODEs accelerated with JAX
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12eRdpoDZXyxHH7HsrChG22zJEZIF9jnW)

Neural Logic ODEs for Signaling Network Inference using JAX+Diffrax

> NOTE: This is an experimental version of CNORode for Logic-ODEs that are trained on data using JAX implicit ODE solvers with Diffrax. It supports the load of experimental setups in MIDAS format using the CNO python API https://github.com/cellnopt/cellnopt.

## Installation

The library requires JAX and Jaxlib. It is advised to use `conda` to create an environment and install all the dependencies there. Here is an example of the environment used to run the examples of the repository:

```
conda create -n codax python=3.8 && conda activate codax
conda install jax=0.4.13 -c conda-forge
conda install pygraphviz=1.9 jupyter
pip install diffrax==0.3.1
pip install optax==0.1.4
pip install git+https://github.com/saezlab/sympy2jax.git
pip install git+https://github.com/saezlab/codax.git

```

## Benchmark

We compared the performance of training a Logic-ODE signaling model using CODAX and the old CellNoptR+CNORode. We benchmarked CODAX with JAX 0.3.25 and Diffrax 0.2.2, using the Heun's method with constant step size for the ODE integrator. For CNORode, we used the eSS method with Dynamic Hill Climbing (recommended settings) from [MEIGOR](https://www.bioconductor.org/packages/release/bioc/html/MEIGOR.html) 0.99, as described in [CNORode vignette](https://www.bioconductor.org/packages/release/bioc/vignettes/CNORode/inst/doc/CNORode-vignette.pdf). We tested both methods on the [toyMSB2009 dataset](https://github.com/saezlab/codax/tree/main/codax/nn_cno/datasets/wcs_benchmark):

![benchmark](https://github.com/saezlab/permedcoe/raw/master/experiments/codax_vs_cno/comparison_cnorode_codax.png)

## Acknowledgements 

CODAX is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773.

<img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto"> <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto"> <img src="https://www.klinikum.uni-heidelberg.de/typo3conf/ext/site_ukhd/Resources/Public/Images/Logo_ukhd_de.svg" alt="UKHD logo" height="64px" style="height:64px; width:auto">  
