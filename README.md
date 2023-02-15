# CODAX: Cellular logic-ODEs accelerated with JAX

Neural Logic ODEs for Signaling Network Inference using JAX+Diffrax

> NOTE: This is an experimental version of CNORode for Logic-ODEs that are trained on data using JAX implicit ODE solvers with Diffrax.

## Installation

The library requires JAX and Jaxlib. It is advised to use `conda` to create an environment and install all the dependencies there. Here is an example of the environment used to run the examples of the repository:

```
conda create -n codax python=3.8 && conda activate codax
conda install jax==0.3.25 -c conda-forge
conda install pandas==1.5.2 sympy==1.11.1 matplotlib==3.6.2 networkx==2.8.4 pygraphviz==1.9 jupyter
pip install diffrax==0.2.2 optax==0.1.4 easydev==0.12.1 colormap==1.0.4 wrapt==1.14.1 biokit==0.5.0
pip install git+https://github.com/saezlab/sympy2jax.git
pip install git+https://github.com/saezlab/codax.git
```

## Acknowledgements 

CODAX is developed at the [Institute for Computational Biomedicine](https://saezlab.org) (Heidelberg University). The development of this project is supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://permedcoe.eu/)) agreement no. 951773.

<img src="https://raw.githubusercontent.com/saezlab/.github/main/profile/logos/saezlab.png" alt="Saez lab logo" height="64px" style="height:64px; width:auto"> <img src="https://lcsb-biocore.github.io/COBREXA.jl/stable/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px" style="height:64px; width:auto"> <img src="https://www.klinikum.uni-heidelberg.de/typo3conf/ext/site_ukhd/Resources/Public/Images/Logo_ukhd_de.svg" alt="UKHD logo" height="64px" style="height:64px; width:auto">  
