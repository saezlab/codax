# CODAX: Cellular logic-ODEs accelerated with JAX

Neural Logic ODEs for Signaling Network Inference using JAX+Diffrax

## Installation

The library requires JAX and Jaxlib. It is recommended to use `conda`: 

```
conda create -n codax python=3.8 && conda activate codax
conda install jax==0.3.25 -c conda-forge
conda install pandas==1.5.2 sympy==1.11.1 matplotlib==3.6.2 networkx==2.8.4 pygraphviz==1.9 jupyter
pip install diffrax==0.2.2 optax==0.1.4 easydev==0.12.1 colormap==1.0.4 wrapt==1.14.1 biokit==0.5.0
pip install git+https://github.com/saezlab/sympy2jax.git
pip install git+https://github.com/saezlab/codax.git
```

