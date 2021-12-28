# Fermionic Neural Network with Effective Core Potential

An implementation combining FermiNet with effective core potential (ecp). For paper, see 
https://arxiv.org/abs/2108.11661.

This repository directly depends on [FermiNet](https://github.com/deepmind/ferminet/tree/jax) (Many thanks to this awesome
software and the team behind it!). Certain files are
modified from the corresponding ones in [FermiNet](https://github.com/deepmind/ferminet/tree/jax), and we added comments
prefixed with "MODIFICATION FROM FERMINET" on the introduced changes.

## Installation

`pip install -e .` will install all required dependencies. 

## Usage

Workflow of ferminet_ecp is similar to the original FermiNet, which uses the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to configure the
system. A few example scripts are included under `ferminet_ecp/configs/ecp`. 

```
ferminet_ecp --config ferminet_ecp/configs/ecp/X.py:Ga,1 --config.batch_size 256 --config.pretrain.iterations 100
```
To use ECP for atoms, define ECP-related fields in the corresponding config file. For instance,
```python
from pyscf import gto
from ferminet_ecp import base_config

def get_config(input_str):
    symbol, spin = input_str.split(',')
    cfg = base_config.default()
    mol = gto.Mole()
    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=int(spin))

    cfg.system.pyscf_mol = mol
    return cfg
```
Moreover, we want to mention that we remove local energy outlier in training phase via config.optim.rm_outlier flag, 
which violates variational principle and needs to be turned off in inference phase.

Our experiments were carried out with jax==0.2.12 and jaxlib==0.1.65+cuda102. We
hit some cuda issues with cuda 11, especially when training with KFAC.


## Giving Credit

If you use this code in your work, please cite the associated paper.

```
@misc{li2021fermionic,
      title={Fermionic Neural Network with Effective Core Potential}, 
      author={Xiang Li and Cunwei Fan and Weiluo Ren and Ji Chen},
      year={2021},
      eprint={2108.11661},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```
