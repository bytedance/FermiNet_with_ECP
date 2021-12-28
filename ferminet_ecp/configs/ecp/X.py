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
