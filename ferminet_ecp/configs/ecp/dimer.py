from pyscf import gto

from ferminet_ecp import base_config

def get_config(input_str):
    symbol, dist, unit, spin = input_str.split(',')

    # Get default options.
    cfg = base_config.default()

    mol = gto.Mole()

    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0; {symbol} 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol:'ccecp'},
        spin=int(spin), unit=unit)

    cfg.system.pyscf_mol = mol
    return cfg