from pyscf import gto

from ferminet_ecp import base_config

def get_config(input_str):
    X, Y, dist, unit, spin = input_str.split(',')

    # Get default options.
    cfg = base_config.default()

    mol = gto.Mole()

    # Set up molecule
    mol.build(
        atom=f'{X} 0 0 0; {Y} 0 0 {dist}',
        basis={X: 'ccecpccpvdz', Y: 'ccecpccpvdz'},
        ecp={X: 'ccecp', Y: ' ccecp'},
        spin=int(spin), unit=unit)

    cfg.system.pyscf_mol = mol
    return cfg
