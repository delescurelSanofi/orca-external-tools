
import time

start_import = time.perf_counter()

import sys
import subprocess
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable
from aimnet2calc import AIMNet2Calculator

end_import = time.perf_counter()
print("Total import time: {:.4f} seconds".format(end_import - start_import))

# Energy and length conversion to atomic units.
ENERGY_CONVERSION = {"eV": 27.21138625}
LENGTH_CONVERSION = {"Ang": 0.529177210903}

def enforce_path_object(fname: str | Path) -> Path:
    """Enforce that the input is a Path object

    Parameters
    ----------
    fname : str | Path
        The filename which should be a string or a Path object

    Returns
    -------
    Path
        The filename as a Path object

    Raises
    ------
    TypeError
        If the input is not a string or a Path object (e.g. a list)
    """
    if isinstance(fname, str):
        return Path(fname)
    elif isinstance(fname, Path):
        return fname
    else:
        msg = "Input must be a string or a Path object."
        raise TypeError(msg)
    
def xyz2xsf(xyzname: str | Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read an XYZ file and return the atom types and coordinates.

    Parameters
    ----------
    xyzname : str | Path
        The XYZ file to read.

    Returns
    -------
    tuple[list[str], list[tuple[float, float, float]]]
        atom_types: list[str]
            A list of element symbols in order.
        coordinates: list[tuple[float, float, float]]
            A list of (x, y, z) coordinates.
    """
    atom_types = []
    coordinates = []
    xyzname = enforce_path_object(xyzname)
    with xyzname.open() as xyzf:
        natoms = int(xyzf.readline().strip())
        xyzf.readline()
        for _ in range(natoms):
            line = xyzf.readline()
            if not line:
                break
            parts = line.split()
            atom_types.append(parts[0])
            coords = tuple(float(c) for c in parts[1:4])
            coordinates.append(coords)
    return atom_types, coordinates

atom_types, coordinates = xyz2xsf("water_test.xyz")
print(atom_types)
print(coordinates)

def atomic_symbol_to_number(symbol: str) -> int:
    element_to_atomic_number = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "As": 33, "Se": 34, "Br": 35,
    "I": 53,
}
    # Afaik these are the only elements covered by AIMNet2
    if symbol not in element_to_atomic_number:
        raise ValueError(f"Unknown element symbol: {symbol}")
    return element_to_atomic_number[symbol]

def run_aimnet2(
    atom_types: list[str],
    coordinates: list[tuple[float, float, float]],
    dograd: bool,
    model: str = "aimnet2_wb97m_0.jpt",
    charge: int = 0,
    mult: int = 1,
) -> tuple[float, list[float]]:
    """
    Runs an AIMNet2 calculation.

    Parameters
    ----------
    atom_types : list[str]
        List of element symbols (e.g., ["O", "H", "H"]).
    coordinates : list[tuple[float, float, float]]
        List of (x, y, z) coordinates.
    dograd : bool
        Whether to compute the gradient.
    model : str, optional
        The AIMNet2 model to use (default is "aimnet2_wb97m_0.jpt").
    charge : int, optional
        Molecular charge (default is 0).
    mult : int, optional
        Spin multiplicity (default is 1).

    Returns
    -------
    tuple[float, list[float]]
        energy : float
            The computed energy.
        gradient : list[float]
            Flattened gradient vector (if computed), otherwise empty.
    """

    numbers = [atomic_symbol_to_number(sym) for sym in atom_types]

    inputs = {
        "coord": np.array([coordinates]),
        "numbers": np.array([numbers]),
        "charge": np.array([charge]),
        "mult": np.array([mult]),
    }

    start_init = time.perf_counter()
    calc = AIMNet2Calculator(model=model)
    end_init = time.perf_counter()

    start_run = time.perf_counter()
    result = calc(inputs, forces=dograd, stress=False, hessian=False)
    end_run = time.perf_counter()
    energy_tensor = result["energy"]
    energy = float(energy_tensor)/ ENERGY_CONVERSION["eV"]
    gradient = []
    if dograd:
        fac = LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
        forces = np.asarray(result["forces"])
        gradient = (-forces * fac).flatten().tolist()
    
    init_time = end_init - start_init
    run_time = end_run - start_run
    print("Initialization time: {:.4f} seconds".format(init_time))
    print("Run time: {:.4f} seconds".format(run_time))

    return energy, gradient

energy, gradient = run_aimnet2(atom_types=atom_types, coordinates=coordinates, dograd=True)
print(energy)
print(gradient)