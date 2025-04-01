#!/usr/bin/env python3

from __future__ import annotations

import sys
import subprocess
import requests
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterable

# Energy and length conversion to atomic units.
ENERGY_CONVERSION = {"eV": 27.21138625}
LENGTH_CONVERSION = {"Ang": 0.529177210903}


# ----------------------------------------------------------------------------------------------------------------------
# Common functions: these are duplicated in all scripts to make them self-contained


def strip_comments(s: str) -> str:
    """Strip comment starting with '#' and continuing until the end of the string. Also strip whitespace."""
    return s.split("#")[0].strip()


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


def read_input(inpfile: str | Path) -> tuple[str, int, int, int, bool]:
    """Read the ORCA-generated input file

    Parameters
    ----------
    inpfile : str | Path
        The input file

    Returns
    -------
    tuple[str, int, int, int, bool]
        xyzname: str
            Name of the XYZ coordinates file
        charge: int
            Total charge
        mult: int
            Spin multiplicity
        ncores: int
            Number of parallel cores available
        dograd: bool
            Whether to compute the gradient
    """
    inpfile = enforce_path_object(inpfile)
    with inpfile.open() as f:
        xyzname = strip_comments(f.readline())
        charge = int(strip_comments(f.readline()))
        mult = int(strip_comments(f.readline()))
        ncores = int(strip_comments(f.readline()))
        dograd = bool(int(strip_comments(f.readline())))
        # TODO POINT CHARGES
    return xyzname, charge, mult, ncores, dograd


def write_engrad(
    outfile: str | Path,
    natoms: int,
    energy: float,
    dograd: bool,
    gradient: Iterable[float] = None,
) -> None:
    """Write the energy/gradient file to feed back to ORCA.

    Parameters
    ----------
    outfile : str | Path
        The engrad file
    natoms : int
        Number of atoms
    energy : float
        Total energy
    dograd : bool
        Whether the gradient is computed
    gradient : Iterable[float], optional
        The gradient (X,Y,Z) for each atom
    """
    outfile = enforce_path_object(outfile)
    with outfile.open("w") as f:
        output = "#\n"
        output += "# Number of atoms\n"
        output += "#\n"
        output += f"{natoms}\n"
        output += "#\n"
        output += "# Total energy [Eh]\n"
        output += "#\n"
        output += f"{energy:.12e}\n"
        if dograd:
            output += "#\n"
            output += "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...\n"
            output += "#\n"
            output += "\n".join(f"{g: .12e}" for g in gradient) + "\n"
        f.write(output)


def run_command(command: str | Path, outname: str | Path, *args: tuple[str, ...]) -> None:
    """
    Run the given command and redirect its STDOUT and STDERR to a file. Exists on a non-zero return code.

    Parameters
    ----------
    command : str | Path
        The command to run or path to an executable
    outname : str | Path
        The output file to be written to (overwritten!)
    args : tuple[str, ...]
        arguments to be passed to the command
    """
    command = enforce_path_object(command)
    outname = enforce_path_object(outname)
    with outname.open("w") as of:
        try:
            subprocess.run(
                [command] + list(args), stdout=of, stderr=subprocess.STDOUT, check=True
            )
        except subprocess.CalledProcessError as err:
            print(err)
            exit(err.returncode)


def clean_output(outfile: str | Path, namespace: str) -> None:
    """
    Print the output file to STDOUT and remove all files starting with `namespace`

    Parameters
    ----------
    outfile : str | Path
        The output file to print
    namespace : str
        The starting string of all files to remove.
    """
    # print the output to STDOUT
    outfile = enforce_path_object(outfile)
    with outfile.open() as f:
        for line in f:  # line by line to avoid memory overflow
            print(line, end="")
    # remove all file from the namespace
    for f in Path(".").glob(namespace + "*"):
        f.unlink()


# ----------------------------------------------------------------------------------------------------------------------


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

def run_aimnet2(
    atom_types: list[str],
    coordinates: list[tuple[float, float, float]],
    dograd: bool,
    model: str = "aimnet2_wb97m_0.jpt",
    charge: int = 0,
    mult: int = 1,
) -> tuple[float, list[float]]:
    """
    Runs an AIMNet2 calculation on a flask server.

    Parameters
    ----------
    atom_types : list[str]
        List of element symbols (e.g., ["O", "H", "H"])
    coordinates : list[tuple[float, float, float]]
        List of (x, y, z) coordinates
    dograd : bool
        Whether to compute the gradient
    model : str, optional
        The AIMNet2 model to use (default is "aimnet2_wb97m_0.jpt")
    charge : int, optional
        Molecular charge (default is 0)
    mult : int, optional
        Spin multiplicity (default is 1)

    Returns
    -------
    tuple[float, list[float]]
        energy : float
            The computed energy
        gradient : list[float]
            The gradient (X,Y,Z) for each atom
    """

    payload = {
       "atom_types": atom_types,
       "coordinates": coordinates,
       "dograd": dograd,
       "charge": charge,
       "mult": mult,
    }
    response = requests.post("http://localhost:5000/calculate", json=payload)
    response.raise_for_status()
    data = response.json()
    energy = data["energy"]
    gradient = data["gradient"]
    return energy, gradient

def main(argv: list[str]):
    """Main function to run the AIMNet2 calculation using a specified model file."""
    parser = ArgumentParser(
        prog=argv[0],
        description="Run AIMNet2 calculation with a user-specified model file.")
    parser.add_argument("inputfile", help="ORCA-generated input file.")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="aimnet2_wb97m_0.jpt",
        help="Path to the AIMNet2 model file (default: 'aimnet2_wb97m_0.jpt').")
    args = parser.parse_args(argv[1:])

    # read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)

    # set filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # process the XYZ file
    atom_types, coordinates = xyz2xsf(xyzname)
    natoms = len(atom_types)
    # run aimnet2 calculator
    energy, gradient = run_aimnet2(atom_types=atom_types, coordinates=coordinates, dograd=True, model=args.model, charge=charge, mult=mult)
    # convert to ORCA engrad
    write_engrad(orca_engrad, natoms, energy, dograd, gradient)
    # pipe the output to STDOUT and remove the generated files


if __name__ == "__main__":
    main(sys.argv)
