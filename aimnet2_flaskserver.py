#!/usr/bin/env python3

from __future__ import annotations

from flask import Flask, request, jsonify
import numpy as np
from aimnet2calc import AIMNet2Calculator

app = Flask(__name__)

# Energy and length conversion to atomic units.
ENERGY_CONVERSION = {"eV": 27.21138625}
LENGTH_CONVERSION = {"Ang": 0.529177210903}

model = "/home/U1040765/Desktop/aimnet2_wb97m_0.jpt"
calc = AIMNet2Calculator(model=model)
print("Model loaded.")

def atomic_symbol_to_number(symbol: str) -> int:
    element_to_atomic_number = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "As": 33, "Se": 34, "Br": 35,
    "I": 53}
    # Afaik these are the only elements covered by AIMNet2
    if symbol not in element_to_atomic_number:
        raise ValueError(f"Unknown element symbol: {symbol}")
    return element_to_atomic_number[symbol]

@app.route('/calculate', methods=['POST'])
def run_aimnet2():
    """
    Runs an AIMNet2 calculation.
    Expects a JSON payload with keys:
      - atom_types: list[str]
      - coordinates: list of (x, y, z) lists
      - dograd: bool
      - charge: int (optional)
      - mult: int (optional)
    Returns JSON with energy and gradient.
    """
    data = request.get_json()
    atom_types = data["atom_types"]
    coordinates = data["coordinates"]
    dograd = data.get("dograd", False)
    charge = data.get("charge", 0)
    mult = data.get("mult", 1)

    numbers = [atomic_symbol_to_number(sym) for sym in atom_types]
    inputs = {
        "coord": np.array([coordinates]),
        "numbers": np.array([numbers]),
        "charge": np.array([charge]),
        "mult": np.array([mult]),
    }

    result = calc(inputs, forces=dograd, stress=False, hessian=False)
    energy_tensor = result["energy"]
    energy = float(energy_tensor) / ENERGY_CONVERSION["eV"]
    gradient = []
    if dograd:
        fac = LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
        forces = np.asarray(result["forces"])
        gradient = (-forces * fac).flatten().tolist()

    return jsonify({'energy': energy, 'gradient': gradient})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)