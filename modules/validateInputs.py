#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from modules import configLoader

def main():
    configFile = "configParse.ini";
    config = configLoader.load_config(configFile);
    validateInputs(config)
    
def validateInputs(config):
    errors = []
    
    # Velocity model consistency
    n_layers = len(config.thickness)
    for arr, name in [(config.vP, 'vP'), (config.vS, 'vS'), (config.rho, 'rho')]:
        if len(arr) != n_layers:
            errors.append(f"Length mismatch: {name} has {len(arr)}, expected {n_layers}.")
    if config.thickness[-1] != 0.0:
        errors.append("Last layer thickness must be 0.0 (half-space).")
    if np.any(config.vP <= config.vS):
        errors.append("Some layers have vP <= vS (physically invalid).")
    if np.any(config.rho <= 0):
        errors.append("Some densities are non-positive.")
    
    # Frequency band
    if config.fMin <= 0 or config.fMax <= 0 or config.fMin >= config.fMax:
        errors.append(f"Invalid frequency range: fMin={config.fMin}, fMax={config.fMax}.")
    
    # Resolution and geometry
    if config.lambdaRes < 4:
        errors.append("lambdaRes must be >= 4 for stable grid resolution.")
    if config.xMaxGF <= 0 or config.zMaxGF <= 0:
        errors.append("xMaxGF and zMaxGF must be positive.")
    if config.maxRec <= 0:
        errors.append("maxRec must be > 0.")
    
    # File paths
    if not os.path.isfile(config.templateFile):
        errors.append(f"Template file not found: {config.templateFile}")
    if not os.path.isdir(config.inputPath):
        errors.append(f"Missing input directory: {config.inputPath}")
    if not os.path.isdir(config.outDispPath):
        errors.append(f"Missing output directory: {config.outDispPath}")
    if not os.path.isdir(config.outDispPathRea):
        errors.append(f"Missing output directory: {config.outDispPathRea}")
    if not os.path.isfile(config.qseisExe):
        errors.append(f"QSEIS executable not found: {config.qseisExe}")
    elif not os.access(config.qseisExe, os.X_OK):
        errors.append(f"QSEIS executable is not marked executable: {config.qseisExe}")

    # flags
    if(config.ifRay):
        if(config.ifBody):
            errors.append(f"ifRay and ifBody both cannot be set to True")
        elif(config.ifFullField):
            errors.append(f"ifRay and ifFullField both cannot be set to True")
    else:
        if config.ifBody and config.ifFullField:
            errors.append(f"ifBody and ifFullField both cannot be set to True")
    
    # Summary
    if errors:
        print("\n Configuration check failed:")
        for e in errors:
            print("  -", e)
        raise ValueError("Invalid QSEIS configuration.")
    else:
        print("All configuration and model checks passed.");

if __name__ == "__main__":
    main()