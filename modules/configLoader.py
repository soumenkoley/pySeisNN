#!/usr/bin/env python
# coding: utf-8

import configparser
import numpy as np

class Config:
    def __init__(self, path):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str  # preserve case
        cfg.read(path)

        def arr(key):
            return np.array([float(v) for v in key.split(',') if v.strip() != ''])

        # Parse and store everything as attributes
        self.thickness = arr(cfg['model']['thickness'])
        self.vP = arr(cfg['model']['vp'])
        self.vS = arr(cfg['model']['vs'])
        self.rho = arr(cfg['model']['rho'])
        self.qP = arr(cfg['model']['qp'])
        self.qS = arr(cfg['model']['qs'])
        self.rhoAir = float(cfg['model']['rhoAir'])

        self.fMin = float(cfg['frequency']['fmin'])
        self.fMax = float(cfg['frequency']['fmax'])
        self.df = float(cfg['frequency']['df'])
        
        self.lambdaFrac = float(cfg['grid']['lambda_frac'])
        self.lambdaRes = int(cfg['grid']['lambda_res'])
        
        self.xMaxGF = float(cfg['geometry']['xmax_gf'])
        self.zMaxGF = float(cfg['geometry']['zmax_gf'])
        self.maxRec = int(cfg['geometry']['max_receivers'])
        self.xCenter = float(cfg['geometry']['x_center'])
        self.yCenter = float(cfg['geometry']['y_center'])
        self.xExtent = float(cfg['geometry']['x_extent'])
        self.yExtent = float(cfg['geometry']['y_extent'])
        self.zExtent = float(cfg['geometry']['z_extent'])
        self.tMax = float(cfg['geometry']['t_max'])
        self.nSamp = int(cfg['geometry']['n_samp'])
        self.cavityType = str(cfg['geometry']['cavity_type'])

        self.itmAC = arr(cfg['tmPosition']['itm_ac'])
        self.itmAB = arr(cfg['tmPosition']['itm_ab'])
        self.etmCA = arr(cfg['tmPosition']['etm_ca'])
        self.etmBA = arr(cfg['tmPosition']['etm_ba'])

        self.nRea = int(cfg['simSource']['n_rea'])
        self.R1 = float(cfg['simSource']['r_1'])
        self.R2 = float(cfg['simSource']['r_2'])
        self.nSrc = int(cfg['simSource']['n_src'])
        self.srcDistri = str(cfg['simSource']['src_distri'])
        self.scaleVH = float(cfg['simSource']['scale_vh'])
        
        self.templateFile = cfg['paths']['template_file']
        self.inputPath = cfg['paths']['input_folder']
        self.qseisExe = cfg['paths']['qseis_exe']
        self.outDispPath = cfg['paths']['out_disp_folder']
        self.outDispPathRea = cfg['paths']['out_disp_rea']
        self.delFlag = cfg['paths']['qseis_out_del']
        self.siteASDPath = cfg['paths']['site_asd_path']
        self.etdPath = cfg['paths']['et_d_path']

        self.ifBody = cfg.getboolean('simFlags', 'if_body')
        self.ifRay = cfg.getboolean('simFlags','if_ray')
        self.ifFullField = cfg.getboolean('simFlags','if_full_field')
        self.saveHV = cfg.getboolean('simFlags','save_hv')
        self.decoupledHV = cfg.getboolean('simFlags','decoupled_hv')
        self.onFlyNN = cfg.getboolean('simFlags','on_fly_nn')

        self.cpuCoresQseis = int(cfg['compute']['compute_cores_qseis'])
        self.cpuCoresDisp = int(cfg['compute']['compute_cores_disp'])
        self.cpuCoresNN = int(cfg['compute']['compute_cores_nn'])
        self.computeStrategy = str(cfg['compute']['compute_strategy'])

# Global config instance
CONFIG = None

def load_config(path="configParse.ini"):
    global CONFIG
    CONFIG = Config(path)
    return CONFIG

def loadCavities(filename):
    config = configparser.ConfigParser()
    config.optionxform = str   # preserve key case
    config.read(filename)

    allCavities = []

    for section in config.sections():
        cav = {"name": section}

        for key, value in config[section].items():
            if key == "shape":
                cav[key] = value
            else:
                try:
                    cav[key] = float(value)
                except ValueError:
                    cav[key] = value

        shape = cav["shape"].lower()

        if shape == "cuboid":
            required = ["xC", "yC", "zC", "length", "breadth", "height", "angleDeg","xRefineFactor","yRefineFactor","zRefineFactor"]
        elif shape == "sphere":
            required = ["xC", "yC", "zC", "radius","xRefineFactor","yRefineFactor","zRefineFactor"]
        else:
            raise ValueError(f"Unsupported shape in section {section}: {cav['shape']}")

        for key in required:
            if key not in cav:
                raise ValueError(f"Section {section} missing required key '{key}'")

        allCavities.append(cav)

    return allCavities