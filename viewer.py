#!/usr/bin/env python3
"""
BiologyViewer - compact 3D biology structure atlas.

Examples:
    python viewer.py --item hemoglobin
    python viewer.py --item dna_double_helix
    python viewer.py --item glucose
    python viewer.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import struct
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pygame
from Bio.PDB import PDBParser
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, OPENGL, QUIT


WIN_W = 820
WIN_H = 820
AUTO_SPD = 0.20
SIDES = 16
STEPS = 12
CACHE_DIR = Path(__file__).with_name(".viewer_cache")
HELIX_COL = (0.90, 0.24, 0.24)
SHEET_COL = (0.95, 0.80, 0.18)
COIL_COL = (0.20, 0.76, 0.35)
CHAIN_COLORS = [
    (0.32, 0.68, 0.98),
    (0.98, 0.40, 0.64),
    (0.56, 0.88, 0.44),
    (0.98, 0.76, 0.28),
]
BG_COL = (0.03, 0.04, 0.07, 1.0)
CATEGORY_COLORS = {
    "Protein": (0.94, 0.42, 0.42),
    "Molecule": (0.40, 0.72, 0.98),
    "Nucleic Acid": (0.98, 0.78, 0.30),
    "Human Organ": (0.86, 0.56, 0.72),
}
ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[96m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "magenta": "\033[95m",
    "red": "\033[91m",
    "white": "\033[97m",
}
MESH_COLORS = {
    "liver": (0.70, 0.26, 0.22),
    "stomach": (0.83, 0.48, 0.60),
    "spleen": (0.62, 0.22, 0.34),
    "bladder": (0.86, 0.72, 0.30),
    "esophagus": (0.78, 0.66, 0.54),
    "trachea": (0.74, 0.84, 0.92),
    "gallbladder": (0.38, 0.70, 0.30),
    "kidneys": (0.62, 0.24, 0.24),
    "left_kidney": (0.62, 0.24, 0.24),
    "right_kidney": (0.62, 0.24, 0.24),
    "lungs": (0.88, 0.66, 0.72),
    "right_lung_upper_lobe": (0.88, 0.66, 0.72),
    "right_lung_middle_lobe": (0.88, 0.66, 0.72),
    "right_lung_lower_lobe": (0.88, 0.66, 0.72),
    "left_lung_upper_lobe": (0.88, 0.66, 0.72),
    "left_lung_lower_lobe": (0.88, 0.66, 0.72),
    "small_intestine": (0.88, 0.76, 0.52),
    "duodenum": (0.88, 0.76, 0.52),
    "jejunum": (0.88, 0.76, 0.52),
    "ileum": (0.88, 0.76, 0.52),
    "rectum": (0.76, 0.56, 0.44),
    "large_intestine": (0.78, 0.62, 0.42),
}
BOND_COLORS = {
    "peptide": (0.92, 0.92, 0.92),
    "hydrogen": (0.55, 0.82, 1.0),
    "disulfide": (0.98, 0.82, 0.18),
    "ionic": (0.95, 0.50, 0.20),
    "phosphodiester": (0.95, 0.58, 0.22),
    "glycosidic": (0.92, 0.72, 0.28),
    "hydrophobic": (0.62, 0.84, 0.50),
    "van_der_waals": (0.62, 0.62, 0.62),
    "covalent": (0.96, 0.96, 0.96),
}
ELEMENT_COLORS = {
    "C": (0.78, 0.78, 0.78),
    "N": (0.36, 0.56, 0.96),
    "O": (0.94, 0.28, 0.24),
    "S": (0.96, 0.84, 0.22),
    "P": (0.98, 0.62, 0.20),
    "H": (0.95, 0.95, 0.95),
    "CL": (0.22, 0.80, 0.32),
    "FE": (0.84, 0.46, 0.22),
    "MG": (0.52, 0.96, 0.52),
}
ELEMENT_RADII = {
    "C": 0.18,
    "N": 0.19,
    "O": 0.19,
    "S": 0.23,
    "P": 0.24,
    "CL": 0.23,
    "FE": 0.24,
    "MG": 0.23,
}
BOND_RADII = {
    "covalent": 0.070,
    "hydrogen": 0.035,
    "ionic": 0.050,
    "phosphodiester": 0.060,
}
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "FE": 1.24,
    "MG": 1.30,
    "ZN": 1.22,
    "CA": 1.76,
    "CL": 1.02,
}


@dataclass
class EntitySpec:
    key: str
    label: str
    category: str
    subtitle: str
    bonds: list[str]
    key_features: list[str]
    colors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    pdb_id: str | None = None
    mode: str = "protein"
    group_note: str = ""
    mesh_urls: list[str] = field(default_factory=list)
    compound_cid: int | None = None


def normalize_key(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


PROTEIN_GROUPS = {
    "enzymes": {
        "items": ["lysozyme", "pkzilla_1", "hexokinase", "catalase", "pepsin", "trypsin", "chymotrypsin", "amylase", "lipase", "ribonuclease", "atpase", "carbonic_anhydrase", "aldolase", "papain", "elastase", "thrombin", "adenylate_kinase", "alcohol_dehydrogenase", "dihydrofolate_reductase", "enolase", "fumarase", "glutamate_dehydrogenase", "glycogen_phosphorylase", "lactate_dehydrogenase", "malate_dehydrogenase", "phosphoglycerate_kinase", "pyruvate_kinase", "superoxide_dismutase", "acetylcholinesterase", "citrate_synthase", "dna_polymerase", "topoisomerase", "proteasome"],
        "subtitle": "Catalytic protein with a compact active-site core and folded substrate pocket.",
        "bonds": ["peptide", "hydrogen", "ionic", "hydrophobic", "van_der_waals"],
        "features": ["Active-site geometry controls specificity", "Backbone fold packed for catalytic efficiency", "Side-chain interactions stabilize the substrate pocket"],
        "colors": ((0.82, 0.28, 0.22), (0.95, 0.80, 0.18), (0.20, 0.72, 0.32)),
    },
    "structural_signaling": {
        "items": ["hemoglobin", "myoglobin", "actin", "collagen", "fibronectin", "calmodulin", "ubiquitin", "cytochrome_c", "ferritin", "albumin", "transferrin", "immunoglobulin", "rhodopsin", "histone", "insulin", "titin"],
        "subtitle": "Folded structural or signaling protein with shape-driven binding and support roles.",
        "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic", "van_der_waals"],
        "features": ["Compact tertiary packing preserves function", "Specific domains expose binding surfaces", "Secondary structure distribution defines overall shape"],
        "colors": ((0.78, 0.26, 0.36), (0.94, 0.76, 0.22), (0.25, 0.70, 0.62)),
    },
    "membrane_transport": {"items": ["aquaporin", "porin"], "subtitle": "Membrane-spanning protein with a compact channel or pore geometry.", "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic"], "features": ["Hydrophobic exterior matches the membrane", "Polar lining shapes transport selectivity", "Compact fold minimizes leakage"], "colors": ((0.28, 0.55, 0.82), (0.95, 0.80, 0.30), (0.18, 0.80, 0.58))},
    "nucleic_acid_machinery": {"items": ["helicase", "telomerase_rbd"], "subtitle": "Protein specialized for DNA or RNA interaction and controlled strand handling.", "bonds": ["peptide", "hydrogen", "ionic", "hydrophobic"], "features": ["Nucleic-acid binding residues cluster on one face", "ATP-linked conformational shifts coordinate action", "Folded domains clamp or guide strands"], "colors": ((0.30, 0.50, 0.86), (0.94, 0.70, 0.26), (0.28, 0.76, 0.44))},
    "disease_targets": {"items": ["prion", "hiv_protease", "p53", "ras", "bcl2", "brca1"], "subtitle": "Biomedically important protein where folding and binding interfaces control disease outcomes.", "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic", "disulfide"], "features": ["Key binding regions are often mutation-sensitive", "Compact domain organization controls regulatory behavior", "Misfolding or interface changes alter function strongly"], "colors": ((0.86, 0.30, 0.28), (0.92, 0.78, 0.24), (0.28, 0.64, 0.76))},
    "chaperones": {"items": ["hsp70", "chaperonin"], "subtitle": "Protein-folding helper with ATP-coupled conformational changes and stabilizing interfaces.", "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic"], "features": ["Large domain motions assist folding", "Hydrophobic patches transiently bind unfolded chains", "Compact chamber or clamp-like shape protects intermediates"], "colors": ((0.75, 0.36, 0.28), (0.96, 0.80, 0.30), (0.20, 0.78, 0.46))},
    "markers": {"items": ["gfp"], "subtitle": "Marker protein with a compact beta-barrel that protects its internal chromophore.", "bonds": ["peptide", "hydrogen", "hydrophobic"], "features": ["Beta barrel shields fluorophore chemistry", "Tight packing stabilizes fluorescence", "Compact fold is resilient in imaging workflows"], "colors": ((0.18, 0.78, 0.30), (0.92, 0.86, 0.20), (0.18, 0.56, 0.90))},
    "motor": {"items": ["myosin", "kinesin"], "subtitle": "Motor protein with an ATPase head and shape-tuned track binding surfaces.", "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic"], "features": ["Mechanical stroke depends on domain reorientation", "Compact heads convert ATP into movement", "Binding loops align with actin or microtubules"], "colors": ((0.84, 0.42, 0.24), (0.94, 0.74, 0.22), (0.26, 0.74, 0.72))},
    "viral": {"items": ["spike_protein", "neuraminidase"], "subtitle": "Viral protein shaped for host binding, membrane entry, or particle release.", "bonds": ["peptide", "hydrogen", "hydrophobic", "ionic", "glycosidic"], "features": ["Surface exposure supports recognition", "Compact domains assemble into higher-order viral architecture", "Interface chemistry controls infectivity"], "colors": ((0.80, 0.30, 0.34), (0.94, 0.74, 0.24), (0.24, 0.66, 0.86))},
    "antibiotic_targets": {"items": ["vancomycin_binding"], "subtitle": "Target structure where precise chemistry governs antimicrobial recognition and resistance.", "bonds": ["peptide", "hydrogen", "ionic", "van_der_waals"], "features": ["Local geometry determines drug fit", "Backbone and side-chain chemistry define selectivity", "Compact recognition surface supports tight binding"], "colors": ((0.82, 0.28, 0.44), (0.94, 0.78, 0.24), (0.26, 0.70, 0.54))},
}

PDB_IDS = {
    "lysozyme": "1LYZ", "pkzilla_1": "2FAE", "hexokinase": "2YHX", "catalase": "1QQW", "pepsin": "4AA9", "trypsin": "1S0Q", "chymotrypsin": "1YPH", "amylase": "1SMD", "lipase": "1TCA", "ribonuclease": "7RSA", "atpase": "1BMF", "carbonic_anhydrase": "1CA2", "aldolase": "4ALD", "papain": "9PAP", "elastase": "3EST", "thrombin": "1PPB", "adenylate_kinase": "4AKE", "alcohol_dehydrogenase": "1HLD", "dihydrofolate_reductase": "7DFR", "enolase": "2ONE", "fumarase": "1FUO", "glutamate_dehydrogenase": "1HWY", "glycogen_phosphorylase": "1NOI", "lactate_dehydrogenase": "1LDM", "malate_dehydrogenase": "4MDH", "phosphoglycerate_kinase": "3PGK", "pyruvate_kinase": "1A3W", "superoxide_dismutase": "2SOD", "acetylcholinesterase": "1ACJ", "citrate_synthase": "1CSH", "dna_polymerase": "1TAU", "topoisomerase": "1AB4", "proteasome": "1RYP", "hemoglobin": "1HHO", "myoglobin": "1MBN", "actin": "1ATN", "collagen": "1CGD", "fibronectin": "1FNF", "calmodulin": "1CLL", "ubiquitin": "1UBQ", "cytochrome_c": "1HRC", "ferritin": "2FHA", "albumin": "1AO6", "transferrin": "1D3K", "immunoglobulin": "1IGT", "rhodopsin": "1F88", "histone": "1AOI", "insulin": "1MSO", "titin": "1TIT", "aquaporin": "1FQY", "porin": "2POR", "helicase": "1FUQ", "telomerase_rbd": "3KYL", "prion": "1QLX", "hiv_protease": "3HVP", "p53": "2OCJ", "ras": "4Q21", "bcl2": "1G5M", "brca1": "1JM7", "hsp70": "1YUW", "chaperonin": "1AON", "gfp": "1GFL", "myosin": "1B7T", "kinesin": "1BG2", "spike_protein": "6VXX", "neuraminidase": "2BAT", "vancomycin_binding": "1FVM",
}

REAL_ORGAN_MESHES = {
    "liver": {
        "label": "Liver",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7197.stl"],
    },
    "stomach": {
        "label": "Stomach",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7148.stl"],
    },
    "spleen": {
        "label": "Spleen",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7196.stl"],
    },
    "bladder": {
        "label": "Urinary Bladder",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA15900.stl"],
    },
    "esophagus": {
        "label": "Esophagus",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7131.stl"],
    },
    "trachea": {
        "label": "Trachea",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7394.stl"],
    },
    "gallbladder": {
        "label": "Gallbladder",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7202.stl"],
    },
    "kidneys": {
        "label": "Kidneys",
        "urls": [
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7204.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7205.stl",
        ],
    },
    "lungs": {
        "label": "Lungs",
        "urls": [
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7333.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7383.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7337.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7370.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7371.stl",
        ],
    },
    "small_intestine": {
        "label": "Small Intestine",
        "urls": [
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7206.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7207.stl",
            "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7208.stl",
        ],
    },
    "left_kidney": {
        "label": "Left Kidney",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7205.stl"],
    },
    "right_kidney": {
        "label": "Right Kidney",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7204.stl"],
    },
    "duodenum": {
        "label": "Duodenum",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7206.stl"],
    },
    "jejunum": {
        "label": "Jejunum",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7207.stl"],
    },
    "ileum": {
        "label": "Ileum",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7208.stl"],
    },
    "rectum": {
        "label": "Rectum",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA14544.stl"],
    },
    "right_lung_upper_lobe": {
        "label": "Right Lung Upper Lobe",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7333.stl"],
    },
    "right_lung_middle_lobe": {
        "label": "Right Lung Middle Lobe",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7383.stl"],
    },
    "right_lung_lower_lobe": {
        "label": "Right Lung Lower Lobe",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7337.stl"],
    },
    "left_lung_upper_lobe": {
        "label": "Left Lung Upper Lobe",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7370.stl"],
    },
    "left_lung_lower_lobe": {
        "label": "Left Lung Lower Lobe",
        "urls": ["https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl/FMA7371.stl"],
    },
}

NUCLEIC_ACID_PDBS = {
    "dna_double_helix": {"label": "DNA Double Helix", "pdb_id": "1BNA"},
    "trna_rna": {"label": "tRNA", "pdb_id": "1EHZ"},
}

PUBCHEM_MOLECULES = {
    "glucose": {"label": "Glucose", "cid": 5793},
    "atp": {"label": "ATP", "cid": 5957},
    "adp": {"label": "ADP", "cid": 6022},
    "amp": {"label": "AMP", "cid": 6083},
    "cholesterol": {"label": "Cholesterol", "cid": 5997},
    "caffeine": {"label": "Caffeine", "cid": 2519},
    "aspirin": {"label": "Aspirin", "cid": 2244},
    "dopamine": {"label": "Dopamine", "cid": 681},
    "serotonin": {"label": "Serotonin", "cid": 5202},
    "histamine": {"label": "Histamine", "cid": 774},
    "glycine": {"label": "Glycine", "cid": 750},
    "alanine": {"label": "Alanine", "cid": 5950},
    "lactic_acid": {"label": "Lactic Acid", "cid": 612},
    "urea": {"label": "Urea", "cid": 1176},
    "citric_acid": {"label": "Citric Acid", "cid": 311},
    "creatine": {"label": "Creatine", "cid": 586},
    "acetylcholine": {"label": "Acetylcholine", "cid": 187},
    "palmitic_acid": {"label": "Palmitic Acid", "cid": 985},
    "fructose": {"label": "Fructose", "cid": 5984},
    "sucrose": {"label": "Sucrose", "cid": 5988},
}


def build_catalog() -> dict[str, EntitySpec]:
    catalog: dict[str, EntitySpec] = {}
    for group in PROTEIN_GROUPS.values():
        for item in group["items"]:
            label = item.replace("_", " ").title()
            catalog[item] = EntitySpec(
                key=item,
                label=label,
                category="Protein",
                subtitle=group["subtitle"],
                bonds=list(group["bonds"]),
                key_features=list(group["features"]),
                colors=group["colors"],
                pdb_id=PDB_IDS[item],
                mode="protein",
                group_note=f"{label} is displayed with compact folding and bond-focused metadata.",
            )
    for key, organ in REAL_ORGAN_MESHES.items():
        catalog[key] = EntitySpec(
            key=key,
            label=organ["label"],
            category="Human Organ",
            subtitle="Source-backed anatomy mesh from BodyParts3D.",
            bonds=[],
            key_features=[],
            colors=((0.82, 0.55, 0.48), (0.70, 0.40, 0.34), (0.55, 0.74, 0.88)),
            mode="mesh",
            group_note="BodyParts3D anatomy mesh",
            mesh_urls=list(organ["urls"]),
        )
    for key, entry in NUCLEIC_ACID_PDBS.items():
        catalog[key] = EntitySpec(
            key=key,
            label=entry["label"],
            category="Nucleic Acid",
            subtitle="RCSB PDB all-atom nucleic-acid structure.",
            bonds=["phosphodiester", "hydrogen", "base_stacking"],
            key_features=[],
            colors=((0.30, 0.62, 0.96), (0.96, 0.34, 0.54), (0.96, 0.84, 0.42)),
            pdb_id=entry["pdb_id"],
            mode="nucleic",
            group_note="RCSB PDB nucleic-acid structure",
        )
    for key, entry in PUBCHEM_MOLECULES.items():
        catalog[key] = EntitySpec(
            key=key,
            label=entry["label"],
            category="Molecule",
            subtitle="PubChem 3D compound structure.",
            bonds=["covalent"],
            key_features=[],
            colors=((0.82, 0.82, 0.82), (0.36, 0.56, 0.96), (0.94, 0.28, 0.24)),
            mode="compound",
            group_note="PubChem 3D compound",
            compound_cid=entry["cid"],
        )
    return catalog


CATALOG = build_catalog()


def fetch_pdb(pdb_id: str, refresh: bool = False) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{pdb_id.upper()}.pdb"
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    req = urllib.request.Request(url, headers={"User-Agent": "BiologyViewer/2.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8", errors="ignore")
    cache_file.write_text(text, encoding="utf-8")
    return text


def fetch_binary_asset(url: str, refresh: bool = False) -> bytes:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / Path(url).name
    if cache_file.exists() and not refresh:
        return cache_file.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "BiologyViewer/2.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    cache_file.write_bytes(data)
    return data


def fetch_text_asset(url: str, refresh: bool = False) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    base_name = Path(url.split("?")[0]).name or "asset.txt"
    safe_name = f"{base_name}_{digest}.txt"
    cache_file = CACHE_DIR / safe_name
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": "BiologyViewer/2.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8", errors="ignore")
    cache_file.write_text(text, encoding="utf-8")
    return text


def fetch_pubchem_sdf(cid: int, refresh: bool = False) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF?record_type=3d"
    return fetch_text_asset(url, refresh=refresh)


def load_binary_stl(data: bytes, max_faces: int = 35000) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 84:
        raise ValueError("STL file is too small")
    face_count = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("v3", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )
    faces = np.frombuffer(data, dtype=dtype, count=face_count, offset=84)
    if len(faces) > max_faces:
        step = math.ceil(len(faces) / max_faces)
        faces = faces[::step]
    triangles = np.stack([faces["v1"], faces["v2"], faces["v3"]], axis=1).astype(np.float32)
    normals = faces["normal"].astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1)
    bad = lengths < 1e-6
    if np.any(bad):
        edge_a = triangles[bad, 1] - triangles[bad, 0]
        edge_b = triangles[bad, 2] - triangles[bad, 0]
        fixed = np.cross(edge_a, edge_b)
        fixed_len = np.linalg.norm(fixed, axis=1, keepdims=True)
        fixed_len[fixed_len < 1e-6] = 1.0
        normals[bad] = fixed / fixed_len
    else:
        normals /= lengths[:, None]
    return triangles, normals


def normalize_mesh(triangles: np.ndarray) -> np.ndarray:
    vertices = triangles.reshape(-1, 3)
    centered, _, _ = center_scale_points(vertices, target_radius=4.0)
    return centered.reshape(triangles.shape)


def parse_sdf(sdf_text: str) -> tuple[list[dict], list[dict]]:
    lines = sdf_text.splitlines()
    if len(lines) < 4:
        raise ValueError("SDF content is incomplete")
    counts = lines[3]
    atom_count = int(counts[0:3])
    bond_count = int(counts[3:6])
    atoms: list[dict] = []
    bonds: list[dict] = []
    for idx in range(4, 4 + atom_count):
        line = lines[idx]
        x = float(line[0:10])
        y = float(line[10:20])
        z = float(line[20:30])
        element = line[31:34].strip().upper()
        atoms.append({"pos": np.array([x, y, z], dtype=float), "element": element, "chain": "L", "resnum": 1})
    for idx in range(4 + atom_count, 4 + atom_count + bond_count):
        line = lines[idx]
        a = int(line[0:3]) - 1
        b = int(line[3:6]) - 1
        order = int(line[6:9])
        bonds.append({"a_idx": a, "b_idx": b, "kind": "covalent", "order": order})
    return atoms, bonds


def parse_atom_structure(pdb_text: str, atom_limit: int = 900) -> list[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", io.StringIO(pdb_text))
    atoms: list[dict] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = (atom.element or atom.get_name()[0]).strip().upper()
                    if element in {"", "H"}:
                        continue
                    atoms.append({"pos": np.array(atom.coord, dtype=float), "element": element, "chain": chain.id, "resnum": residue.id[1]})
                    if len(atoms) >= atom_limit:
                        return atoms
        break
    return atoms


def normalize_atoms(atoms: list[dict], target_radius: float = 4.0) -> list[dict]:
    points = np.array([atom["pos"] for atom in atoms], dtype=float)
    scaled, _, _ = center_scale_points(points, target_radius=target_radius)
    out = []
    for atom, pos in zip(atoms, scaled):
        updated = dict(atom)
        updated["pos"] = pos
        out.append(updated)
    return out


def parse_nucleic_backbone(pdb_text: str) -> dict[str, list[dict]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("nucleic", io.StringIO(pdb_text))
    chains: dict[str, list[dict]] = {}
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                atom_name = "P" if "P" in residue else "C4'"
                if atom_name not in residue:
                    continue
                pos = np.array(residue[atom_name].coord, dtype=float)
                base_atoms = []
                for atom in residue:
                    name = atom.get_name().strip().upper()
                    if name in {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}:
                        continue
                    if (atom.element or name[0]).strip().upper() == "H":
                        continue
                    base_atoms.append(np.array(atom.coord, dtype=float))
                base_center = np.mean(base_atoms, axis=0) if base_atoms else pos
                residues.append({"pos": pos, "base_center": base_center, "resnum": residue.id[1]})
            if len(residues) >= 3:
                chains[chain.id] = residues
        break
    if not chains:
        raise ValueError("no nucleic backbone residues parsed")
    all_points = np.array([entry["pos"] for residues in chains.values() for entry in residues], dtype=float)
    scaled_points, centroid, scale = center_scale_points(all_points, target_radius=4.6)
    idx = 0
    normalized: dict[str, list[dict]] = {}
    for chain_id, residues in chains.items():
        items = []
        for residue in residues:
            entry = dict(residue)
            entry["pos"] = scaled_points[idx]
            entry["base_center"] = (residue["base_center"] - centroid) * scale
            items.append(entry)
            idx += 1
        normalized[chain_id] = items
    return normalized


def build_nucleic_geometry(chains: dict[str, list[dict]]) -> tuple[list[tuple[list, tuple[float, float, float]]], list[dict]]:
    chain_items = sorted(chains.items(), key=lambda item: len(item[1]), reverse=True)
    render_data = []
    bridges: list[dict] = []
    for index, (chain_id, residues) in enumerate(chain_items):
        points = [entry["pos"] for entry in residues]
        spline = catmull_rom(points)
        color = CHAIN_COLORS[index % len(CHAIN_COLORS)]
        rings = compute_tube(spline, 0.20 if len(chain_items) > 1 else 0.18)
        if rings:
            render_data.append((rings, color))
    if len(chain_items) >= 2:
        left = chain_items[0][1]
        right = chain_items[1][1]
        pairs = min(len(left), len(right))
        for i in range(pairs):
            a = left[i]["base_center"]
            b = right[pairs - i - 1]["base_center"]
            if np.linalg.norm(a - b) <= 2.8:
                bridges.append({"a": a, "b": b, "kind": "hydrogen"})
    return render_data, bridges


def parse_secondary_structure(pdb_text: str):
    helix, sheet = set(), set()
    for line in pdb_text.splitlines():
        try:
            if line.startswith("HELIX "):
                chain = line[19]
                start, end = int(line[21:25]), int(line[33:37])
                for resnum in range(start, end + 1):
                    helix.add((chain, resnum))
            elif line.startswith("SHEET "):
                chain = line[21]
                start, end = int(line[22:26]), int(line[33:37])
                for resnum in range(start, end + 1):
                    sheet.add((chain, resnum))
        except (ValueError, IndexError):
            continue
    return helix, sheet


def parse_protein(pdb_text: str) -> tuple[dict[str, list[tuple[np.ndarray, str]]], list[dict]]:
    helix, sheet = parse_secondary_structure(pdb_text)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_text))
    chains: dict[str, list[tuple[np.ndarray, str]]] = {}
    atoms: list[dict] = []
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resnum = residue.id[1]
                if "CA" in residue:
                    pos = np.array(residue["CA"].coord, dtype=float)
                    ss = "H" if (chain.id, resnum) in helix else "S" if (chain.id, resnum) in sheet else "C"
                    residues.append((pos, ss))
                if len(atoms) < 350:
                    for atom in residue:
                        element = (atom.element or atom.get_name()[0]).strip().upper()
                        if element in {"", "H"}:
                            continue
                        atoms.append({"pos": np.array(atom.coord, dtype=float), "element": element, "chain": chain.id, "resnum": resnum})
                        if len(atoms) >= 350:
                            break
            if len(residues) >= 4:
                chains[chain.id] = residues
        break
    return chains, atoms


def classify_bond(atom_a: dict, atom_b: dict, distance: float) -> str:
    elements = {atom_a["element"], atom_b["element"]}
    if elements == {"S"} and distance < 2.3:
        return "disulfide"
    if "N" in elements and "O" in elements and distance < 3.2:
        return "hydrogen"
    if "P" in elements and "O" in elements:
        return "phosphodiester"
    if ("N" in elements or "O" in elements) and ("FE" in elements or "MG" in elements or "CA" in elements):
        return "ionic"
    return "covalent"


def estimate_bonds(atoms: list[dict], limit: int = 280) -> list[dict]:
    bonds = []
    for i, atom_a in enumerate(atoms):
        for atom_b in atoms[i + 1:]:
            if atom_a["chain"] != atom_b["chain"] or abs(atom_a["resnum"] - atom_b["resnum"]) > 2:
                continue
            distance = float(np.linalg.norm(atom_b["pos"] - atom_a["pos"]))
            cutoff = min(3.3, COVALENT_RADII.get(atom_a["element"], 0.78) + COVALENT_RADII.get(atom_b["element"], 0.78) + 0.55)
            if 0.35 < distance <= cutoff:
                bonds.append({"a": atom_a["pos"], "b": atom_b["pos"], "kind": classify_bond(atom_a, atom_b, distance), "distance": distance})
                if len(bonds) >= limit:
                    return bonds
    return bonds


def estimate_indexed_bonds(atoms: list[dict], limit: int = 1200, residue_window: int = 2) -> list[dict]:
    bonds = []
    for i, atom_a in enumerate(atoms):
        for j, atom_b in enumerate(atoms[i + 1:], start=i + 1):
            if atom_a["chain"] == atom_b["chain"] and abs(atom_a["resnum"] - atom_b["resnum"]) > residue_window:
                continue
            distance = float(np.linalg.norm(atom_b["pos"] - atom_a["pos"]))
            cutoff = min(3.3, COVALENT_RADII.get(atom_a["element"], 0.78) + COVALENT_RADII.get(atom_b["element"], 0.78) + 0.55)
            if 0.35 < distance <= cutoff:
                bonds.append({"a_idx": i, "b_idx": j, "kind": classify_bond(atom_a, atom_b, distance), "distance": distance})
                if len(bonds) >= limit:
                    return bonds
    return bonds


def center_scale_points(points: np.ndarray, target_radius: float = 4.0) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    radius = max(float(np.max(np.linalg.norm(centered, axis=1))), 1e-6)
    scale = target_radius / radius
    return centered * scale, centroid, scale


def center_and_scale_protein(chains: dict[str, list[tuple[np.ndarray, str]]], bonds: list[dict]):
    all_points = np.array([point for residues in chains.values() for point, _ in residues])
    scaled_points, centroid, scale = center_scale_points(all_points, target_radius=4.2)
    idx = 0
    scaled_chains = {}
    for chain_id, residues in chains.items():
        count = len(residues)
        scaled_chains[chain_id] = [(scaled_points[idx + i], residues[i][1]) for i in range(count)]
        idx += count
    scaled_bonds = [{"a": (bond["a"] - centroid) * scale, "b": (bond["b"] - centroid) * scale, "kind": bond["kind"], "distance": bond["distance"]} for bond in bonds]
    return scaled_chains, scaled_bonds


def catmull_rom(points, steps: int = STEPS) -> np.ndarray:
    points = np.array(points, dtype=float)
    if len(points) < 4:
        return points
    out = []
    for i in range(1, len(points) - 2):
        p0, p1, p2, p3 = points[i - 1], points[i], points[i + 1], points[i + 2]
        for t in np.linspace(0.0, 1.0, steps, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(points[-2])
    return np.array(out)


def unit_tangents(points: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    magnitudes = np.linalg.norm(tangents, axis=1, keepdims=True)
    magnitudes[magnitudes < 1e-8] = 1.0
    return tangents / magnitudes


def compute_tube(points, radius: float):
    points = np.array(points, dtype=float)
    if len(points) < 2:
        return []
    tangents = unit_tangents(points)
    ref = np.array([1.0, 0.0, 0.0]) if abs(tangents[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    normal = ref - np.dot(ref, tangents[0]) * tangents[0]
    normal /= np.linalg.norm(normal)
    normals = [normal]
    for i in range(1, len(points)):
        previous, current = tangents[i - 1], tangents[i]
        axis = np.cross(previous, current)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            normals.append(normals[-1])
            continue
        axis /= axis_len
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(previous, current)))))
        old = normals[-1]
        rotated = old * math.cos(angle) + np.cross(axis, old) * math.sin(angle) + axis * np.dot(axis, old) * (1 - math.cos(angle))
        normals.append(rotated / max(np.linalg.norm(rotated), 1e-8))
    rings = []
    for i in range(len(points)):
        binormal = np.cross(tangents[i], normals[i])
        length = np.linalg.norm(binormal)
        if length > 1e-8:
            binormal /= length
        vertices, ring_normals = [], []
        for j in range(SIDES):
            angle = 2 * math.pi * j / SIDES
            offset = math.cos(angle) * normals[i] + math.sin(angle) * binormal
            vertices.append(points[i] + radius * offset)
            ring_normals.append(offset)
        rings.append((vertices, ring_normals))
    return rings


def build_protein_geometry(chains: dict[str, list[tuple[np.ndarray, str]]]) -> list[tuple[list, tuple[float, float, float]]]:
    segments = []
    for residues in chains.values():
        current_points = [residues[0][0]]
        current_ss = residues[0][1]
        for point, ss in residues[1:]:
            if ss == current_ss:
                current_points.append(point)
            else:
                segments.append((current_ss, list(current_points)))
                current_points = [current_points[-1], point]
                current_ss = ss
        segments.append((current_ss, current_points))
    render_data = []
    for ss, points in segments:
        if len(points) < 2:
            continue
        spline = catmull_rom(points)
        color = HELIX_COL if ss == "H" else SHEET_COL if ss == "S" else COIL_COL
        radius = 0.34 if ss == "H" else 0.26 if ss == "S" else 0.14
        rings = compute_tube(spline, radius)
        if rings:
            render_data.append((rings, color))
    return render_data


def rotation_from_z(direction: np.ndarray) -> tuple[float, float, float, float]:
    unit = direction / max(np.linalg.norm(direction), 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z_axis, unit)
    axis_len = np.linalg.norm(axis)
    dot = float(np.dot(z_axis, unit))
    if axis_len < 1e-8:
        if dot > 0:
            return 0.0, 0.0, 0.0, 1.0
        return 180.0, 1.0, 0.0, 0.0
    axis /= axis_len
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return angle, axis[0], axis[1], axis[2]


def draw_cylinder_between(quad, start: np.ndarray, end: np.ndarray, radius: float):
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        return
    angle, ax, ay, az = rotation_from_z(delta)
    glPushMatrix()
    glTranslatef(*start)
    if angle:
        glRotatef(angle, ax, ay, az)
    gluCylinder(quad, radius, radius, length, 14, 1)
    glPopMatrix()


def surface_to_texture(surface: pygame.Surface):
    data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex, width, height


def make_texture(font, text: str, color):
    return surface_to_texture(font.render(text, True, color))


def blit_texture(tex, width, height, x, y):
    glBindTexture(GL_TEXTURE_2D, tex)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y)
    glTexCoord2f(1, 0); glVertex2f(x + width, y)
    glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
    glTexCoord2f(0, 1); glVertex2f(x, y + height)
    glEnd()


def build_overlay(spec: EntitySpec, stats: list[str]) -> list[tuple[int, int, int, int, int]]:
    title_font = pygame.font.SysFont("consolas", 20, bold=True)
    items = []
    title_color = tuple(int(c * 255) for c in CATEGORY_COLORS.get(spec.category, (0.96, 0.96, 0.96)))
    items.append((*make_texture(title_font, spec.label.upper(), title_color), 16, WIN_H - 34))
    if spec.mode == "protein":
        body_font = pygame.font.SysFont("consolas", 13, bold=True)
        for i, (label, col) in enumerate([
            ("| HELIX", (230, 60, 60)),
            ("| SHEET", (230, 205, 30)),
            ("| LOOP", (50, 200, 75)),
        ]):
            items.append((*make_texture(body_font, label, col), 12, 12 + i * 18))
    return items


def draw_overlay(items):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIN_W, 0, WIN_H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glColor4f(1, 1, 1, 1)
    for tex, width, height, x, y in items:
        blit_texture(tex, width, height, x, y)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def build_protein_display_list(render_list: list[tuple[list, tuple[float, float, float]]], bonds: list[dict]) -> int:
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    for rings, color in render_list:
        glColor3f(*color)
        for i in range(len(rings) - 1):
            rv1, rn1 = rings[i]
            rv2, rn2 = rings[i + 1]
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(SIDES + 1):
                idx = j % SIDES
                glNormal3fv(rn1[idx]); glVertex3fv(rv1[idx])
                glNormal3fv(rn2[idx]); glVertex3fv(rv2[idx])
            glEnd()
    glDisable(GL_LIGHTING)
    glLineWidth(2.6)
    glBegin(GL_LINES)
    for bond in bonds:
        glColor3f(*BOND_COLORS.get(bond["kind"], (0.92, 0.92, 0.92)))
        glVertex3fv(bond["a"])
        glVertex3fv(bond["b"])
    glEnd()
    glEnable(GL_LIGHTING)
    glEndList()
    return dl


def build_mesh_display_list(triangles: np.ndarray, normals: np.ndarray, color=(0.84, 0.60, 0.56)) -> int:
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    glColor3f(*color)
    glBegin(GL_TRIANGLES)
    for tri, normal in zip(triangles, normals):
        glNormal3fv(normal)
        glVertex3fv(tri[0])
        glVertex3fv(tri[1])
        glVertex3fv(tri[2])
    glEnd()
    glEndList()
    return dl


def build_atom_display_list(atoms: list[dict], bonds: list[dict]) -> int:
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    quad = gluNewQuadric()
    for bond in bonds:
        color = BOND_COLORS.get(bond.get("kind", "covalent"), (0.92, 0.92, 0.92))
        glColor3f(*color)
        radius = BOND_RADII.get(bond.get("kind", "covalent"), 0.065)
        draw_cylinder_between(quad, atoms[bond["a_idx"]]["pos"], atoms[bond["b_idx"]]["pos"], radius)
    for atom in atoms:
        glColor3f(*ELEMENT_COLORS.get(atom["element"], (0.82, 0.82, 0.82)))
        radius = ELEMENT_RADII.get(atom["element"], 0.18)
        glPushMatrix()
        glTranslatef(*atom["pos"])
        gluSphere(quad, radius, 22, 16)
        glPopMatrix()
    gluDeleteQuadric(quad)
    glEndList()
    return dl


def build_nucleic_display_list(render_list: list[tuple[list, tuple[float, float, float]]], bridges: list[dict]) -> int:
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    for rings, color in render_list:
        glColor3f(*color)
        for i in range(len(rings) - 1):
            rv1, rn1 = rings[i]
            rv2, rn2 = rings[i + 1]
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(SIDES + 1):
                idx = j % SIDES
                glNormal3fv(rn1[idx]); glVertex3fv(rv1[idx])
                glNormal3fv(rn2[idx]); glVertex3fv(rv2[idx])
            glEnd()
    quad = gluNewQuadric()
    for bridge in bridges:
        glColor3f(*BOND_COLORS.get(bridge["kind"], (0.95, 0.85, 0.42)))
        draw_cylinder_between(quad, bridge["a"], bridge["b"], 0.05)
    gluDeleteQuadric(quad)
    glEndList()
    return dl


def configure_gl():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [3.5, 4.0, 5.2, 0.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.18, 0.18, 0.18, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.48, 0.48, 0.48, 1.0])
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-3.0, -2.5, -4.0, 0.0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.30, 0.32, 0.35, 1.0])
    glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.28, 0.28, 0.28, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 54.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(42.0, WIN_W / WIN_H, 0.1, 5000.0)
    glMatrixMode(GL_MODELVIEW)

def zoom_for_points(points: np.ndarray, scale_factor: float = 3.4, padding: float = 6.5) -> float:
    if len(points) == 0:
        return -14.0
    centroid = points.mean(axis=0)
    radius = max(float(np.max(np.linalg.norm(points - centroid, axis=1))), 0.5)
    return -(radius * scale_factor + padding)


def list_items():
    print_cli_banner()
    grouped: dict[str, list[str]] = {}
    for key, spec in sorted(CATALOG.items()):
        grouped.setdefault(spec.category, []).append(key)
    print(f"\n{ANSI['white']}[ catalog ]{ANSI['reset']} verified source-backed structures\n")
    for category, items in grouped.items():
        tint = {
            "Protein": ANSI["red"],
            "Molecule": ANSI["blue"],
            "Nucleic Acid": ANSI["yellow"],
            "Human Organ": ANSI["magenta"],
        }.get(category, ANSI["white"])
        print(f"{tint}{category.upper()} ({len(items)}){ANSI['reset']}")
        for item in items:
            print(f"  - {item}")
        print()


def print_cli_banner():
    banner = [
        f"{ANSI['cyan']}=============================================================={ANSI['reset']}",
        f"{ANSI['cyan']}  ____  _       _                 __      ___                         {ANSI['reset']}",
        f"{ANSI['cyan']} | __ )(_) ___ | | ___   __ _ _   \\ \\    / (_) _____      _____ _ __  {ANSI['reset']}",
        f"{ANSI['blue']} |  _ \\| |/ _ \\| |/ _ \\ / _` | | | \\ \\  / /| |/ _ \\ \\ /\\ / / _ \\ '__| {ANSI['reset']}",
        f"{ANSI['magenta']} | |_) | | (_) | | (_) | (_| | |_| |\\ \\/ / | |  __/\\ V  V /  __/ |    {ANSI['reset']}",
        f"{ANSI['green']} |____/|_|\\___/|_|\\___/ \\__, |\\__, | \\__/  |_|\\___| \\_/\\_/ \\___|_|    {ANSI['reset']}",
        f"{ANSI['green']}                        |___/ |___/                                    {ANSI['reset']}",
        f"  {ANSI['yellow']}[stamp]{ANSI['reset']} source-backed | {ANSI['red']}pdb{ANSI['reset']} + {ANSI['magenta']}bodyparts3d{ANSI['reset']} + {ANSI['blue']}pubchem 3d{ANSI['reset']}",
        f"  {ANSI['white']}[mode ]{ANSI['reset']} quality review build | centered layouts | verified fetch path",
        f"{ANSI['cyan']}=============================================================={ANSI['reset']}",
    ]
    print("\n".join(banner))


def build_scene(spec: EntitySpec, refresh: bool = False):
    if spec.mode == "mesh":
        try:
            triangle_sets = []
            normal_sets = []
            for url in spec.mesh_urls:
                mesh_bytes = fetch_binary_asset(url, refresh=refresh)
                triangles, normals = load_binary_stl(mesh_bytes)
                triangle_sets.append(triangles)
                normal_sets.append(normals)
            merged_triangles = np.concatenate(triangle_sets, axis=0)
            merged_normals = np.concatenate(normal_sets, axis=0)
            normalized_triangles = normalize_mesh(merged_triangles)
            zoom = zoom_for_points(normalized_triangles.reshape(-1, 3), scale_factor=3.5, padding=6.5)
            color = MESH_COLORS.get(spec.key, (0.84, 0.60, 0.56))
            return build_mesh_display_list(normalized_triangles, merged_normals, color=color), zoom, build_overlay(spec, [])
        except Exception as exc:
            raise RuntimeError(f"Failed to load real anatomy mesh for {spec.key}: {exc}") from exc
    if spec.mode == "nucleic":
        try:
            pdb_text = fetch_pdb(spec.pdb_id, refresh=refresh)
            chains = parse_nucleic_backbone(pdb_text)
            render_list, bridges = build_nucleic_geometry(chains)
            all_points = np.array([entry["pos"] for residues in chains.values() for entry in residues], dtype=float)
            zoom = zoom_for_points(all_points, scale_factor=3.6, padding=6.5)
            return build_nucleic_display_list(render_list, bridges), zoom, build_overlay(spec, [])
        except Exception as exc:
            raise RuntimeError(f"Failed to load nucleic-acid structure for {spec.key} from RCSB PDB: {exc}") from exc
    if spec.mode == "atom_pdb":
        try:
            pdb_text = fetch_pdb(spec.pdb_id, refresh=refresh)
            atoms = parse_atom_structure(pdb_text)
            if not atoms:
                raise ValueError("no atoms parsed")
            atoms = normalize_atoms(atoms, target_radius=4.2)
            bonds = estimate_indexed_bonds(atoms, limit=1500, residue_window=1)
            zoom = zoom_for_points(np.array([atom["pos"] for atom in atoms], dtype=float), scale_factor=3.5, padding=6.0)
            return build_atom_display_list(atoms, bonds), zoom, build_overlay(spec, [])
        except Exception as exc:
            raise RuntimeError(f"Failed to load all-atom structure for {spec.key} from RCSB PDB: {exc}") from exc
    if spec.mode == "compound":
        try:
            sdf_text = fetch_pubchem_sdf(spec.compound_cid, refresh=refresh)
            atoms, bonds = parse_sdf(sdf_text)
            atoms = normalize_atoms(atoms, target_radius=4.2)
            zoom = zoom_for_points(np.array([atom["pos"] for atom in atoms], dtype=float), scale_factor=3.7, padding=5.6)
            return build_atom_display_list(atoms, bonds), zoom, build_overlay(spec, [])
        except Exception as exc:
            raise RuntimeError(f"Failed to load 3D compound for {spec.key} from PubChem: {exc}") from exc
    try:
        pdb_text = fetch_pdb(spec.pdb_id, refresh=refresh)
        chains, atoms = parse_protein(pdb_text)
        if not chains:
            raise ValueError("no valid protein chains parsed")
        bonds = estimate_bonds(atoms)
        chains, bonds = center_and_scale_protein(chains, bonds)
        render_list = build_protein_geometry(chains)
        protein_points = np.array([point for residues in chains.values() for point, _ in residues], dtype=float)
        zoom = zoom_for_points(protein_points, scale_factor=3.7, padding=7.2)
        return build_protein_display_list(render_list, bonds), zoom, build_overlay(spec, [])
    except Exception as exc:
        raise RuntimeError(f"Failed to load real structure data for {spec.key} from RCSB PDB: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Compact 3D biology structure atlas.")
    parser.add_argument("--item", help="Item key to visualize, for example hemoglobin, glucose, or liver.")
    parser.add_argument("--protein", help="Backward-compatible alias for --item.")
    parser.add_argument("--list", action="store_true", help="List every available structure.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached PDB downloads.")
    args = parser.parse_args()
    if args.list:
        list_items()
        return
    print_cli_banner()
    requested = args.item or args.protein
    if not requested:
        parser.error("provide --item <name> or use --list")
    key = normalize_key(requested)
    if key not in CATALOG:
        print(f"\n[!] Unknown structure: {key}\n")
        list_items()
        sys.exit(1)
    spec = CATALOG[key]
    print(f"[*] BiologyViewer  |  {spec.label}  |  {spec.category}")
    print(f"[>] item={spec.key}  mode={spec.mode}  source={spec.category.lower()}")
    pygame.init()
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"BiologyViewer | {spec.label}")
    configure_gl()
    try:
        display_list, initial_zoom, overlay = build_scene(spec, refresh=args.refresh)
    except RuntimeError as exc:
        pygame.quit()
        print(f"[!] {exc}")
        sys.exit(1)
    rot_x, rot_y, zoom = -8.0, 22.0, float(initial_zoom)
    dragging = False
    auto_spin = True
    last_mouse = (0, 0)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                glDeleteLists(display_list, 1)
                pygame.quit()
                sys.exit(0)
            if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                auto_spin = not auto_spin
            if event.type == KEYDOWN and event.key == pygame.K_r:
                rot_x, rot_y, zoom = -8.0, 22.0, float(initial_zoom)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    zoom = min(zoom + max(abs(zoom) * 0.06, 1.5), -4.0)
                elif event.button == 5:
                    zoom -= max(abs(zoom) * 0.06, 1.5)
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                rot_y += dx * 0.42
                rot_x += dy * 0.42
                last_mouse = event.pos
        if auto_spin and not dragging:
            rot_y += AUTO_SPD
        glClearColor(*BG_COL)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rot_x, 1.0, 0.0, 0.0)
        glRotatef(rot_y, 0.0, 1.0, 0.0)
        glCallList(display_list)
        draw_overlay(overlay)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
