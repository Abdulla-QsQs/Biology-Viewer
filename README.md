# Biology-Viewer

BiologyViewer is a compact 3D biology structure viewer built with Python, Pygame, and OpenGL.

It displays real-source-backed biological structures including:

- Proteins from RCSB PDB
- Human organs from BodyParts3D meshes
- Molecules from PubChem 3D
- Nucleic acids such as DNA and RNA from RCSB PDB

## Features

- Clean 3D rendering with compact framing
- Category-based visual identity
- CLI-style startup banner
- Protein secondary-structure coloring
- Ball-and-stick molecule rendering
- Source-backed anatomy mesh rendering
- Cached downloads for faster reuse

## Supported Categories

- Protein
- Molecule
- Nucleic Acid
- Human Organ

## Example Items

- `hemoglobin`
- `actin`
- `glucose`
- `atp`
- `dna_double_helix`
- `trna_rna`
- `liver`
- `lungs`
- `stomach`

## Usage

List all available structures:

python viewer.py --list
Open a structure:

python viewer.py --item hemoglobin
Other examples:

python viewer.py --item glucose
python viewer.py --item dna_double_helix
python viewer.py --item liver

## Data Sources

RCSB Protein Data Bank
PubChem 3D
BodyParts3D

## NOTES

Some structures are fetched from online sources and cached locally.
Protein rendering quality depends on source availability and network access.
Cached files may improve repeat loading speed.

## Requirements

Install the required Python packages before running:

```bash
pip install pygame PyOpenGL biopython numpy



Install the required Python packages before running:

