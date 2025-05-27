from dataclasses import dataclass
from cxml_lib.logger import logger
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class Args:
    smiles: str


def smiles_to_pdb_string(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        # mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        optimized_pdb = Chem.MolToPDBBlock(mol)
        return optimized_pdb, None
    except Exception as e:
        return None, str(e)


def main(args: Args):
    logger.info(f"Analyzing molecule with SMILES string: {args.smiles}")
    optimized_pdb, error = smiles_to_pdb_string(args.smiles)
    if error:
        logger.error(f"Error: {error}")

    return {
        "optimized_pdb": optimized_pdb,
        "error": error,
    }
