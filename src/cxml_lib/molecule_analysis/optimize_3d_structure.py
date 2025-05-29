from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from cxml_lib.logger import logger
from rdkit import Chem
from rdkit.Chem import AllChem

from cxml_lib.utils import parse_args


@dataclass
class OptimizationConfig:
    """Configuration for 3D structure optimization."""

    max_attempts: int = 10
    random_seed: int = 42
    force_field: str = "MMFF94s"  # Options: MMFF94s, MMFF94, UFF
    optimization_steps: int = 1000
    energy_threshold: float = 1e-4


@dataclass
class Args:
    """Input arguments for molecule optimization."""

    smiles: str
    config: Optional[OptimizationConfig] = None

    def __post_init__(self):
        if self.config is None:
            self.config = OptimizationConfig()


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SMILES string and return molecule if valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not smiles or not isinstance(smiles, str):
        return False, "Invalid SMILES input: must be a non-empty string"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string: could not parse molecule"

    return True, None


def optimize_3d_structure(
    mol: Chem.Mol, config: OptimizationConfig
) -> Tuple[Chem.Mol, Optional[str]]:
    """
    Optimize 3D structure of a molecule using specified configuration.

    Args:
        mol: RDKit molecule object
        config: Optimization configuration

    Returns:
        Tuple of (optimized_molecule, error_message)
    """
    try:
        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Embed molecule with ETKDG
        embedding_success = False
        for attempt in range(config.max_attempts):
            try:
                embedding_result = AllChem.EmbedMolecule(
                    mol,
                    randomSeed=config.random_seed + attempt,
                    useExpTorsionAnglePrefs=True,
                    useBasicKnowledge=True,
                )
                if embedding_result >= 0:
                    embedding_success = True
                    break
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                continue

        if not embedding_success:
            return None, "Failed to generate 3D coordinates after maximum attempts"

        # Optimize using specified force field
        if config.force_field == "MMFF94s":
            AllChem.MMFFOptimizeMolecule(mol, maxIters=config.optimization_steps)
        elif config.force_field == "MMFF94":
            AllChem.MMFFOptimizeMolecule(
                mol, maxIters=config.optimization_steps, mmffVariant="MMFF94"
            )
        else:  # UFF
            AllChem.UFFOptimizeMolecule(mol, maxIters=config.optimization_steps)

        return mol, None

    except Exception as e:
        return None, f"Optimization failed: {str(e)}"


def smiles_to_pdb_string(
    smiles: str, config: Optional[OptimizationConfig] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert SMILES string to optimized 3D structure in PDB format.

    Args:
        smiles: SMILES string of the molecule
        config: Optional configuration for optimization

    Returns:
        Tuple of (pdb_string, error_message)
    """
    try:
        # Validate SMILES
        is_valid, error = validate_smiles(smiles)
        if not is_valid:
            return None, error

        # Create molecule
        mol = Chem.MolFromSmiles(smiles)

        # Set default config if none provided
        if config is None:
            config = OptimizationConfig()

        # Optimize 3D structure
        optimized_mol, error = optimize_3d_structure(mol, config)
        if error:
            return None, error

        # Convert to PDB format
        optimized_pdb = Chem.MolToPDBBlock(optimized_mol)
        return optimized_pdb, None

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def main(args: Args) -> Dict[str, Any]:
    """
    Main function to optimize molecule structure from SMILES.

    Args:
        args: Input arguments containing SMILES and optional configuration

    Returns:
        Dictionary containing optimization results and any error messages
    """
    args = parse_args(args.__dict__, Args)
    # Set default config if none provided
    if args.config is None:
        args.config = OptimizationConfig()

    logger.info(f"Starting molecule optimization for SMILES: {args.smiles}")

    # Validate input
    is_valid, error = validate_smiles(args.smiles)
    if not is_valid:
        logger.error(f"Input validation failed: {error}")
        return {"optimized_pdb": None, "error": error}

    # Perform optimization
    optimized_pdb, error = smiles_to_pdb_string(args.smiles, args.config)

    if error:
        logger.error(f"Optimization failed: {error}")
    else:
        logger.info("Successfully optimized molecule structure")

    return {
        "optimized_pdb": optimized_pdb,
        "error": error,
    }
