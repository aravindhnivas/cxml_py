from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal
from cxml_lib.logger import logger
from rdkit import Chem
from rdkit.Chem import AllChem

from cxml_lib.utils import parse_args


@dataclass
class OptimizationConfig:
    """Configuration for 3D structure optimization.

    Attributes:
        max_attempts (int): Maximum number of embedding attempts.
        random_seed (int): Random seed for reproducible embedding.
        force_field (Literal["MMFF94s", "MMFF94", "UFF"]): Force field to use for optimization.
        optimization_steps (int): Maximum number of optimization steps.
        energy_threshold (float): Energy convergence threshold.
    """

    max_attempts: int = 10
    random_seed: int = 42
    force_field: Literal["MMFF94s", "MMFF94", "UFF"] = "MMFF94s"
    optimization_steps: int = 1000
    energy_threshold: float = 1e-4

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.optimization_steps < 1:
            raise ValueError("optimization_steps must be at least 1")
        if self.energy_threshold <= 0:
            raise ValueError("energy_threshold must be positive")


@dataclass
class Args:
    """Input arguments for molecule optimization.

    Attributes:
        smiles (str): SMILES string of the molecule to optimize.
        config (Optional[OptimizationConfig]): Optional configuration for optimization.
    """

    smiles: str
    config: Optional[OptimizationConfig] = None

    def __post_init__(self):
        """Set default configuration if none provided."""
        if self.config is None:
            self.config = OptimizationConfig()


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SMILES string and return molecule if valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: Boolean indicating if SMILES is valid
        - error_message: Error message if validation failed, None otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False, "Invalid SMILES input: must be a non-empty string"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES string: could not parse molecule"

        # Additional validation for 3D optimization
        if mol.GetNumAtoms() == 0:
            return False, "Invalid SMILES string: molecule has no atoms"

        return True, None
    except Exception as e:
        return False, f"Error validating SMILES: {str(e)}"


def optimize_3d_structure(
    mol: Chem.Mol, config: OptimizationConfig
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Optimize 3D structure of a molecule using specified configuration.

    Args:
        mol: RDKit molecule object
        config: Optimization configuration

    Returns:
        Tuple of (optimized_molecule, error_message)
        - optimized_molecule: The optimized RDKit molecule object, or None if optimization failed
        - error_message: Error message if optimization failed, None otherwise
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
        try:
            if config.force_field == "MMFF94s":
                result = AllChem.MMFFOptimizeMolecule(
                    mol, maxIters=config.optimization_steps
                )
            elif config.force_field == "MMFF94":
                result = AllChem.MMFFOptimizeMolecule(
                    mol, maxIters=config.optimization_steps, mmffVariant="MMFF94"
                )
            else:  # UFF
                result = AllChem.UFFOptimizeMolecule(
                    mol, maxIters=config.optimization_steps
                )

            if result != 0:
                return None, f"Force field optimization failed with code: {result}"

            return mol, None

        except Exception as e:
            return None, f"Force field optimization failed: {str(e)}"

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
        - pdb_string: PDB format string of the optimized molecule, or None if conversion failed
        - error_message: Error message if conversion failed, None otherwise
    """
    try:
        # Validate SMILES
        is_valid, error = validate_smiles(smiles)
        if not is_valid:
            return None, error

        # Create molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Failed to create molecule from SMILES"

        # Set default config if none provided
        if config is None:
            config = OptimizationConfig()

        # Optimize 3D structure
        optimized_mol, error = optimize_3d_structure(mol, config)
        if error:
            return None, error

        # Convert to PDB format
        try:
            optimized_pdb = Chem.MolToPDBBlock(optimized_mol)
            if not optimized_pdb:
                return None, "Failed to convert molecule to PDB format"
            return optimized_pdb, None
        except Exception as e:
            return None, f"Error converting to PDB format: {str(e)}"

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def main(args: Args) -> Dict[str, Any]:
    """
    Main function to optimize molecule structure from SMILES.

    Args:
        args: Input arguments containing SMILES and optional configuration

    Returns:
        Dictionary containing optimization results and any error messages
        - optimized_pdb: PDB format string of the optimized molecule, or None if optimization failed
        - error: Error message if optimization failed, None otherwise
    """
    try:
        args = parse_args(args.__dict__, Args)
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
    except Exception as e:
        error_msg = f"Unexpected error in main function: {str(e)}"
        logger.error(error_msg)
        return {"optimized_pdb": None, "error": error_msg}
