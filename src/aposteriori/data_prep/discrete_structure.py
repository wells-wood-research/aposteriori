"""Tools for discretizing a protein structure into voxels."""

import gzip
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np  # type: ignore

import ampal  # type: ignore


class DiscreteAtom:
    """An atom with discretized coordinates."""

    __slots__ = [
        "pdb_code",
        "indices",
        "chain",
        "res_num",
        "mol_code",
        "mol_letter",
        "res_label",
        "element",
        "protein",
    ]

    def __init__(
        self,
        indices: Tuple[(int, int, int)],
        pdb_code: str,
        chain: str,
        res_num: int,
        mol_code: str,
        mol_letter: str,
        res_label: str,
        element: str,
        protein: bool,
    ):
        self.indices = indices
        self.pdb_code = pdb_code
        self.chain = chain
        self.res_num = res_num
        self.mol_code = mol_code
        self.mol_letter = mol_letter
        self.res_label = res_label
        self.element = element
        self.protein = protein

    def __repr__(self):
        return f"<DiscreteAtom: {self.mol_code}, {self.res_label}, {self.element}>"

    def monomer_info(self):
        return (self.pdb_code, self.chain, self.res_num)

    @classmethod
    def from_ampal_atom(
        cls, atom: ampal.Atom, max_distance: float = 1.0, padding: int = 0
    ):
        """Converts an ampal.Atom to a DiscreteAtom.
        
        Notes
        -----

        In order to make sure 2 non-hydrogen atoms are not in the same
        zone, the edges of the zone must have a length of at most
        0.57, as Pythagoras tells us for a cube, that s, the longest
        corner to corner distance, is equal to sqrt(3x**2),
        rearranging to find a value the sides for a longest distance
        give sqrt((a**2)/3).
        """
        padding = int(padding)  # make sure the padding is an integer
        edge_length = np.sqrt((max_distance ** 2) / 3)
        x = cls.discretize(atom.x, edge_length)
        y = cls.discretize(atom.y, edge_length)
        z = cls.discretize(atom.z, edge_length)
        indices = (x + padding, y + padding, z + padding)
        residue = atom.parent
        polymer = residue.parent
        assembly = polymer.parent
        protein = isinstance(residue, ampal.Residue)
        mol_letter = residue.mol_letter if hasattr(residue, "mol_letter") else "!"
        return cls(
            indices,
            assembly.id[:4],
            polymer.id,
            int(residue.id),
            residue.mol_code,
            mol_letter,
            atom.res_label,
            atom.element,
            protein,
        )

    @staticmethod
    def discretize(number: float, box_size: float) -> int:
        """Rounds and then converts to an integer."""
        return int(number // box_size)


class DiscreteStructure:
    """Creates a discretized protein structure.
    
    Examples
    --------
    You can create a `DiscreteStructure` one of 2 ways: from a file
    or a list ampal.Atoms. The latter is useful when you are wanting
    to be very specific about the atoms that will be included in
    your structure. This example will focus on using a file as an
    input.

    To start, I'll discretize a gzipped PDB file:
    >>> from pathlib import Path
    >>> d_structure = DiscreteStructure.from_pdb_path(
    ...     Path('tests/testing_files/biounit/mk/1mkk.pdb1.gz'),
    ...     padding=10,
    ...     gzipped=True,
    ... )

    This class method returns a `DiscreteStructure`. With default
    arguments non-backbone protein, hydrogen atoms and water
    molecules will be ignored. A padding value can be used to add
    empty space around the structure, this is useful if you're going
    to be working with regions of structure near the boundaries of
    the box. The voxel data is stored in the `data` attribute, which
    consists of a 3D array containing the atomic number of the
    atom in each discrete region of space.
    >>> d_structure.data.shape
    (81, 102, 120)

    If the region of space is empty, the value is 0.
    >>> d_structure.data[0, 0, 0]
    0

    In this structure, there's an alpha carbon at index
    (68, 81, 47):
    >>> d_structure.data[68, 81, 47]
    6

    Information about all of the atoms in the discrete structure is
    contained in the `discrete_atoms` attribute, which is a list of
    `DiscreteAtoms`:

    >>> d_structure.discrete_atoms[:2]
    [<DiscreteAtom: VAL, N, N>, <DiscreteAtom: VAL, CA, C>]
    
    >>> ca_atom = d_structure.discrete_atoms[1]
    >>> ca_atom.indices
    (68, 81, 47)
    >>> d_structure.data[ca_atom.indices]
    6

    There's also a convenience property for getting a iterable of all
    the c-alpha atoms:

    >>> list(d_structure.ca_atoms)[:2]
    [<DiscreteAtom: VAL, CA, C>, <DiscreteAtom: VAL, CA, C>]

    There's lots more information in the `DiscreteAtom` objects, take
    a look at the `DiscreteAtom` documentation for more information.

    It's easy to get a region around an atom too:
    >>> ca_region = d_structure.get_region(
    ...     ca_atom.indices, radius=10
    ... )
    >>> ca_region.shape
    (21, 21, 21)
    >>> ca_region[10, 10, 10]  # ca_atom is at the centre
    6

    Finally, you can also get an iterable of all the regions around
    the c-alpha carbons:
    >>> ca_regions = list(d_structure.get_ca_regions(radius=10))
    >>> np.array_equal(ca_regions[0], ca_region)
    True
    """

    backbone_atoms = ("N", "CA", "C", "O")

    def __init__(
        self, atoms: Iterable[ampal.Atom], padding=0, max_atom_distance: float = 1.0
    ):
        self.padding = padding
        self.discrete_atoms = [
            DiscreteAtom.from_ampal_atom(
                atom, max_distance=max_atom_distance, padding=self.padding
            )
            for atom in atoms
        ]
        # add max padding so that it's even all round as min padding has been added
        dimensions = self._find_dimensions(self.discrete_atoms) + self.padding
        # uint8 should be safe here as it's storing the atomic number
        self.data = np.zeros(dimensions, dtype=np.uint8)
        self.data.fill(0)
        for d_atom in self.discrete_atoms:
            assert (d_atom.element != "") or (d_atom.element != " "), (
                f"Atom element should not be blank:\n"
                f"{d_atom.chain}:{d_atom.res_num}:{d_atom.res_label}"
            )
            assert (d_atom.mol_letter != "") or (d_atom.mol_letter != " "), (
                f"Atom mol letter should not be blank:\n"
                f"{d_atom.chain}:{d_atom.res_num}:{d_atom.res_label}"
            )
            assert self.data[d_atom.indices] == 0, (
                f"Voxel should not be occupied: Currently "
                f"{self.data[tuple(d_atom.indices)]}, "
                f"{d_atom.pdb_code}:{d_atom.chain}:{d_atom.res_num}:{d_atom.res_label}"
            )
            element_data = ampal.data.ELEMENT_DATA[d_atom.element.capitalize()]
            self.data[d_atom.indices] = element_data["atomic number"]

    @classmethod
    def from_pdb_path(
        cls,
        pdb_path: Path,
        max_atom_distance: float = 1.0,
        backbone_only=True,
        include_hydrogen=False,
        filter_monomers=("HOH",),
        padding=0,
        gzipped=False,
    ):
        """Creates a DiscreteStructure from a PDB file."""
        pdb_id = pdb_path.stem
        if gzipped:
            with gzip.open(pdb_path) as inf:
                structure = ampal.load_pdb(
                    inf.read().decode(), pdb_id=pdb_id, path=False
                )
        else:
            with open(pdb_path) as inf:
                structure = ampal.load_pdb(inf.read(), pdb_id=pdb_id, path=False)
        assert isinstance(
            structure, ampal.AmpalContainer
        ), "Structure should be an AMPAL container."
        assembly = structure[0]
        assembly.translate(_create_min_vector(assembly))

        # if you're confused by this closure, it is used as the filter function as
        # a lambda is a bit long
        def is_included_atom(atom):
            if include_hydrogen and (atom.element == "H"):
                return False
            if atom.parent.mol_code in filter_monomers:
                return False
            if (
                backbone_only
                and isinstance(atom.parent, ampal.Residue)
                and (atom.res_label not in cls.backbone_atoms)
            ):
                return False
            return True

        return cls(
            filter(is_included_atom, assembly.get_atoms()),
            padding=padding,
            max_atom_distance=max_atom_distance,
        )

    @property
    def ca_atoms(self):
        """Returns an iterable of all c-alpha carbons."""
        return (x for x in self.discrete_atoms if x.protein and (x.res_label == "CA"))

    def get_region(self, focus: Tuple[int, int, int], radius: int) -> np.ndarray:
        """Selects regions of the structure around the focus point."""
        fx, fy, fz = focus
        region = self.data[
            fx - radius : fx + radius + 1,
            fy - radius : fy + radius + 1,
            fz - radius : fz + radius + 1,
        ]
        return region

    def get_ca_regions(self, radius) -> Iterable[np.ndarray]:
        """Returns an iterable of all regions around c-alpha carbons."""
        return (self.get_region(x.indices, radius) for x in self.ca_atoms)

    @staticmethod
    def _find_dimensions(atoms: List[DiscreteAtom]) -> np.ndarray:
        """Finds the maximum indices in each dimension."""
        x = max(atoms, key=lambda atom: atom.indices[0]).indices[0]
        y = max(atoms, key=lambda atom: atom.indices[1]).indices[1]
        z = max(atoms, key=lambda atom: atom.indices[2]).indices[2]
        return np.array((x + 1, y + 1, z + 1))


def _create_min_vector(assembly: ampal.Assembly) -> Tuple[float, float, float]:
    """Returns a vector of the minimun x, y and z values in the structure."""
    min_x = min(atom.x for atom in assembly.get_atoms())
    min_y = min(atom.y for atom in assembly.get_atoms())
    min_z = min(atom.z for atom in assembly.get_atoms())
    return (-min_x, -min_y, -min_z)
