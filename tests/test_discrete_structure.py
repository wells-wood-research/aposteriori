"""Tests data processing functionality in src/aposteriori/format_data.py"""
from collections import Counter
import gzip
from pathlib import Path

from aposteriori.data_prep.discrete_structure import DiscreteStructure  # type: ignore
import ampal  # type: ignore


class TestDiscreteStructure:
    pdb_path = Path("tests/testing_files/biounit/mk/1mkk.pdb1.gz")
    with gzip.open(str(pdb_path)) as inf:
        assembly = ampal.load_pdb(inf.read().decode(), path=False)[0]
    elements_by_atomic_num = {
        v["atomic number"]: k for k, v in ampal.data.ELEMENT_DATA.items()
    }
    d_structure = DiscreteStructure.from_pdb_path(
        pdb_path,
        padding=10,
        backbone_only=False,
        include_hydrogen=True,
        filter_monomers=[],
    )

    def test_check_sequence(self):
        all_residues = [x.mol_letter for x in self.d_structure.ca_atoms]
        assert Counter(all_residues) == Counter("".join(self.assembly.sequences))

    def test_elemental_composition(self):
        dataset_elements = Counter(
            (
                self.elements_by_atomic_num[x].upper()
                for x in filter(lambda x: x != 0, self.d_structure.data.flatten())
            )
        )
        d_atoms = self.d_structure.discrete_atoms
        d_atom_elements = Counter((d_atom.element for d_atom in d_atoms))
        assembly_elements = Counter(
            (atom.element for atom in self.assembly.get_atoms())
        )
        assert dataset_elements == d_atom_elements == assembly_elements

    def test_indices(self):
        # Check the minimum index in each dimension
        d_atoms = self.d_structure.discrete_atoms
        a_x = min(d_atoms, key=lambda atom: atom.indices[0]).indices[0]
        a_y = min(d_atoms, key=lambda atom: atom.indices[1]).indices[1]
        a_z = min(d_atoms, key=lambda atom: atom.indices[2]).indices[2]
        padding = self.d_structure.padding
        assert (a_x, a_y, a_z) == (padding, padding, padding)
        # TODO: add a test for maximum dimensions

    def test_discrete_structure(self):
        # Check the sectors around CAs in labels
        padding = self.d_structure.padding
        for region in self.d_structure.get_ca_regions(padding):
            # Check the dimensions of the sector
            dim_length = padding * 2 + 1
            assert region.shape == (dim_length, dim_length, dim_length)
            # Check that the central atom in in a sector is a carbon (the CA)
            assert self.elements_by_atomic_num[region[padding][padding][padding]] == "C"
