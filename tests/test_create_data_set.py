"""Tests data processing functionality in src/aposteriori/create_data_set.py"""
from collections import Counter
import numpy as np  # type: ignore
from pathlib import Path
import pytest  # type: ignore

from aposteriori.data_prep.discrete_structure import DiscreteStructure
from aposteriori.data_prep.create_data_set import create_discrete_structure_v1
import ampal  # type: ignore


class TestEncodedCreateDataSet:
    pdb_path = Path("tests/testing_files/biounit/mk/1mkk.pdb1.gz")
    elements_by_atomic_num = {
        v["atomic number"]: k for k, v in ampal.data.ELEMENT_DATA.items()
    }
    d_structure = DiscreteStructure.from_pdb_path(pdb_path, padding=0, gzipped=True)
    pdb_code, data, indices, labels = create_discrete_structure_v1(pdb_path)

    def test_fail_to_create(self):
        """Should raise an assestion error as structure has no protein."""
        path_to_bad_file = Path("tests/testing_files/biounit/mk/4mkw.pdb1.gz")
        with pytest.raises(AssertionError):
            _ = create_discrete_structure_v1(path_to_bad_file)

    def test_check_pdb_code(self):
        assert self.pdb_code == "1mkk.pdb1"

    def test_dimensions(self):
        assert self.d_structure.data.shape == self.data.shape[:3]

    def test_elemental_composition(self):
        d_elements = [a.element for a in self.d_structure.discrete_atoms]
        data_elements = [
            self.elements_by_atomic_num[a] for a in self.data.flatten() if a != 0
        ]
        assert Counter(d_elements) == Counter(data_elements)

    def test_amino_acid_composition(self):
        d_sequence = "".join(
            [
                a.mol_letter
                for a in self.d_structure.discrete_atoms
                if a.protein and (a.res_label == "CA")
            ]
        )
        label_sequence = "".join(
            # need to decode the label as it is a byte string
            ampal.amino_acids.get_aa_letter(l.decode())
            for l in self.labels
        )
        assert d_sequence == label_sequence

    def test_indices(self):
        print(self.indices)
        print(self.data.shape)
        ca_atoms = np.array([self.data[tuple(i)] for i in self.indices])
        assert len(ca_atoms) > 0
        ca_elements = [self.elements_by_atomic_num[a] for a in ca_atoms]
        assert all(map(lambda x: x == "C", ca_elements))
