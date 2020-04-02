"""Tools for creating a frame data set.

In this type of data set, all individual entries are stored separately in a flat
structure.
"""

import typing as t

import ampal
import ampal.geometry as geometry
import numpy as np


def align_to_residue_plane(residue: ampal.Residue):
    """Reorients the parent ampal.Assembly that the peptide plane lies on xy.
    
    Notes
    -----
    This changes the assembly **in place**.
    
    Parameters
    ----------
    residue : ampal.Residue
        Residue that will be used as a reference to reorient the assemble.
        The assembly will be reoriented so that the residue['CA'] lies on the origin,
        residue['N'] lies on +Y and residue['C'] lies on +X-Y, assuming correct
        geometry.
    """
    # unit vectors used for alignment
    origin = (0, 0, 0)
    unit_y = (0, 1, 0)
    unit_x = (1, 0, 0)

    # translate the whole parent assembly so that residue['CA'] lies on the origin
    translation_vector = residue["CA"].array
    assembly = residue.parent.parent
    assembly.translate(-translation_vector)

    # rotate whole assembly so that N-CA lies on Y
    n_vector = residue["N"].array
    rotation_axis = np.cross(n_vector, unit_y)
    try:
        assembly.rotate(geometry.angle_between_vectors(n_vector, unit_y), rotation_axis)
    except ZeroDivisionError:
        pass

    # align C with xy plane
    rotation_angle = geometry.dihedral(unit_x, origin, unit_y, residue["C"])
    assembly.rotate(-rotation_angle, unit_y)
    return


def within_frame(radius: float, atom: ampal.Atom) -> bool:
    """Tests if an atom is within the `radius` of the origin."""
    return all([0 <= v <= 2 * radius for v in atom.array])


def discretize(atom: ampal.Atom, voxel_edge_length: float) -> t.Tuple[int, int, int]:
    """Rounds and then converts to an integer."""
    return (
        int(atom.x // voxel_edge_length),
        int(atom.y // voxel_edge_length),
        int(atom.z // voxel_edge_length),
    )


def create_residue_frame(
    residue: ampal.Residue, radius: float, voxels_per_side: int
) -> t.Tuple[str, np.ndarray]:
    """Creates a discreet representation of a volume of space around a residue.
    
    Notes
    -----
    We use the term "frame" to refer to a cube of space around a residue.
    
    Parameters
    ----------
    residue : ampal.Residue
        The residue to be converted to a frame.
    radius : float
        This term is slightly confusing as it's not really the radius as the frame is
        a cube. It is helf the edge length of the frame.
    voxels_per_side : int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    
    Returns
    -------
    unique_key : str
        A unique identifier for the residue e.g.`3qy1:A:3:ASP` (pdb_code:chain:residue number:res code)
    
    Raises
    ------
    AssertionError
        Raised if:
        
        * If any atom does not have an element label.
        * If any residue does not have a three letter `mol_code` i.e. "LYS" etc
        * If any voxel is already occupied
        * If the central voxel in the frame is not carbon as it should the the CA atom
    """
    assert voxels_per_side % 2, "The number of voxels per side should be odd."
    voxel_edge_length = (2 * radius) / voxels_per_side
    assembly = residue.parent.parent
    chain = residue.parent

    align_to_residue_plane(residue)
    # after the alignment, the assembly is translated so that the cube has positive xyz
    assembly.translate((radius, radius, radius))
    # create an empty array for discreet frame
    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side), dtype=np.uint8
    )
    frame.fill(0)
    # iterate through all atoms within the frame
    for atom in (
        a for a in assembly.get_atoms(ligands=False) if within_frame(radius, a)
    ):
        # 3d coordinates are converted to relative indices in frame array
        indices = discretize(atom, voxel_edge_length)
        ass = atom.parent.parent.parent
        cha = atom.parent.parent
        res = atom.parent
        assert (atom.element != "") or (atom.element != " "), (
            f"Atom element should not be blank:\n"
            f"{atom.chain}:{atom.res_num}:{atom.res_label}"
        )
        assert (res.mol_code != "") or (res.mol_code != " "), (
            f"Residue mol_code should not be blank:\n"
            f"{cha.id}:{res.id}:{atom.res_label}"
        )
        assert frame[indices] == 0, (
            f"Voxel should not be occupied: Currently "
            f"{frame[indices]}, "
            f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
        )
        element_data = ampal.data.ELEMENT_DATA[atom.element.capitalize()]
        frame[indices] = element_data["atomic number"]
    centre = voxels_per_side // 2
    assert (
        frame[centre, centre, centre] == 6
    ), f"The central atom should be carbon, but it is {frame[centre, centre, centre]}."
    unique_key = f"{assembly.id}:{chain.id}:{residue.id}:{residue.mol_code}"
    return (unique_key, frame)


def create_frames(
    polypeptide: ampal.Polypeptide, radius: float, voxels_per_side: int
) -> t.Dict[str, np.ndarray]:
    """Creates all discretized frames for all residues in the input polypeptide.
    
    Parameters
    ----------
    polypeptide : ampal.Polypeptide
        Frames will be created for all residues in this polypeptide.
    radius : float
        This term is slightly confusing as it's not really the radius as the frame is
        a cube. It is helf the edge length of the frame.
    voxels_per_side : int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    
    Returns
    -------
    frames : Dict[str, np.ndarray]
        
    """
    print(
        f"Longest possible distance in voxel: {np.sqrt(((radius/voxels_per_side)**2)*3)}"
    )
    assert isinstance(
        polypeptide, ampal.Polypeptide
    ), f"Expected an ampal.Polypeptide, got a {type(polypeptide)}."
    frames = dict(create_residue_frame(r, radius, voxels_per_side) for r in polypeptide)
    return frames
