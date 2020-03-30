# aposteriori-dnn

# Dataset

TODO: Instructions to download
TODO: Explain 

# Data Preparation

## Discretization  

Initial discretization is done by the `aposteriori.create_data_set.py` which
 discretizes the structures into voxels. Each voxel can be either empty (0
 ) or contain an atom. Atomic numbers are used to encode atoms. For
  convenience, H atoms are ignored. 

`max_atom_distance` is the distance required for two atoms to be distinguished as two separate voxels and is set to 1 Armstrong by default (which should cover most bond lengths, except for H-F which does not occur in proteins). This means that for each voxel representing an atom, the distance between two opposite corners is 1 Armstrong. 

This means that 1 voxel of atom corresponds to a cube of 0.58x0.58x0.58 armstrong (1 A  = 3x**2 where x is the edge of the voxel). This then saved into an hdf5 file. 

```
hdf5[pdb_code]
├─['data'] Full discreetized structure with ints representing atomic number.
├─['indices'] x, y, z indexes for all the CA atoms in the structure.
└─['labels'] Labels for all of the CA atoms in the structure.
```

Further details are available in `aposteriori.discrete_structure.py`.


## Creation of Residue Frames 

The 3D structure is padded to allow for residue frames to be extracted. The function `make_data_points` creates a residue frame at each amino acid position. Each residue frame contains an alpha-Carbon atom (Ca) at the very centre of the frame. The size of the residue frame is determined by the `radius` parameter. Figure 1 shows a graphical representation of how this looks like. 


![alt text][fig-protbox]

[fig-protbox]: img/voxelization.png
**Figure 1.** A graphical representation of the residue frames. Carbon atoms are grey, Oxygen red, Nitrogen blue. Ca is present at the centre of the residue frame. 



