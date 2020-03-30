"""Tools for creating a HDF5 data set of discretized structures."""

import multiprocessing as mp
from pathlib import Path
import time
from typing import List, Tuple, Union

import h5py  # type: ignore
import numpy as np  # type: ignore

from .discrete_structure import DiscreteStructure


EncodedStructureT = Tuple[str, np.ndarray, np.ndarray, np.ndarray]


def create_discrete_structure_v1(pdb_path: Path) -> EncodedStructureT:
    """Creates dump of all the enties in a numpy like format."""
    pdb_code = pdb_path.stem
    d_structure = DiscreteStructure.from_pdb_path(
        pdb_path,
        padding=0,  # no padding saves space and allows variable padding on load
        max_atom_distance=1.0,
        backbone_only=True,
        include_hydrogen=False,
        filter_monomers=("HOH",),
        gzipped=True,
    )
    assert len(list(d_structure.ca_atoms)) > 0, "No protein CA atoms found."
    indices_labels = [(ca.indices, ca.mol_code) for ca in d_structure.ca_atoms]
    # This looks ugly, there's neater ways to do this _i.e._ zip(*[...]) but this is the
    # clearest way to do this, as it is catastrophic if the indices and labels do not
    # match up!
    indices = np.array([x[0] for x in indices_labels], dtype=np.int32)
    labels = np.array([x[1] for x in indices_labels], dtype="|S3")
    assert len(labels) == len(indices) == len(indices_labels), (
        f"The number of labels ({len(labels)}) and the number of "
        f"indices ({len(indices)}) should match."
    )
    return pdb_code, d_structure.data, indices, labels


PROCESS_FNS = {"v1": create_discrete_structure_v1}


def attempt_structure_encode(
    path: Path, process_fn_version: str
) -> Union[EncodedStructureT, str]:
    """Trys to process a PDB path with a specified function."""
    try:
        encoded_structure = PROCESS_FNS[process_fn_version](path)
    except Exception as e:
        return f"An error occured while processing: {path}\n\t{type(e).__name__}\n\t{str(e)}"
    return encoded_structure


def _process_path(
    path_queue: mp.SimpleQueue,
    result_queue: mp.SimpleQueue,
    process_fn_version: str,
    errors: List[str],
):
    """Processes a path and puts the results into a queue."""
    while True:
        path = path_queue.get()
        result = attempt_structure_encode(path, process_fn_version)
        if isinstance(result, str):
            errors.append(result)
        else:
            result_queue.put(result)


def _save_results(
    result_queue: mp.SimpleQueue,
    h5_path: str,
    process_fn_version: str,
    complete: mp.Value,
):
    """Saves voxelized structures to a hdf5 object.

    Notes
    -----
    The hdf5 object itself is like a Python dict. The structure is
    simple:

        hdf5[pdb_code]
        ├─['data'] Full discreetized structure with ints representing atomic number.
        ├─['indices'] x, y, z indexes for all the CA atoms in the structure.
        └─['labels'] Labels for all of the CA atoms in the structure.
    """
    with h5py.File(str(h5_path), "w") as hd5:
        while True:
            result = result_queue.get()
            if result == "BREAK":
                break
            pdb_code, data, indices, labels = result
            hd5_group = hd5.create_group(pdb_code)
            hd5_group.attrs["process_fn_version"] = process_fn_version
            hd5_group.create_dataset("data", data=data)
            hd5_group.create_dataset("indices", data=indices)
            hd5_group.create_dataset("labels", data=labels)
            complete.value += 1


def process_paths(
    pdb_paths: List[Path], process_fn_version: str, processes: int, output_path: Path
) -> List[str]:
    """Discretizes a list of structures and stores them in a HDF5 object."""
    with mp.Manager() as manager:
        path_queue = manager.Queue()
        for path in pdb_paths:
            path_queue.put(path)
        result_queue = manager.Queue()
        complete = manager.Value("i", 0)
        errors = manager.list()
        total = len(pdb_paths)
        workers = [
            mp.Process(
                target=_process_path,
                args=(path_queue, result_queue, process_fn_version, errors),
            )
            for proc_i in range(processes - 1)
        ]
        storer = mp.Process(
            target=_save_results,
            args=(result_queue, output_path, process_fn_version, complete),
        )
        all_processes = workers + [storer]
        for proc in all_processes:
            proc.start()
        while (complete.value + len(errors)) < total:
            if not all([p.is_alive() for p in all_processes]):
                print("One or more of the processes died, aborting...")
                break
            print(f"{((complete.value + len(errors))/total)*100:.0f}% processed...")
            time.sleep(5)
        else:
            print(f"Finished processing files.")
            result_queue.put("BREAK")
            storer.join()
        for proc in all_processes:
            proc.terminate()
        errors = list(errors)
    return errors
