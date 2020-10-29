import pathlib

from ampal.data import ELEMENT_DATA
import numpy as np

# Config paths
MAKE_FRAME_DATASET_VER = "1.0.0"
PROJECT_ROOT_DIR = pathlib.Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT_DIR / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
ATOM_COLORS = {
    # Atomic number : Color
    0: ELEMENT_DATA['C']['CPK'],  # Carbon
    1: ELEMENT_DATA['N']['CPK'],  # Nitrogen
    2: ELEMENT_DATA['O']['CPK'],  # Oxygen
    3: "orange",  # +1
    4: "green"  # +2
}
PDB_PATH = DATA_FOLDER / "pdb"
PDB_PATH.mkdir(parents=True, exist_ok=True)
PDB_REQUEST_URL = "https://files.rcsb.org/download/"
PDB_CODES = ["1qys", "6ct4"]
HDF5_STRUCTURES_PATH = DATA_FOLDER / "frame_dataset.hdf5"
FETCH_PDB = True
UNCOMMON_RESIDUE_DICT = {"DLY": "LYS", "OTH": "THR", "GHP": "GLY", "YOF": "TYR", "HS9": "HIS", "HVA": "VAL", "C5C": "CYS", "TMD": "THR", "NC1": "SER", "CSR": "CYS", "LYP": "LYS", "PR4": "PRO", "KPI": "LYS", "02K": "ALA", "4AW": "TRP", "MLE": "LEU", "NMM": "ARG", "DNE": "LEU", "NYS": "CYS", "SEE": "SER", "DSG": "ASN", "ALA": "ALA", "CSA": "CYS", "SCH": "CYS", "TQQ": "TRP", "PTM": "TYR", "XPR": "PRO", "VLL": "UNK", "B3Y": "TYR", "PAQ": "TYR", "FME": "MET", "NAL": "ALA", "TYI": "TYR", "OXX": "ASP", "CSS": "CYS", "OCS": "CYS", "193": "UNK", "GLJ": "GLU", "PM3": "PHE", "DTR": "TRP", "MEQ": "GLN", "HSO": "HIS", "TYW": "TYR", "LED": "LEU", "PHL": "PHE", "TDD": "LEU", "MEA": "PHE", "FGA": "GLU", "GGL": "GLU", "PSH": "HIS", "3CF": "PHE", "MSE": "MET", "2SO": "HIS", "B3S": "SER", "PSW": "SEC", "C4R": "CYS", "XCP": "UNK", "LYF": "LYS", "WFP": "PHE", "A8E": "VAL", "0AF": "TRP", "PEC": "CYS", "JJJ": "CYS", "3TY": "UNK", "SVY": "SER", "DIL": "ILE", "MHS": "HIS", "MME": "MET", "MMO": "ARG", "B3A": "ALA", "CHG": "UNK", "PHI": "PHE", "AR2": "ARG", "MND": "ASN", "BTR": "TRP", "AEI": "ASP", "TIH": "ALA", "DDE": "HIS", "S1H": "SER", "DSE": "SER", "AR4": "GLU", "FDL": "LYS", "PRJ": "PRO", "CY3": "CYS", "2TY": "TYR", "AR7": "ARG", "CTH": "THR", "DTY": "TYR", "SYS": "CYS", "C1X": "LYS", "SVV": "SER", "ASN": "ASN", "SNC": "CYS", "AKZ": "ASP", "OMY": "TYR", "JJL": "CYS", "XSN": "ASN", "0UO": "TRP", "TCQ": "TYR", "OSE": "SER", "NPH": "CYS", "0A0": "ASP", "1PA": "PHE", "SIC": "CYS", "TY8": "TYR", "AYA": "ALA", "ALN": "ALA", "SXE": "SER", "B3T": "UNK", "BB9": "CYS", "HL2": "LEU", "0AR": "ARG", "SVA": "SER", "DBB": "THR", "KPY": "LYS", "DPP": "ALA", "32S": "UNK", "FGL": "GLY", "N80": "PRO", "IGL": "GLY", "PF5": "PHE", "OYL": "HIS", "MNL": "LEU", "PBF": "PHE", "CEA": "CYS", "OHI": "HIS", "ESC": "MET", "2JG": "SER", "1X6": "SER", "4BF": "TYR", "MAA": "ALA", "3X9": "CYS", "BFD": "ASP", "CZ2": "CYS", "23P": "ALA", "I4G": "GLY", "CMT": "CYS", "LVN": "VAL", "OAS": "SER", "TY2": "TYR", "SCS": "CYS", "PFX": "UNK", "MF3": "UNK", "OBS": "LYS", "GL3": "GLY", "0A9": "PHE", "MVA": "VAL", "B3Q": "UNK", "DOA": "UNK", "MP8": "PRO", "CYR": "CYS", "5PG": "GLY", "ILY": "LYS", "DNW": "ALA", "BCX": "CYS", "AZK": "LYS", "AAR": "ARG", "TRN": "TRP", "NBQ": "TYR", "RVX": "SER", "PSA": "PHE", "Z3E": "THR", "OCY": "CYS", "2ZC": "SER", "N2C": "UNK", "SBD": "SER", "MSA": "GLY", "SET": "SER", "HS8": "HIS", "SMF": "PHE", "HYP": "PRO", "PYX": "CYS", "XPL": "PYL", "DMK": "ASP", "BIF": "PHE", "M3L": "LYS", "CYF": "CYS", "O12": "UNK", "SRZ": "SER", "LAL": "ALA", "2MR": "ARG", "4PH": "PHE", "2LT": "TYR", "LPL": "UNK", "3YM": "TYR", "LRK": "LYS", "FVA": "VAL", "MED": "MET", "ILM": "ILE", "6CL": "LYS", "CXM": "MET", "DHV": "VAL", "PR3": "CYS", "HAR": "ARG", "KWS": "GLY", "SAR": "GLY", "0LF": "PRO", "45F": "PRO", "12A": "A","CLG": "LYS", "DHI": "HIS", "PTR": "TYR", "DMT": "UNK", "OMT": "MET", "TBG": "VAL", "PLJ": "PRO", "IAM": "ALA", "DBY": "TYR", "CPC": "UNK", "GLZ": "GLY", "4FW": "TRP", "SLZ": "LYS", "HIA": "HIS", "FOE": "CYS", "IYR": "TYR", "KST": "LYS", "B3M": "UNK", "BB6": "CYS", "CYW": "CYS", "MPQ": "GLY", "HHK": "LYS", "HGL": "UNK", "SE7": "ALA", "ELY": "LYS", "TRO": "TRP", "DNP": "ALA", "MK8": "LEU", "200": "PHE", "WVL": "VAL", "LPD": "PRO", "NCB": "ALA", "DDZ": "ALA", "MYK": "LYS", "OLD": "HIS", "DYS": "CYS", "LET": "LYS", "ESB": "TYR", "HR7": "ARG", "DI7": "TYR", "QCS": "CYS", "ASA": "ASP", "CSX": "CYS", "P3Q": "TYR", "OHS": "ASP", "SOY": "SER", "EHP": "PHE", "ZCL": "PHE", "32T": "UNK", "AHB": "ASN", "TRX": "TRP", "0AK": "ASP", "TH5": "THR", "GHG": "GLN", "XW1": "ALA", "23F": "PHE", "1OP": "TYR", "AGT": "CYS", "PYA": "ALA", "2MT": "PRO", "4FB": "PRO", "CSB": "CYS", "TRQ": "TRP", "MDO": "GLY", "CAS": "CYS", "TTQ": "TRP", "T0I": "TYR", "LLY": "LYS", "GVL": "SER", "BPE": "CYS", "0TD": "ASP", "TYY": "TYR", "BH2": "ASP", "D3P": "GLY", "CY4": "CYS", "CHP": "GLY", "DFO": "UNK", "NLB": "LEU", "QPH": "PHE", "DTH": "THR", "LLO": "LYS", "LYN": "LYS", "DPN": "PHE", "EFC": "CYS", "FP9": "PRO", "OMX": "TYR", "AGQ": "TYR", "PHD": "ASP", "PR9": "PRO", "B3L": "UNK", "LYX": "LYS", "IT1": "LYS", "DBU": "THR", "0A8": "CYS", "TYX": "UNK", "QMM": "GLN", "CME": "CYS", "ACB": "ASP", "TRF": "TRP", "HOX": "PHE", "DA2": "ARG", "DNS": "LYS", "BIL": "UNK", "SUN": "SER", "TYJ": "TYR", "3PX": "PRO", "CLD": "SER", "IPG": "GLY", "CLH": "LYS", "XCN": "CYS", "CZZ": "CYS", "THO": "UNK", "CY1": "CYS", "CYS": "CYS", "PFF": "PHE", "MLL": "LEU", "PG1": "SER", "BMT": "THR", "CSZ": "CYS", "DSN": "SER", "NIY": "TYR", "FH7": "LYS", "CGV": "CYS", "SVZ": "SER", "ORQ": "ARG", "DLS": "LYS", "DVA": "VAL", "BHD": "ASP", "TPQ": "TYR", "STY": "TYR", "CSP": "CYS", "31Q": "CYS", "B3E": "GLU", "LEF": "LEU", "GLH": "GLU", "LCK": "LYS", "GME": "GLU", "FHO": "LYS", "MDH": "UNK", "ECC": "GLN", "34E": "VAL", "ASB": "ASP", "HCS": "UNK", "KYN": "TRP", "OIC": "UNK", "VR0": "ARG", "U2X": "TYR", "PHE": "PHE", "TYS": "TYR", "SBG": "SER", "A5N": "ASN", "CYD": "CYS", "4DP": "TRP", "3AH": "HIS", "FCL": "PHE", "PRV": "GLY", "CYQ": "CYS", "MBQ": "TYR", "DAS": "ASP", "CS4": "CYS", "B3K": "LYS", "NLE": "LEU", "143": "CYS", "PR7": "PRO", "DAH": "PHE", "LE1": "VAL", "TQZ": "CYS", "LGY": "LYS", "CML": "CYS", "CSW": "CYS", "N10": "SER", "2RX": "SER", "TOQ": "TRP", "0AH": "SER", "P2Q": "TYR", "CYG": "CYS", "DGL": "GLU", "KOR": "MET", "DAR": "ARG", "2ML": "LEU", "PTH": "TYR", "CCS": "CYS", "HMR": "ARG", "33X": "ALA", "UN2": "UNK", "IML": "ILE", "4CY": "MET", "ZZJ": "ALA", "DFI": "UNK", "TIS": "SER", "LLP": "LYS", "MHU": "PHE", "QPA": "CYS", "175": "GLY", "SAH": "CYS", "IIL": "ILE", "BCS": "CYS", "R4K": "TRP", "TYQ": "TYR", "NCY": "UNK", "FT6": "TRP", "OBF": "UNK", "0CS": "ALA", "4HL": "TYR", "TXY": "TYR", "DOH": "ASP", "CSE": "CYS", "DAB": "ALA", "GLK": "GLU", "TYN": "TYR", "LEI": "VAL", "M0H": "CYS", "CLB": "SER", "MGG": "ARG", "CGU": "GLU", "UF0": "SER", "SLL": "LYS", "ML3": "LYS", "HPH": "PHE", "SME": "MET", "ALC": "ALA", "ASL": "ASP", "CHS": "UNK", "2TL": "THR", "HT7": "TRP", "SGB": "SER", "OPR": "ARG", "B3D": "ASP", "FLT": "TYR", "DGN": "GLN", "4CF": "PHE", "HLU": "LEU", "FZN": "LYS", "C6C": "CYS", "HTI": "CYS", "OMH": "SER", "WLU": "LEU", "23S": "UNK", "U3X": "PHE", "SEB": "SER", "DBZ": "ALA", "BB7": "CYS", "2RA": "ALA", "SCY": "CYS", "6CW": "TRP", "AHP": "ALA", "ARO": "ARG", "RE3": "TRP", "1TQ": "TRP", "VDL": "UNK", "4IN": "TRP", "GFT": "SER", "CPI": "UNK", "LSO": "LYS", "CGA": "GLU", "MLZ": "LYS", "HTR": "TRP", "00C": "CYS", "FAK": "LYS", "PRS": "PRO", "ME0": "MET", "SDP": "SER", "HSL": "SER", "C3Y": "CYS", "823": "ASN", "PHA": "PHE", "LYZ": "LYS", "HTN": "ASN", "LP6": "LYS", "ALV": "ALA", "NVA": "VAL", "CSD": "CYS", "DMH": "ASN", "PG9": "GLY", "PCA": "GLU", "KCX": "LYS", "MDF": "TYR", "TYB": "TYR", "MHL": "LEU", "GNC": "GLN", "NLO": "LEU", "MEN": "ASN", "POM": "PRO", "2HF": "HIS", "CY0": "CYS", "ZYK": "PRO", "R1A": "CYS", "CAF": "CYS", "YCM": "CYS", "ORN": "ALA", "H5M": "PRO", "MLY": "LYS", "KYQ": "LYS", "DPQ": "TYR", "MIS": "SER", "TPO": "THR", "XX1": "LYS", "SMC": "CYS", "DHA": "SER", "MGN": "GLN", "FLA": "ALA", "ILX": "ILE", "QIL": "ILE", "2KP": "LYS", "CS1": "CYS", "HNC": "CYS", "PRK": "LYS", "LYR": "LYS", "DM0": "LYS", "TSY": "CYS", "NYB": "CYS", "MHO": "MET", "KFP": "LYS", "SEN": "SER", "999": "ASP", "VLM": "UNK", "CMH": "CYS", "ONL": "UNK", "M2L": "LYS", "LME": "GLU", "AIB": "ALA", "CYJ": "LYS", "CS3": "CYS", "WPA": "PHE", "MTY": "TYR", "MIR": "SER", "HZP": "PRO", "LTA": "UNK", "HIP": "HIS", "PPN": "PHE", "APK": "LYS", "HPE": "PHE", "SVX": "SER", "JJK": "CYS", "03Y": "CYS", "D4P": "UNK", "1AC": "ALA", "B3X": "ASN", "0FL": "ALA", "2KK": "LYS", "LMQ": "GLN", "RE0": "TRP", "MSO": "MET", "ZYJ": "PRO", "GMA": "GLU", "DPR": "PRO", "1TY": "TYR", "TOX": "TRP", "DPL": "PRO", "M2S": "MET", "4HT": "TRP", "BUC": "CYS", "C1S": "CYS", "TA4": "UNK", "CSO": "CYS", "5CW": "TRP", "TRW": "TRP", "DCY": "CYS", "DAL": "ALA", "0QL": "CYS", "THC": "THR", "FGP": "SER", "MCS": "CYS", "AZH": "ALA", "HIQ": "HIS", "ABA": "ASN", "TH6": "THR", "FHL": "LYS", "ZAL": "ALA", "ICY": "CYS", "IZO": "MET", "F2F": "PHE", "VAI": "VAL", "TY5": "TYR", "07O": "CYS", "AA4": "ALA", "RGL": "ARG", "SAC": "SER", "PXU": "PRO", "NFA": "PHE", "LA2": "LYS", "0BN": "PHE", "LYK": "LYS", "FTY": "TYR", "NZH": "HIS", "CSJ": "CYS", "30V": "CYS", "DLE": "LEU", "TLY": "LYS", "L3O": "LEU", "LDH": "LYS", "NEP": "HIS", "ALY": "LYS", "GPL": "LYS", "01W": "UNK", "WRP": "TRP", "MCL": "LYS", "2AS": "UNK", "CSU": "CYS", "SOC": "CYS", "HRG": "ARG", "NMC": "GLY", "TYO": "TYR", "LHC": "UNK", "D11": "THR", "I2M": "ILE", "TTS": "TYR", "FC0": "PHE", "HIC": "HIS", "YPZ": "TYR", "5CS": "CYS", "SEP": "SER", "BBC": "CYS", "3MY": "TYR", "HQA": "ALA", "11Q": "PRO", "AGM": "ARG", "BG1": "SER", "IAS": "ASP", "SBL": "SER", "56A": "HIS", "FTR": "TRP", "DIV": "VAL", "ALO": "THR", "BTK": "LYS"}
UNCOMMON_RES_CONVERSION = True
GAUSSIAN_ATOMS = {
    "0" : np.array(
        [[[0.002195, 0.01688, 0.002195],
          [0.01688, 0.1299, 0.01688],
          [0.002195, 0.01688, 0.002195]],

         [[0.01688, 0.1299, 0.01688],
          [0.1299, 1., 0.1299],
          [0.01688, 0.1299, 0.01688]],

         [[0.002195, 0.01688, 0.002195],
          [0.01688, 0.1299, 0.01688],
          [0.002195, 0.01688, 0.002195]]]),

    "1" : np.array(
        [[[8.240e-04, 8.789e-03, 8.240e-04],
          [8.789e-03, 9.375e-02, 8.789e-03],
          [8.240e-04, 8.789e-03, 8.240e-04]],

         [[8.789e-03, 9.375e-02, 8.789e-03],
          [9.375e-02, 1.000e+00, 9.375e-02],
          [8.789e-03, 9.375e-02, 8.789e-03]],

         [[8.240e-04, 8.789e-03, 8.240e-04],
          [8.789e-03, 9.375e-02, 8.789e-03],
          [8.240e-04, 8.789e-03, 8.240e-04]]]),

    "2" : np.array(
        [[[2.397e-04, 3.870e-03, 2.397e-04],
          [3.870e-03, 6.219e-02, 3.870e-03],
          [2.397e-04, 3.870e-03, 2.397e-04]],

         [[3.870e-03, 6.219e-02, 3.870e-03],
          [6.219e-02, 1.000e+00, 6.219e-02],
          [3.870e-03, 6.219e-02, 3.870e-03]],

         [[2.397e-04, 3.870e-03, 2.397e-04],
          [3.870e-03, 6.219e-02, 3.870e-03],
          [2.397e-04, 3.870e-03, 2.397e-04]]])

}
ATOMIC_CENTER = (1, 1, 1)