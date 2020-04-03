import logging
import pathlib
from datetime import datetime


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.keras.backend
from tensorflow.keras.optimizers import Adam

# Config paths
PROJECT_ROOT_DIR = pathlib.Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT_DIR / "data"
TRAINING_PATH = DATA_FOLDER / "cnos_contig_training_balanced_small.pickle"
VALIDATION_PATH = DATA_FOLDER / "cnos_contig_validation_balanced_small.pickle"
# TODO: Use more appropriate names:
TRAINING_PATH_FRAME = DATA_FOLDER / "cnos_training_balanced_r20.pickle"
VALIDATION_PATH_FRAME = DATA_FOLDER / "cnos_validation_balanced_r20.pickle"
# TRAINING_PATH_FRAME = "cnos_training_balanced_r20.pickle"
# VALIDATION_PATH_FRAME = "cnos_validation_balanced_r20.pickle"
# TODO: Refactor below:
# PIECES_DATA_PATH = pathlib.Path(
#     "../../structural_data/dnn_data/2-pc90-pieces-high-res.h5"
# )
PIECES_DATA_PATH = DATA_FOLDER / "2-pc90-pieces-high-res.h5"

CURRENT_DATE = datetime.now().strftime("_%Y-%m-%d %H:%M:%S")
BASE_MODEL = "2c"
NAME_MODEL = BASE_MODEL + "_prodcnn"
# TODO: Output folder may need identifier like random number to avoid two
#  programs running at the same time overriding each others.
OUTPUT_DIR = PROJECT_ROOT_DIR / (NAME_MODEL + CURRENT_DATE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_PATH = DATA_FOLDER / "encoders.pickle"
CONV_LSTM_MODEL = PROJECT_ROOT_DIR / "model" / "conv_lstm.h5"
FRAME_CONV_MODEL = PROJECT_ROOT_DIR / "model" / "frame_conv.h5"
LOG_FILENAME = OUTPUT_DIR / ('logs.txt')
logging.basicConfig(filename=LOG_FILENAME,
                    format='%(asctime)s - %(name)s - '
                           '%(levelname)s - %(message)s',
                    level=logging.INFO)

# TensorBoard:
TENSORBOARD_OUTPUT_DIR = OUTPUT_DIR / "tb_logs"
TENSORBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WRITE_GRADS = False  # write_grads = True breaks 1D CNN
HISTOGRAM_FREQ = 0
UPDATE_FREQ = "batch"

# Local configs:
RADIUS = 20
EDGE_LENGTH = RADIUS*2 + 1
UNITS = 11
ATOMIC_NUMBERS = [0, 6, 7, 8, 16]
WANDERWAAL_RADII = [0, 0.70, 0.65, 0.60, 0] # order by idx 0:0, 6:1, 7:2, 8:3, 16:4
# TODO refactor name below for contig
INPUT_SHAPE = (UNITS, EDGE_LENGTH, EDGE_LENGTH, EDGE_LENGTH, len(ATOMIC_NUMBERS))
INPUT_SHAPE_FRAME = (EDGE_LENGTH, EDGE_LENGTH, EDGE_LENGTH, len(ATOMIC_NUMBERS))

# Network:
BATCH_SIZE = 64
EPOCHS = 200
LOSS_FUNC = "categorical_crossentropy"
ACTIVATION_FUNC = "relu"
METRICS = ["accuracy", "top_3_cat_acc"]
OPTIMIZER = Adam()
SHUFFLE = True
WORKERS = 8
MULTIPROCESSING = True

# Activation Layer Visualization:
VISUALIZE_ACTIVATION_AFTER_TRAINING = True
ATOM_COLORS = {
    # Atomic number : Color
    6: "black",  # Carbon
    7: "blue",  # Nitrogen
    8: "red",  # Oxygen
    16: "orange",  # Sulphur
}
FIG_SIZE = (10, 5)
COLOR_MAP = plt.cm.rainbow
LOCAL_COLOR_MAP = True
ACTIVATION_ALPHA = 0.05
PLOT_DIR = OUTPUT_DIR / ("activation_plot_" + NAME_MODEL)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Entropy Visualization:
VISUALIZE_ENTROPY_AFTER_TRAINING = True
PDB_PATH = DATA_FOLDER / "pdb"
PDB_PATH.mkdir(parents=True, exist_ok=True)
SAVE_ANNOTATED_PDB_TO_FILE = True
ANNOTATED_ENTROPY_PDB_PATH = OUTPUT_DIR / "pdb_entropy"
if SAVE_ANNOTATED_PDB_TO_FILE:
    ANNOTATED_ENTROPY_PDB_PATH.mkdir(parents=True, exist_ok=True)
PDB_REQUEST_URL = "https://files.rcsb.org/download/"
PDB_CODES = ["1qys", "6ct4"]
H5_STRUCTURES_PATH = DATA_FOLDER / "pdb_voxelised.h5"
REBUILD_H5_DATASET = True
FETCH_PDB = True

# Hardware:
config = tf.ConfigProto()
# Allow GPU
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tensorflow.keras.backend.set_session(session)
# Check GPU
devices = device_lib.list_local_devices()
[print(d.name) for d in devices]

RESIDUES_THREE_TO_ONE_LETTER = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
UNCOMMON_RESIDUE_DICT = {"DLY": "LYS", "OTH": "THR", "GHP": "GLY", "YOF": "TYR", "HS9": "HIS", "HVA": "VAL", "C5C": "CYS", "TMD": "THR", "NC1": "SER", "CSR": "CYS", "LYP": "LYS", "PR4": "PRO", "KPI": "LYS", "02K": "ALA", "4AW": "TRP", "MLE": "LEU", "NMM": "ARG", "DNE": "LEU", "NYS": "CYS", "SEE": "SER", "DSG": "ASN", "ALA": "ALA", "CSA": "CYS", "SCH": "CYS", "TQQ": "TRP", "PTM": "TYR", "XPR": "PRO", "VLL": "UNK", "B3Y": "TYR", "PAQ": "TYR", "FME": "MET", "NAL": "ALA", "TYI": "TYR", "OXX": "ASP", "CSS": "CYS", "OCS": "CYS", "193": "UNK", "GLJ": "GLU", "PM3": "PHE", "DTR": "TRP", "MEQ": "GLN", "HSO": "HIS", "TYW": "TYR", "LED": "LEU", "PHL": "PHE", "TDD": "LEU", "MEA": "PHE", "FGA": "GLU", "GGL": "GLU", "PSH": "HIS", "3CF": "PHE", "MSE": "MET", "2SO": "HIS", "B3S": "SER", "PSW": "SEC", "C4R": "CYS", "XCP": "UNK", "LYF": "LYS", "WFP": "PHE", "A8E": "VAL", "0AF": "TRP", "PEC": "CYS", "JJJ": "CYS", "3TY": "UNK", "SVY": "SER", "DIL": "ILE", "MHS": "HIS", "MME": "MET", "MMO": "ARG", "B3A": "ALA", "CHG": "UNK", "PHI": "PHE", "AR2": "ARG", "MND": "ASN", "BTR": "TRP", "AEI": "ASP", "TIH": "ALA", "DDE": "HIS", "S1H": "SER", "DSE": "SER", "AR4": "GLU", "FDL": "LYS", "PRJ": "PRO", "CY3": "CYS", "2TY": "TYR", "AR7": "ARG", "CTH": "THR", "DTY": "TYR", "SYS": "CYS", "C1X": "LYS", "SVV": "SER", "ASN": "ASN", "SNC": "CYS", "AKZ": "ASP", "OMY": "TYR", "JJL": "CYS", "XSN": "ASN", "0UO": "TRP", "TCQ": "TYR", "OSE": "SER", "NPH": "CYS", "0A0": "ASP", "1PA": "PHE", "SIC": "CYS", "TY8": "TYR", "AYA": "ALA", "ALN": "ALA", "SXE": "SER", "B3T": "UNK", "BB9": "CYS", "HL2": "LEU", "0AR": "ARG", "SVA": "SER", "DBB": "THR", "KPY": "LYS", "DPP": "ALA", "32S": "UNK", "FGL": "GLY", "N80": "PRO", "IGL": "GLY", "PF5": "PHE", "OYL": "HIS", "MNL": "LEU", "PBF": "PHE", "CEA": "CYS", "OHI": "HIS", "ESC": "MET", "2JG": "SER", "1X6": "SER", "4BF": "TYR", "MAA": "ALA", "3X9": "CYS", "BFD": "ASP", "CZ2": "CYS", "23P": "ALA", "I4G": "GLY", "CMT": "CYS", "LVN": "VAL", "OAS": "SER", "TY2": "TYR", "SCS": "CYS", "PFX": "UNK", "MF3": "UNK", "OBS": "LYS", "GL3": "GLY", "0A9": "PHE", "MVA": "VAL", "B3Q": "UNK", "DOA": "UNK", "MP8": "PRO", "CYR": "CYS", "5PG": "GLY", "ILY": "LYS", "DNW": "ALA", "BCX": "CYS", "AZK": "LYS", "AAR": "ARG", "TRN": "TRP", "NBQ": "TYR", "RVX": "SER", "PSA": "PHE", "Z3E": "THR", "OCY": "CYS", "2ZC": "SER", "N2C": "UNK", "SBD": "SER", "MSA": "GLY", "SET": "SER", "HS8": "HIS", "SMF": "PHE", "HYP": "PRO", "PYX": "CYS", "XPL": "PYL", "DMK": "ASP", "BIF": "PHE", "M3L": "LYS", "CYF": "CYS", "O12": "UNK", "SRZ": "SER", "LAL": "ALA", "2MR": "ARG", "4PH": "PHE", "2LT": "TYR", "LPL": "UNK", "3YM": "TYR", "LRK": "LYS", "FVA": "VAL", "MED": "MET", "ILM": "ILE", "6CL": "LYS", "CXM": "MET", "DHV": "VAL", "PR3": "CYS", "HAR": "ARG", "KWS": "GLY", "SAR": "GLY", "0LF": "PRO", "45F": "PRO", "12A": "A","CLG": "LYS", "DHI": "HIS", "PTR": "TYR", "DMT": "UNK", "OMT": "MET", "TBG": "VAL", "PLJ": "PRO", "IAM": "ALA", "DBY": "TYR", "CPC": "UNK", "GLZ": "GLY", "4FW": "TRP", "SLZ": "LYS", "HIA": "HIS", "FOE": "CYS", "IYR": "TYR", "KST": "LYS", "B3M": "UNK", "BB6": "CYS", "CYW": "CYS", "MPQ": "GLY", "HHK": "LYS", "HGL": "UNK", "SE7": "ALA", "ELY": "LYS", "TRO": "TRP", "DNP": "ALA", "MK8": "LEU", "200": "PHE", "WVL": "VAL", "LPD": "PRO", "NCB": "ALA", "DDZ": "ALA", "MYK": "LYS", "OLD": "HIS", "DYS": "CYS", "LET": "LYS", "ESB": "TYR", "HR7": "ARG", "DI7": "TYR", "QCS": "CYS", "ASA": "ASP", "CSX": "CYS", "P3Q": "TYR", "OHS": "ASP", "SOY": "SER", "EHP": "PHE", "ZCL": "PHE", "32T": "UNK", "AHB": "ASN", "TRX": "TRP", "0AK": "ASP", "TH5": "THR", "GHG": "GLN", "XW1": "ALA", "23F": "PHE", "1OP": "TYR", "AGT": "CYS", "PYA": "ALA", "2MT": "PRO", "4FB": "PRO", "CSB": "CYS", "TRQ": "TRP", "MDO": "GLY", "CAS": "CYS", "TTQ": "TRP", "T0I": "TYR", "LLY": "LYS", "GVL": "SER", "BPE": "CYS", "0TD": "ASP", "TYY": "TYR", "BH2": "ASP", "D3P": "GLY", "CY4": "CYS", "CHP": "GLY", "DFO": "UNK", "NLB": "LEU", "QPH": "PHE", "DTH": "THR", "LLO": "LYS", "LYN": "LYS", "DPN": "PHE", "EFC": "CYS", "FP9": "PRO", "OMX": "TYR", "AGQ": "TYR", "PHD": "ASP", "PR9": "PRO", "B3L": "UNK", "LYX": "LYS", "IT1": "LYS", "DBU": "THR", "0A8": "CYS", "TYX": "UNK", "QMM": "GLN", "CME": "CYS", "ACB": "ASP", "TRF": "TRP", "HOX": "PHE", "DA2": "ARG", "DNS": "LYS", "BIL": "UNK", "SUN": "SER", "TYJ": "TYR", "3PX": "PRO", "CLD": "SER", "IPG": "GLY", "CLH": "LYS", "XCN": "CYS", "CZZ": "CYS", "THO": "UNK", "CY1": "CYS", "CYS": "CYS", "PFF": "PHE", "MLL": "LEU", "PG1": "SER", "BMT": "THR", "CSZ": "CYS", "DSN": "SER", "NIY": "TYR", "FH7": "LYS", "CGV": "CYS", "SVZ": "SER", "ORQ": "ARG", "DLS": "LYS", "DVA": "VAL", "BHD": "ASP", "TPQ": "TYR", "STY": "TYR", "CSP": "CYS", "31Q": "CYS", "B3E": "GLU", "LEF": "LEU", "GLH": "GLU", "LCK": "LYS", "GME": "GLU", "FHO": "LYS", "MDH": "UNK", "ECC": "GLN", "34E": "VAL", "ASB": "ASP", "HCS": "UNK", "KYN": "TRP", "OIC": "UNK", "VR0": "ARG", "U2X": "TYR", "PHE": "PHE", "TYS": "TYR", "SBG": "SER", "A5N": "ASN", "CYD": "CYS", "4DP": "TRP", "3AH": "HIS", "FCL": "PHE", "PRV": "GLY", "CYQ": "CYS", "MBQ": "TYR", "DAS": "ASP", "CS4": "CYS", "B3K": "LYS", "NLE": "LEU", "143": "CYS", "PR7": "PRO", "DAH": "PHE", "LE1": "VAL", "TQZ": "CYS", "LGY": "LYS", "CML": "CYS", "CSW": "CYS", "N10": "SER", "2RX": "SER", "TOQ": "TRP", "0AH": "SER", "P2Q": "TYR", "CYG": "CYS", "DGL": "GLU", "KOR": "MET", "DAR": "ARG", "2ML": "LEU", "PTH": "TYR", "CCS": "CYS", "HMR": "ARG", "33X": "ALA", "UN2": "UNK", "IML": "ILE", "4CY": "MET", "ZZJ": "ALA", "DFI": "UNK", "TIS": "SER", "LLP": "LYS", "MHU": "PHE", "QPA": "CYS", "175": "GLY", "SAH": "CYS", "IIL": "ILE", "BCS": "CYS", "R4K": "TRP", "TYQ": "TYR", "NCY": "UNK", "FT6": "TRP", "OBF": "UNK", "0CS": "ALA", "4HL": "TYR", "TXY": "TYR", "DOH": "ASP", "CSE": "CYS", "DAB": "ALA", "GLK": "GLU", "TYN": "TYR", "LEI": "VAL", "M0H": "CYS", "CLB": "SER", "MGG": "ARG", "CGU": "GLU", "UF0": "SER", "SLL": "LYS", "ML3": "LYS", "HPH": "PHE", "SME": "MET", "ALC": "ALA", "ASL": "ASP", "CHS": "UNK", "2TL": "THR", "HT7": "TRP", "SGB": "SER", "OPR": "ARG", "B3D": "ASP", "FLT": "TYR", "DGN": "GLN", "4CF": "PHE", "HLU": "LEU", "FZN": "LYS", "C6C": "CYS", "HTI": "CYS", "OMH": "SER", "WLU": "LEU", "23S": "UNK", "U3X": "PHE", "SEB": "SER", "DBZ": "ALA", "BB7": "CYS", "2RA": "ALA", "SCY": "CYS", "6CW": "TRP", "AHP": "ALA", "ARO": "ARG", "RE3": "TRP", "1TQ": "TRP", "VDL": "UNK", "4IN": "TRP", "GFT": "SER", "CPI": "UNK", "LSO": "LYS", "CGA": "GLU", "MLZ": "LYS", "HTR": "TRP", "00C": "CYS", "FAK": "LYS", "PRS": "PRO", "ME0": "MET", "SDP": "SER", "HSL": "SER", "C3Y": "CYS", "823": "ASN", "PHA": "PHE", "LYZ": "LYS", "HTN": "ASN", "LP6": "LYS", "ALV": "ALA", "NVA": "VAL", "CSD": "CYS", "DMH": "ASN", "PG9": "GLY", "PCA": "GLU", "KCX": "LYS", "MDF": "TYR", "TYB": "TYR", "MHL": "LEU", "GNC": "GLN", "NLO": "LEU", "MEN": "ASN", "POM": "PRO", "2HF": "HIS", "CY0": "CYS", "ZYK": "PRO", "R1A": "CYS", "CAF": "CYS", "YCM": "CYS", "ORN": "ALA", "H5M": "PRO", "MLY": "LYS", "KYQ": "LYS", "DPQ": "TYR", "MIS": "SER", "TPO": "THR", "XX1": "LYS", "SMC": "CYS", "DHA": "SER", "MGN": "GLN", "FLA": "ALA", "ILX": "ILE", "QIL": "ILE", "2KP": "LYS", "CS1": "CYS", "HNC": "CYS", "PRK": "LYS", "LYR": "LYS", "DM0": "LYS", "TSY": "CYS", "NYB": "CYS", "MHO": "MET", "KFP": "LYS", "SEN": "SER", "999": "ASP", "VLM": "UNK", "CMH": "CYS", "ONL": "UNK", "M2L": "LYS", "LME": "GLU", "AIB": "ALA", "CYJ": "LYS", "CS3": "CYS", "WPA": "PHE", "MTY": "TYR", "MIR": "SER", "HZP": "PRO", "LTA": "UNK", "HIP": "HIS", "PPN": "PHE", "APK": "LYS", "HPE": "PHE", "SVX": "SER", "JJK": "CYS", "03Y": "CYS", "D4P": "UNK", "1AC": "ALA", "B3X": "ASN", "0FL": "ALA", "2KK": "LYS", "LMQ": "GLN", "RE0": "TRP", "MSO": "MET", "ZYJ": "PRO", "GMA": "GLU", "DPR": "PRO", "1TY": "TYR", "TOX": "TRP", "DPL": "PRO", "M2S": "MET", "4HT": "TRP", "BUC": "CYS", "C1S": "CYS", "TA4": "UNK", "CSO": "CYS", "5CW": "TRP", "TRW": "TRP", "DCY": "CYS", "DAL": "ALA", "0QL": "CYS", "THC": "THR", "FGP": "SER", "MCS": "CYS", "AZH": "ALA", "HIQ": "HIS", "ABA": "ASN", "TH6": "THR", "FHL": "LYS", "ZAL": "ALA", "ICY": "CYS", "IZO": "MET", "F2F": "PHE", "VAI": "VAL", "TY5": "TYR", "07O": "CYS", "AA4": "ALA", "RGL": "ARG", "SAC": "SER", "PXU": "PRO", "NFA": "PHE", "LA2": "LYS", "0BN": "PHE", "LYK": "LYS", "FTY": "TYR", "NZH": "HIS", "CSJ": "CYS", "30V": "CYS", "DLE": "LEU", "TLY": "LYS", "L3O": "LEU", "LDH": "LYS", "NEP": "HIS", "ALY": "LYS", "GPL": "LYS", "01W": "UNK", "WRP": "TRP", "MCL": "LYS", "2AS": "UNK", "CSU": "CYS", "SOC": "CYS", "HRG": "ARG", "NMC": "GLY", "TYO": "TYR", "LHC": "UNK", "D11": "THR", "I2M": "ILE", "TTS": "TYR", "FC0": "PHE", "HIC": "HIS", "YPZ": "TYR", "5CS": "CYS", "SEP": "SER", "BBC": "CYS", "3MY": "TYR", "HQA": "ALA", "11Q": "PRO", "AGM": "ARG", "BG1": "SER", "IAS": "ASP", "SBL": "SER", "56A": "HIS", "FTR": "TRP", "DIV": "VAL", "ALO": "THR", "BTK": "LYS"}
UNCOMMON_RES_CONVERSION = True
BLOSUM_DICT = {("W", "F"): 1,("L", "R"): -2,("S", "P"): -1,("V", "T"): 0,("N", "A"): -2,("Z", "Y"): -2,("W", "R"): -3,("Q", "A"): -1,("S", "D"): 0,("S", "H"): -1,("H", "D"): -1,("L", "N"): -3,("W", "A"): -3,("Y", "M"): -1,("G", "R"): -2,("Y", "I"): -1,("Y", "E"): -2,("B", "Y"): -3,("Y", "A"): -2,("V", "D"): -3,("B", "S"): 0,("G", "N"): 0,("E", "C"): -4,("Y", "Q"): -1,("V", "A"): 0,("M", "R"): -1,("V", "E"): -2,("T", "N"): 0,("V", "I"): 3,("V", "S"): -2,("Z", "P"): -1,("V", "M"): 1,("T", "F"): -2,("V", "Q"): -2,("K", "K"): 5,("P", "D"): -1,("I", "H"): -3,("I", "D"): -3,("T", "R"): -1,("P", "L"): -3,("K", "G"): -2,("M", "N"): -2,("P", "H"): -2,("F", "Q"): -3,("Z", "G"): -2,("X", "L"): -1,("T", "M"): -1,("Z", "C"): -3,("X", "H"): -1,("D", "R"): -2,("B", "W"): -4,("X", "D"): -1,("Z", "K"): 1,("F", "A"): -2,("Z", "W"): -3,("F", "E"): -3,("D", "N"): 1,("B", "K"): 0,("F", "I"): 0,("B", "G"): -1,("X", "T"): 0,("F", "M"): 0,("B", "C"): -3,("Z", "I"): -3,("Z", "V"): -2,("L", "Q"): -2,("W", "E"): -3,("Q", "R"): 1,("W", "M"): -1,("Q", "C"): -3,("W", "I"): -3,("S", "C"): -1,("L", "A"): -1,("S", "G"): 0,("L", "E"): -3,("W", "Q"): -2,("H", "G"): -2,("S", "K"): 0,("Q", "N"): 0,("N", "R"): 0,("H", "C"): -3,("Y", "N"): -2,("G", "Q"): -2,("Y", "F"): 3,("C", "A"): 0,("V", "L"): 1,("G", "E"): -2,("G", "A"): 0,("K", "R"): 2,("E", "D"): 2,("Y", "R"): -2,("M", "Q"): 0,("T", "I"): -1,("C", "D"): -3,("V", "F"): -1,("T", "A"): 0,("T", "P"): -1,("B", "P"): -2,("T", "E"): -1,("V", "N"): -3,("P", "G"): -2,("M", "A"): -1,("K", "H"): -1,("V", "R"): -3,("P", "C"): -3,("M", "E"): -2,("K", "L"): -2,("M", "I"): 1,("T", "Q"): -1,("I", "G"): -4,("P", "K"): -1,("K", "D"): -1,("I", "C"): -1,("Z", "D"): 1,("F", "R"): -3,("X", "K"): -1,("Q", "D"): 0,("X", "G"): -1,("Z", "L"): -3,("X", "C"): -2,("Z", "H"): 0,("B", "L"): -4,("B", "H"): 0,("X", "W"): -2,("B", "D"): 4,("D", "A"): -2,("S", "L"): -2,("X", "S"): 0,("F", "N"): -3,("S", "R"): -1,("W", "D"): -4,("V", "Y"): -1,("W", "L"): -2,("H", "R"): 0,("W", "H"): -2,("H", "N"): 1,("W", "T"): -2,("S", "F"): -2,("W", "P"): -4,("L", "D"): -4,("B", "I"): -3,("L", "H"): -3,("S", "N"): 1,("B", "T"): -1,("Y", "K"): -2,("E", "Q"): 2,("Y", "G"): -3,("Z", "S"): 0,("Y", "C"): -2,("G", "D"): -1,("B", "V"): -3,("E", "A"): -1,("Y", "W"): 2,("Y", "S"): -2,("C", "N"): -3,("V", "C"): -1,("T", "H"): -2,("P", "R"): -2,("V", "G"): -3,("T", "L"): -1,("V", "K"): -2,("K", "Q"): 1,("R", "A"): -1,("I", "R"): -3,("T", "D"): -1,("P", "F"): -4,("I", "N"): -3,("K", "I"): -3,("M", "D"): -3,("V", "W"): -3,("M", "H"): -2,("P", "N"): -2,("K", "A"): -1,("M", "L"): 2,("K", "E"): 1,("Z", "E"): 4,("X", "N"): -1,("Z", "A"): -1,("Z", "M"): -1,("X", "F"): -1,("K", "C"): -3,("B", "Q"): 0,("X", "B"): -1,("B", "M"): -3,("F", "C"): -2,("Z", "Q"): 3,("X", "Z"): -1,("F", "G"): -3,("B", "E"): 1,("X", "V"): -1,("F", "K"): -3,("B", "A"): -2,("X", "R"): -1,("W", "G"): -2,("Z", "F"): -3,("S", "Q"): 0,("W", "C"): -2,("W", "K"): -3,("H", "Q"): 0,("L", "C"): -1,("W", "N"): -4,("S", "A"): 1,("L", "G"): -4,("W", "S"): -3,("S", "E"): 0,("H", "E"): 0,("S", "I"): -2,("H", "A"): -2,("S", "M"): -1,("Y", "L"): -1,("Y", "H"): 2,("Y", "D"): -3,("E", "R"): 0,("X", "P"): -2,("G", "C"): -3,("E", "N"): 0,("Y", "T"): -2,("Y", "P"): -3,("T", "K"): -1,("P", "Q"): -1,("T", "C"): -1,("V", "H"): -3,("T", "G"): -2,("I", "Q"): -3,("Z", "T"): -1,("C", "R"): -3,("V", "P"): -2,("P", "E"): -1,("M", "C"): -1,("K", "N"): 0,("P", "A"): -1,("M", "G"): -3,("T", "S"): 1,("I", "E"): -3,("P", "M"): -2,("M", "K"): -1,("I", "A"): -1,("P", "I"): -3,("X", "M"): -1,("L", "I"): 2,("X", "I"): -1,("Z", "B"): 1,("X", "E"): -1,("Z", "N"): 0,("X", "A"): 0,("B", "R"): -1,("B", "N"): 3,("F", "D"): -3,("X", "Y"): -1,("Z", "R"): 0,("F", "H"): -1,("B", "F"): -3,("F", "L"): 0,("X", "Q"): -1,("B", "B"): 4,("F", "W"): 1,("R", "L"): -2,("P", "S"): -1,("T", "V"): 0,("A", "N"): -2,("Y", "Z"): -2,("R", "W"): -3,("A", "Q"): -1,("D", "S"): 0,("H", "H"): 8,("H", "S"): -1,("D", "H"): -1,("N", "L"): -3,("A", "W"): -3,("M", "Y"): -1,("R", "G"): -2,("I", "Y"): -1,("E", "Y"): -2,("Y", "B"): -3,("A", "Y"): -2,("D", "V"): -3,("S", "B"): 0,("Y", "Y"): 7,("N", "G"): 0,("C", "E"): -4,("Q", "Y"): -1,("Z", "Z"): 4,("A", "V"): 0,("C", "C"): 9,("R", "M"): -1,("E", "V"): -2,("N", "T"): 0,("P", "P"): 7,("I", "V"): 3,("S", "V"): -2,("P", "Z"): -1,("M", "V"): 1,("F", "T"): -2,("Q", "V"): -2,("K", "K"): 5,("D", "P"): -1,("H", "I"): -3,("D", "I"): -3,("R", "T"): -1,("L", "P"): -3,("G", "K"): -2,("N", "M"): -2,("H", "P"): -2,("Q", "F"): -3,("G", "Z"): -2,("L", "X"): -1,("M", "T"): -1,("C", "Z"): -3,("H", "X"): -1,("R", "D"): -2,("W", "B"): -4,("D", "X"): -1,("K", "Z"): 1,("A", "F"): -2,("W", "Z"): -3,("E", "F"): -3,("N", "D"): 1,("K", "B"): 0,("X", "X"): -1,("I", "F"): 0,("G", "B"): -1,("T", "X"): 0,("M", "F"): 0,("C", "B"): -3,("I", "Z"): -3,("V", "Z"): -2,("S", "S"): 4,("Q", "L"): -2,("E", "W"): -3,("R", "Q"): 1,("N", "N"): 6,("M", "W"): -1,("C", "Q"): -3,("I", "W"): -3,("C", "S"): -1,("A", "L"): -1,("G", "S"): 0,("E", "L"): -3,("Q", "W"): -2,("G", "H"): -2,("K", "S"): 0,("N", "Q"): 0,("R", "N"): 0,("C", "H"): -3,("N", "Y"): -2,("Q", "G"): -2,("F", "Y"): 3,("A", "C"): 0,("L", "V"): 1,("E", "G"): -2,("A", "G"): 0,("R", "K"): 2,("D", "E"): 2,("R", "Y"): -2,("Q", "M"): 0,("I", "T"): -1,("D", "C"): -3,("F", "V"): -1,("A", "T"): 0,("P", "T"): -1,("P", "B"): -2,("E", "T"): -1,("N", "V"): -3,("G", "P"): -2,("A", "M"): -1,("H", "K"): -1,("R", "V"): -3,("C", "P"): -3,("E", "M"): -2,("L", "K"): -2,("V", "V"): 4,("I", "M"): 1,("Q", "T"): -1,("G", "I"): -4,("K", "P"): -1,("M", "M"): 5,("D", "K"): -1,("C", "I"): -1,("D", "Z"): 1,("R", "F"): -3,("K", "X"): -1,("D", "Q"): 0,("G", "X"): -1,("L", "Z"): -3,("C", "X"): -2,("H", "Z"): 0,("L", "B"): -4,("H", "B"): 0,("F", "F"): 6,("W", "X"): -2,("D", "B"): 4,("A", "D"): -2,("L", "S"): -2,("S", "X"): 0,("N", "F"): -3,("R", "S"): -1,("D", "W"): -4,("Y", "V"): -1,("L", "W"): -2,("R", "H"): 0,("H", "W"): -2,("N", "H"): 1,("T", "W"): -2,("T", "T"): 5,("F", "S"): -2,("P", "W"): -4,("D", "L"): -4,("I", "B"): -3,("H", "L"): -3,("N", "S"): 1,("T", "B"): -1,("L", "L"): 4,("K", "Y"): -2,("Q", "E"): 2,("G", "Y"): -3,("S", "Z"): 0,("C", "Y"): -2,("D", "G"): -1,("V", "B"): -3,("A", "E"): -1,("W", "Y"): 2,("E", "E"): 5,("S", "Y"): -2,("N", "C"): -3,("C", "V"): -1,("H", "T"): -2,("R", "P"): -2,("G", "V"): -3,("L", "T"): -1,("K", "V"): -2,("Q", "K"): 1,("A", "R"): -1,("R", "I"): -3,("D", "T"): -1,("F", "P"): -4,("N", "I"): -3,("I", "K"): -3,("D", "M"): -3,("W", "V"): -3,("W", "W"): 11,("H", "M"): -2,("N", "P"): -2,("A", "K"): -1,("L", "M"): 2,("E", "K"): 1,("E", "Z"): 4,("N", "X"): -1,("A", "Z"): -1,("M", "Z"): -1,("F", "X"): -1,("C", "K"): -3,("Q", "B"): 0,("B", "X"): -1,("M", "B"): -3,("C", "F"): -2,("Q", "Z"): 3,("Z", "X"): -1,("G", "F"): -3,("E", "B"): 1,("V", "X"): -1,("K", "F"): -3,("A", "B"): -2,("R", "X"): -1,("D", "D"): 6,("G", "W"): -2,("F", "Z"): -3,("Q", "S"): 0,("C", "W"): -2,("K", "W"): -3,("Q", "H"): 0,("C", "L"): -1,("N", "W"): -4,("A", "S"): 1,("G", "L"): -4,("S", "W"): -3,("E", "S"): 0,("E", "H"): 0,("I", "S"): -2,("A", "H"): -2,("M", "S"): -1,("L", "Y"): -1,("H", "Y"): 2,("D", "Y"): -3,("R", "E"): 0,("P", "X"): -2,("G", "G"): 6,("C", "G"): -3,("N", "E"): 0,("T", "Y"): -2,("P", "Y"): -3,("K", "T"): -1,("A", "A"): 4,("Q", "P"): -1,("C", "T"): -1,("H", "V"): -3,("G", "T"): -2,("Q", "I"): -3,("T", "Z"): -1,("R", "C"): -3,("P", "V"): -2,("E", "P"): -1,("C", "M"): -1,("N", "K"): 0,("I", "I"): 4,("A", "P"): -1,("G", "M"): -3,("S", "T"): 1,("E", "I"): -3,("M", "P"): -2,("K", "M"): -1,("A", "I"): -1,("I", "P"): -3,("R", "R"): 5,("M", "X"): -1,("I", "L"): 2,("I", "X"): -1,("B", "Z"): 1,("E", "X"): -1,("N", "Z"): 0,("A", "X"): 0,("R", "B"): -1,("N", "B"): 3,("D", "F"): -3,("Y", "X"): -1,("R", "Z"): 0,("H", "F"): -1,("F", "B"): -3,("L", "F"): 0,("Q", "X"): -1,("B", "B"): 4,}

