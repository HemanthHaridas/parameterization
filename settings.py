from training_data import _training_set

# fixed values
_fixed_values = {
    "ct_HOH": -16.6225,

    # oxygen oxygen LJ
    "eps_OO": 4.9193,
    "sigma_OO": 3.2700,

    # water gaussians
    "gaussH_OH": -0.4747,
    "gaussR_OH": 1.0700,
    "gaussW_OH": 0.0759,

    # OH SW parameters
    "swG_OH": 70.000,
    "swS_OH": 154.000
}

# constraints : no idea what these are
_constraints_data = [
    1,  2,   2,   2,
    1,  1,   8,  15,
    15, 15, 304, 304
]

_constraints_data_keys = [
    "h3oho",    "o2h3ts",   "o2h3m",        "Nawc",
    "h5o2m",    "h5o2ts",   "aloh3wOHTS",   "d2ohm",
    "d2ohts2",  "d2ohts",   "gibSE_h2oup",  "gibSE_bothup"
]

# counter : no idea what these are
_counter_data = [
    [8, 9],     [8, 9],   [8, 9],
    [17, 18],   [2, 3],   [5, 6],
    [5, 6],     [12, 13], [19, 20],
    [19, 20]
]

_counter_data_keys = [
    "H3Ow", "h5o2m",            "h5o2ts",
    "Na5w", "Na", 		        "Nawc",
    "Naw",  "aloh4toaloh5scan", "2m1tod3scan",
    "d1scan"
]

# files for charge fitting
_charge_do = ["aloh4na", "d1na", "d3na", "Na5w", "naoh", "Na", "Naw", "w2", "w6", "w"]

# for PBC calculations
_pbc_data = ["NSA", "gib", "boehmite"]
_trajs = [item for item in _training_set if "traj" in item]
_pbc_data_keys = [item for item in _training_set for comp in _pbc_data if item.find(comp) != -1]
_pbc_files = _trajs + _pbc_data
