# Training set
_training_set = [
    "gibSEowCNAlAltraj", 	"gibSEowtraj",	"al6OHtraj",		"al6H2Otraj",
    "al6AlAltraj",		 	"3m1traj",		"d2btod3traj",		"2m1tod2btraj",
    "d2tod1traj",	 		"32wOOtraj",	"32wOHtraj",  		"NaOH30wPTtraj",
    "m1NaOHPTtraj",		 	"32wtraj",		"32wPTtraj", 		"aloh42h2obtraj",
    "m1NaOHtraj",			"2m10traj",		"2m1btraj",			"2m1ctraj",
    "aloh4scanAlOH",		"d1scan",		"aloh4toaloh5scan",	"2m1tod3scan",
    "w2",					"w6",			"d1na",				"d3na",
    "aloh4na",				"w",			"Naw",				"Na5w",
    "Na",			 		"wscan",		"naoh",			"gibbulk",
    "gib001",	 			"gib001top",	"gib001bot",		"gib110",
    "gib825", 				"al3oh122na", 	"al3o3oh6na", 		"al3oh9na",
    "boehmiteb"
]

_products = [
    "gibSEowCNAlAltraj",	"gibSEowtraj",	"al6OHtraj",		"al6H2Otraj",
    "al6AlAltraj",		 	"3m1traj", 		"d2btod3traj",		"2m1tod2btraj",
    "d2tod1traj",		 	"32wOOtraj",	"32wOHtraj",		"NaOH30wPTtraj",
    "m1NaOHPTtraj",			"32wtraj",		"32wPTtraj",		"2m10traj",
    "2m1btraj",				"2m1ctraj",  	"m1NaOHtraj",	 	"aloh42h2obtraj",
    "aloh4scanAlOH",		"wscan",		"d1scan", 			"aloh4toaloh5scan",
    "2m1tod3scan",			"d3na",			"d1na",				"w2",
    "w6",					"Na5w",			"Naw",				"gib001",
    "gib825",				"gib110"
]

# This came from ../data/Eref.dat
_reference_data = [
    53.34110,  -17.5500000000000,   67.59140000,   54.8697,
    39.29550,  137.9060000000000,    4.60300000,    0.7247,
    -4.76830,   -4.8418500000000,  -45.92000000,  -93.9900,
    -24.87000,  -10.6600000000000,  -36.53000000,  -25.1700,
    14.95700,  206.6600000000000,  -41.36000000, -128.3600,
    162.92000, -101.9800000000000,   16.02900000,  135.0100,
    70.98000,  -21.5088000000000,   91.83110000,  -92.7814,
    28.70340,   24.6990079916871,    2.20880933,    6.1508,
    17.69662,  224.6836382000000,  -14.64000000,   -9.7900,
    63.36000,   49.8797000000000,  -36.11500000,  -10.8767,
    49.04000,   66.5780000000000,  217.33000000,  152.3200,
    227.86000,  262.4800000000000,  321.90000000,  -27.4000,
    -86.90000, -204.8128000000000, -224.39676900,   98.4900,
    155.90000,  131.1100000000000,  146.29000000,  171.7000,
    115.50000,  160.5000000000000,  137.50000000,  170.5000
]

_reference_data_keys = [
    "aloh5",         "aloh4w",       "d1",           "d3",
    "d2",            "al2oh6",       "o2h3ts",       "d2ohts",
    "d2ohts2",       "w2",           "w6",           "Na5w",
    "Naw",           "aloh5na",      "d3na",         "d1na",
    "aloh3wOHTS",    "aloh6",        "NaOH3A",       "NaOH10A",
    "aloh3w3",       "aloh6na",      "gibSE_h2oup",  "gibSE_bothup",
    "aloh3+w",       "aloh3wfr4",    "aloh3",        "gib001",
    "gibbulkOH2",    "gib825",       "gib85",        "gib100",
    "gib105",        "gib110",       "aloh3w2",      "aloh3wnaoh",
    "aloh3naoh",     "2m1",          "2m1na",        "d12naw",
    "d2b",           "dtrib",        "al3o3oh6",     "al3oh122",
    "al4o3oh9",      "al6oh18",      "al6oh182",     "OHw",
    "OH4w",          "2u22",         "3p33",         "NSA5",
    "NSA4",          "NSA128",       "NSA200",       "NSA300",
    "NSA1",          "NSAb200",      "NSAb100",      "NSAb500"
]

# pressure reference values
_pressure_data_keys = ["gibbulk", "boehmiteb"]
_pressure_data_values = {
    "gibbulk": [6657, 4512, 4963,  499, 350, 1407],
    "boehmiteb": [110,  105,  112, -672, -10, 2706]
}
