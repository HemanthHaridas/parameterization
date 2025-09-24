# When I wrote the original code, only me and God knew the logic behind it.
# The logic behind some of the variable declarations was known only to us.
# Now, only God knows the logic behind these variable declarations.
# So if you can understand the logic behind these variable declarations,
# Congratulations! You are either God or me from January 2024.

# Increment the following counter as a warning to the next unfortunate person
# who has to work on the parameterization of aluminium.
# time wasted understanding the logic behind the code : 0 hours

import random
import numpy
import typing
import glob
import lammps

from mpi4py import MPI

# Importing initial configuration values and simulation parameters
from initial_values import _values, _keys, _weight
from initial_values import _margin, _num_walkers
from initial_values import _local_rate, _global_rate
from initial_values import _max_iter, _num_generations

# Importing settings related to constraints, counters, periodic boundary conditions (PBC), and charge dumping
from settings import _fixed_values, _constraints_data, _constraints_data_keys
from settings import _counter_data, _counter_data_keys, _pbc_files
from settings import _charge_do

# Importing training data and metadata
from training_data import _reference_data, _reference_data_keys, _training_set
from training_data import _pressure_data_keys, _pressure_data_values


class Particle:
    """
    Represents a particle (or walker) in a particle swarm optimization (PSO) algorithm.
    Each particle has a position, velocity, and tracks its personal and global best positions.
    """

    def __init__(self, position: list[float], index: int):
        # Initialize particle's position in parameter space
        self.position = position
        # Assign a random initial velocity vector of same dimension
        self.velocity = numpy.random.random(len(position))
        # Personal best position (to be updated during optimization)
        self.pbest = position
        # Global best position (shared across swarm)
        self.gbest = position
        # Initial error is set to infinity (to be minimized)
        self.error = numpy.inf
        # store the index
        self.index = index

    @property
    def coordinate(self):
        # Returns current position and velocity as a tuple
        return tuple(self.position, self.velocity)

    @property
    def current_best(self):
        # Returns personal best position
        return self.pbest

    @property
    def global_best(self):
        # Returns global best position
        return self.gbest

    def update(self, weight: float, local_rate: float, global_rate: float):
        """
        Updates the particle's velocity and position using PSO update rules.
        """
        # Random scaling factors for stochastic influence
        scale_local = random.random()
        scale_global = random.random()

        # Velocity update equation (influenced by inertia, personal best, and global best)
        self.velocity = (
            _weight * self.velocity +
            scale_local * local_rate * (self.pbest - self.position) +
            scale_global * global_rate * (self.gbest - self.position)
        )

        # Position update: move particle according to new velocity
        self.position = self.position + self.velocity

    def create_lammps_param_set(self, constraints: typing.Dict[str, str], counters: typing.Dict[str, str], index: int) -> None:
        """
        Generates LAMMPS input parameters for each system in the training set,
        _num_chunksd on the particle's current position and system-specific metadata.
        """
        for system in _training_set:
            # Map current position to parameter dictionary using predefined keys
            _params = create_params(keys=_keys, values=self.position)

            # Determine if system requires periodic boundary conditions (PBC)
            if any(info in system for info in _pbc_files):
                # Set PBC-specific simulation parameters
                _params.update({
                    "boundaries": "p p p",
                    "coulps": "coul/long",
                    "coulcut": "8.5",
                    "kspace": "ewald 1.0e-5",
                    "periodic": "XYZ"
                })
            else:
                # Set non-periodic simulation parameters
                _params.update({
                    "boundaries": "f f f",
                    "coulps": "coul/cut",
                    "coulcut": "15.0",
                    "kspace": "none",
                    "periodic": "NONE"
                })

            # Check if system is a scan-type simulation
            if "scan" in system:
                _params.update({
                    "spe": "T",
                    "traj": "F",
                    "outdir": "./data/scan",
                    "datadir": "./data/data",
                    "thermostyle": "pe",
                    "scan": "T"
                })

            # Check if system is a trajectory-type simulation
            if "traj" in system:
                _params.update({
                    "spe": "T",
                    "traj": "T",
                    "datadir": "./data/traj",
                    "thermostyle": "pe",
                    "scan": "F",
                    "boundaries": "p p p"  # Override boundaries for trajectory runs
                })

            # Default settings for non-scan and non-traj systems
            if "traj" not in system and "scan" not in system:
                _params.update({
                    "spe": "F",
                    "traj": "F",
                    "datadir": "./data/data",
                    "thermostyle": "pe fnorm",
                    "scan": "F",
                    "supplements": ""
                })

                # If pressure data is required, extend thermostyle and add pressure outputs
                if system in _pressure_data_keys:
                    _params["thermostyle"] = "pe pxx pyy pzz pxy pxz pyz"
                    _params["supplements"] = "$(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz)"

            # Apply constraint if system is listed in constraints
            if system in _constraints_data_keys:
                _params["constraint"] = f"id {constraints[system]}"
            else:
                _params["constraint"] = "empty"

            # Apply counter if system is listed in counters
            if system in _counter_data_keys:
                counter_values = ' '.join(map(str, counters[system]))
                _params["counter"] = f"id {counter_values}"
            else:
                _params["counter"] = "empty"

            # Enable charge dumping if system is listed
            _params["qdump"] = "T" if system in _charge_do else "F"

            # Set filename for output
            _params["filename"] = system

            # append fixed values
            _params.update(_fixed_values)

            # Generate final LAMMPS input file using assembled parameters
            create_final_lammps_input(params=_params, walker_index=index)

    def compute(self, index: int, communicator):
        """
        Searches for LAMMPS input files matching a specific index pattern
        and prints the list of matched filenames.

        Args:
            index (int): The numerical identifier used to filter input files.
                        Typically corresponds to a walker or simulation ID.
        """

        # Collect all LAMMPS input files matching the pattern "*_{index}.inp"
        # This allows for batch execution of multiple simulation setups indexed by a shared identifier.
        _input_files = glob.glob(f"*_{index}.inp")

        # Loop over each matched input file and attempt to run it through the LAMMPS engine.
        for _file in _input_files:
            try:
                # Execute the LAMMPS input script.
                # This will parse and run the simulation defined in the .inp file.
                # Initialize a LAMMPS handler instance.
                # This object interfaces with the LAMMPS simulation engine and allows execution of input scripts.
                _lammps_handler = lammps.lammps(
                    comm=communicator,
                    cmdargs=["-log", f"{_file[:-4]}.log"]
                )
                _lammps_handler.file(_file)

            except lammps.MPIAbortException:
                # Catch and silently ignore MPIAbortException.
                # This exception typically indicates a controlled termination from within LAMMPS,
                # such as due to a simulation error or convergence failure.
                # Suppressing it allows the loop to continue with remaining input files.
                pass

            except Exception as _exception:
                # Catch any other unexpected exceptions and print the error message.
                # This helps with debugging malformed input files or runtime issues
                # without halting the entire batch execution.
                print(_exception)

            finally:
                if _lammps_handler is not None:
                    _lammps_handler.close()

    def evaluate(self, index: int):
        ener_dat = {}
        with open(f"ener_{index}.dat") as enerObject:
            for line in enerObject:
                line = line.split()
                ener_dat[line[0].strip(":")] = [float(x) for x in line[1:]]

        fmax_values = [ener_dat[key][1] for key in ener_dat]

        # Create containers to hold error information
        error_charge = {}
        error_energy = {}
        error_force = {}
        error_pressure = {}
        for system in _training_set:
            # calculate differences in charges
            # if system not in charge_exclusion:
            if system in _charge_do:
                esp_file = f"./data/esp/{system}.esp"
                charge_file = f"{system}_{index}.q"
                with open(esp_file) as espObject, open(charge_file) as chargeObject:
                    esp_data = [float(data.split()[-1]) for data in espObject.readlines()[2:]]
                    charge_data = [float(data.split()[-1]) for data in chargeObject.readlines()[9:]]
                    error_q_file = [(esp_value - charge_value) for (esp_value, charge_value) in zip(esp_data, charge_data)]
                    error_charge[system] = [x for x in error_q_file]

            # First process the scans
            # calculate differences in energies
            if system.find("scan") != -1:
                energy_file = f"./data/scan/{system}.ener"
                log_file = f"./{system}_{index}.log"
                with open(energy_file) as energyObject, open(log_file) as logObject:
                    energy_ref = [float(data.split()[-1]) for data in energyObject.readlines()]
                    nlines = len(energy_ref) + 5
                    _ff_energy = [data for data in logObject.readlines()[-1 * nlines:]][1:-4]
                    ff_energy = numpy.array([float(data.split()[0]) for data in _ff_energy])
                    ff_energy_s = ff_energy - min(ff_energy)
                    error_e_file = [abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_energy_s, energy_ref)]
                    error_energy[system] = numpy.mean(numpy.array(error_e_file))

            # calculate differences in forces
            if system.find("scan") != -1:
                force_file = f"./data/scan/{system}.frc"
                ff_file = f"./{system}_{index}.FF.frc"
                with open(force_file) as forceObject, open(ff_file) as ffObject:
                    _ref_data = [data.split()[1:] for data in forceObject.readlines() if len(data.split()) == 4]
                    ref_data = [float(value) for force in _ref_data for value in force]
                    _ff_force = [data.split()[1:] for data in ffObject.readlines()
                                 if len(data.split()) == 4 and data.find("ITEM") == -1]
                    ff_force = [float(value) for force in _ff_force for value in force]
                    error_f_file = [(ff_value - ref_value)**2 for (ff_value, ref_value) in zip(ff_force, ref_data)]
                    error_f_file = numpy.array(error_f_file)
                    error_f_file = numpy.sqrt(error_f_file.reshape(-1, 3))  # reshapes the array in x, y, z
                    error_force[system] = numpy.mean((numpy.sum(error_f_file, axis=1)))  # Takes the sum along the rows

            # Now process trajectories
            if system.find("traj") != -1:
                energy_file = f"./data/traj/{system}.ener"
                log_file = f"./{system}_{index}.log"
                with open(energy_file) as energyObject, open(log_file) as logObject:
                    # print(log_file)
                    energy_ref = numpy.array([float(data.split()[-1]) for data in energyObject.readlines()])
                    energy_ref_s = energy_ref - numpy.mean(energy_ref)
                    nlines = len(energy_ref) + 5
                    _ff_energy = [data for data in logObject.readlines()[-1 * nlines:]][1:-4]
                    ff_energy = numpy.array([float(data.split()[0]) for data in _ff_energy])
                    ff_energy_s = ff_energy - numpy.mean(ff_energy)
                    error_e_file = [abs(ff_value - ref_value) for (ff_value, ref_value) in zip(ff_energy_s, energy_ref_s)]
                    error_energy[system] = numpy.mean(numpy.array(error_e_file))

                # Now calculate forces
                force_file = f"./data/traj/{system}.frc"
                ff_file = f"./{system}_{index}.FF.frc"
                with open(force_file) as forceObject, open(ff_file) as ffObject:
                    _ref_data = [data.split() for data in forceObject.readlines() if len(data.split()) == 3]
                    ref_data = [float(value) for force in _ref_data for value in force]
                    ref_force_matrix = numpy.array(ref_data).reshape(-1, 3)
                    _ff_force = [data.split()[1:] for data in ffObject.readlines()
                                 if len(data.split()) == 4 and data.find("ITEM") == -1]
                    ff_force = [float(value) for force in _ff_force for value in force]
                    ff_force_matrix = numpy.array(ff_force).reshape(-1, 3)
                    error_f_file = numpy.sum(numpy.power(ff_force_matrix - ref_force_matrix, 2), axis=1)
                    error_force[system] = numpy.mean(numpy.sqrt(error_f_file))

            if system.find("32wtraj") != -1:
                water_file = f"./data/traj/{system}.frc"
                ff_file = f"./{system}_{index}.FF.frc"
                with open(water_file) as waterObject, open(ff_file) as ffObject:
                    _ref_data = [data.split() for data in waterObject.readlines()]
                    ref_data = [float(value) for force in _ref_data for value in force]
                    _ff_force = [data.split()[1:] for data in ffObject.readlines()
                                 if len(data.split()) == 4 and data.find("ITEM") == -1]
                    ff_force = [float(value) for force in _ff_force for value in force]
                    error_f_file = [(ff_value - ref_value)**2 for (ff_value, ref_value) in zip(ff_force, ref_data)]
                    error_f_file = numpy.array(error_f_file)
                    error_f_file = numpy.sqrt(error_f_file.reshape(-1, 3))  # reshapes the array in x, y, z
                    error_force[system] = numpy.mean((numpy.sum(error_f_file, axis=1)))  # Takes the sum along the rows

            if system.find("traj") == -1 and system.find("scan") == -1:
                if system in _pressure_data_keys:
                    ff_data = ener_dat[system][2:]
                    ref_data = _pressure_data_values[system]
                    error_p_file = [abs(ref_value - ff_value) for (ff_value, ref_value) in zip(ff_data, ref_data)]
                    error_pressure[system] = 0
                    for value in error_p_file:
                        error_pressure[system] += value
                    error_pressure[system] = error_pressure[system]/6

        # Now take care of reactions
        reaction_energies = [
            10.0 * error_energy["gibSEowCNAlAltraj"],
            10.0 * error_energy["gibSEowtraj"],
            10.0 * error_energy["al6OHtraj"],
            10.0 * error_energy["al6H2Otraj"],
            10.0 * error_energy["al6AlAltraj"],
            10.0 * error_energy["3m1traj"],
            10.0 * error_energy["d2btod3traj"],
            10.0 * error_energy["2m1tod2btraj"],
            15.0 * error_energy["d2tod1traj"],
            50.0 * error_energy["32wOOtraj"],
            25.0 * error_energy["32wOHtraj"],
            5.0 * error_energy["NaOH30wPTtraj"],
            5.0 * error_energy["m1NaOHPTtraj"],
            5.0 * error_energy["32wtraj"],
            10.0 * error_energy["32wPTtraj"],
            10.0 * error_energy["2m10traj"],
            10.0 * error_energy["2m1btraj"],
            10.0 * error_energy["2m1ctraj"],
            300./133 * error_energy["m1NaOHtraj"],
            4.0 * error_energy["aloh42h2obtraj"],
            10.0 * error_energy["aloh4scanAlOH"],
            20.0 * error_energy["wscan"],
            10.0 * error_energy["d1scan"],
            2.0 * error_energy["aloh4toaloh5scan"],
            2.0 * error_energy["2m1tod3scan"],
            1.00 * (1 * (ener_dat["d3na"][0] - (2 * ener_dat["aloh4na"][0])) - (_energy_reference["d3na"])),
            1.00 * (1 * (ener_dat["d1na"][0] + ener_dat["w"][0] - (2 * ener_dat["aloh4na"][0])) - (_energy_reference["d1na"])),
            5.00 * (1 * (ener_dat["w2"][0] - (2 * ener_dat["w"][0])) - (_energy_reference["w2"])),
            5.00 * (1 * (ener_dat["w6"][0] - (6 * ener_dat["w"][0])) - (_energy_reference["w6"])),
            0.25 * (1 * (ener_dat["Na5w"][0] - (5 * ener_dat["w"][0] + ener_dat["Na"][0])) - (_energy_reference["Na5w"])),
            1.00 * (1 * (ener_dat["Naw"][0] - (ener_dat["w"][0] + ener_dat["Na"][0])) - (_energy_reference["Naw"])),
            0.50 * (1 * (ener_dat["gib001"][0] - (ener_dat["gib001top"][0] +
                    ener_dat["gib001bot"][0])) - (_energy_reference["gib001"])),
            0.50 * (1 * (ener_dat["gib825"][0] - (ener_dat["gibbulk"][0])) - (_energy_reference["gib825"])),
            0.25 * (1 * (ener_dat["gib110"][0] - (ener_dat["gibbulk"][0])) - (_energy_reference["gib110"]))
        ]

        # Now compute error terms
        fmax_error = numpy.mean(numpy.array([value**2 for value in fmax_values]))
        charge_error = numpy.sqrt(numpy.mean(numpy.array(
            [charge**2 for (key, value) in error_charge.items() for charge in value])))
        reactions_error = numpy.sqrt(numpy.mean(numpy.array([energy**2 for energy in reaction_energies])))
        force_error = numpy.mean(numpy.array([value**2 for (key, value) in error_force.items()]))
        pressure_error = numpy.mean(numpy.array([value**2 for (key, value) in error_pressure.items()]))

        # Now we need to compute the final error
        charge_w = 300.
        reactions_w = 1.
        force_w = 0.05
        fmax_w = 0.005
        pressure_w = 2.5e-6

        errors = [charge_error, reactions_error, force_error, fmax_error, pressure_error]
        weights = [charge_w, reactions_w, force_w, fmax_w, pressure_w]
        for i in range(len(errors)):
            errors[i] *= weights[i]

        _final_error = 100 * sum(errors) / sum(weights)

        if _final_error < self.error:
            self.error = _final_error
            self.pbest = self.position
            # with open(f"error_{index}", "w") as test:
            #     test.write(f"{_final_error}\n")

# Function to create a dictionary of parameters from keys and values


def create_params(keys: list[str],
                  values: typing.Sequence[typing.Union[str, float]]
                  ) -> typing.Dict[str, str]:
    """
    Constructs a dictionary mapping each key to its corresponding value.

    Args:
        keys (list[str]): List of parameter names.
        values (Sequence[Union[str, float]]): List of parameter values.

    Returns:
        dict[str, str]: Dictionary of parameter key-value pairs.
    """
    return {key: value for key, value in zip(keys, values)}


# Function to initialize a swarm of walkers (particles) for optimization

def create_walkers(values: list[float], margin: float, num_walkers: int, gbest_pos: typing.Optional[list[float]] = None):
    """
    Initializes a list of Particle instances with randomized positions
    within a margin around the reference values.

    Args:
        values (list[float]): Reference parameter values.
        margin (float): Fractional margin to define search space bounds.
        num_walkers (int): Number of particles to initialize.

    Returns:
        list[Particle]: List of Particle instances with randomized positions.
    """
    # Define lower and upper bounds for each parameter
    _min_values = values - (margin * numpy.abs(values))
    _max_values = values + (margin * numpy.abs(values))

    # Generate random parameter sets for each walker within bounds
    # Shape: (num_parameters, num_walkers)
    _params = numpy.random.uniform(
        _min_values[:, None], _max_values[:, None], size=(
            _max_values.shape[0], num_walkers))

    # Instantiate Particle objects using column-wise slices of _params
    walkers = [
        Particle(position=_params[:, i], index=i) for i in range(num_walkers)
    ]

    if gbest_pos is not None:
        for walker in walkers:
            walker.gbest = gbest_pos

    return walkers

# create lammps input files


def create_final_lammps_input(params: typing.Dict[str, str], walker_index: int) -> None:
    _file = params["filename"]  # get file name

    # write corresponding input file
    with open(f"{_file}_{walker_index}.inp", "w") as file_object:
        # header section
        file_object.write("units        real\n")
        file_object.write("boundary     {}\n".format(params["boundaries"]))
        file_object.write("atom_style   full\n")
        file_object.write(
            "read_data    {}/{}-q0.data\n".format(params["datadir"], params["filename"]))
        file_object.write("\n")

        # charge section
        file_object.write("set type 1 charge  1.296\n")
        file_object.write("set type 2 charge -0.898\n")
        file_object.write("set type 3 charge  0.449\n")
        file_object.write("set type 4 charge  1.000\n")
        file_object.write("\n")

        # pair_style section
        file_object.write("pair_style hybrid/overlay {} {} sw lj/smooth/linear 5.0 gauss/cut 10.0 coord/gauss/cut 10.0 \n".format(
            params["coulps"], params["coulcut"]))
        file_object.write("pair_coeff * * {}\n".format(params["coulps"]))
        file_object.write("pair_coeff * * {}\n".format(params["coulps"]))
        file_object.write("pair_coeff * * lj/smooth/linear 0 0 0\n")
        file_object.write(f"pair_coeff * * sw param_{walker_index}.sw Al O H NULL\n")
        file_object.write("pair_coeff * * gauss/cut 0 0 1 0\n")
        file_object.write("\n")

        # LJ section
        file_object.write(
            "pair_coeff 1 2 lj/smooth/linear 	{:10.6f}e-2 {:10.6f}\n".format(params["eps_AlO"], 	params["sigma_AlO"]))
        file_object.write(
            "pair_coeff 2 2 lj/smooth/linear 	{:10.6f}e-2 {:10.6f}\n".format(params["eps_OO"], 	params["sigma_OO"]))
        file_object.write(
            "pair_coeff 2 3 lj/smooth/linear 	{:10.6f}e-2 {:10.6f}\n".format(params["eps_OH"], 	params["sigma_OH"]))
        file_object.write(
            "pair_coeff 2 4 lj/smooth/linear 	{:10.6f}e-2 {:10.6f}\n".format(params["eps_NaO"], 	params["sigma_NaO"]))
        file_object.write("\n")

        # gaussian section
        file_object.write("pair_coeff 1 1 coord/gauss/cut 	{:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} \n".format(
            params["gaussH_Al"], params["gaussR_Al"], params["gaussW_Al"], params["coord_Al"], params["radius_Al"]))
        file_object.write("pair_coeff 1 2 coord/gauss/cut 	{:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} \n".format(
            params["gaussH_AlO"], params["gaussR_AlO"], params["gaussW_AlO"], params["coord_AlO"], params["radius_AlO"]))
        file_object.write("pair_coeff 2 3 gauss/cut 		{:10.6f} {:10.6f} {:10.6f} {:10.6f}\n".format(
            params["gaussH_OH"], params["gaussR_OH"], params["gaussW_OH"], 2.0))
        file_object.write("\n")

        # kspace section
        file_object.write("kspace_style {}\n".format(params["kspace"]))
        file_object.write("\n")

        # thermostat
        file_object.write("thermo_style custom {}\n".format(params["thermostyle"]))
        file_object.write("thermo 1\n")
        file_object.write("\n")

        # charge section
        file_object.write(f"fix 1 all qeq/point 1 8.5 1.0e-6 200 param_{walker_index}.qeq\n")
        file_object.write("\n")

        # groups section
        file_object.write("group counter {}\n".format(params["counter"]))
        file_object.write("group todump subtract all counter\n")
        file_object.write("\n")

        # check for conditional arguments
        if params["qdump"] == "T":
            file_object.write("compute 1 all property/atom q\n")
            file_object.write("dump 1 todump custom 1 {}_{}.q id c_1\n".format(params["filename"], walker_index))
            file_object.write("dump_modify 1 sort id\n")
            file_object.write("\n")

        if params["traj"] == "T":
            file_object.write("dump 3 todump custom 1 {}_{}.FF.frc id fx fy fz\n".format(params["filename"], walker_index))
            file_object.write("dump_modify 3 sort id\n")
            file_object.write(
                "rerun {}/{}.dump dump x y z box yes\n".format(params["datadir"], params["filename"]))
            file_object.write("\n")
        elif (params["scan"] == "T"):
            file_object.write("dump 3 todump custom 1 {}_{}.FF.frc id fx fy fz\n".format(params["filename"], walker_index))
            file_object.write("dump_modify 3 sort id\n")
            file_object.write(
                "rerun {}/{}.xyz dump x y z box no format xyz\n".format(params["outdir"], params["filename"]))
            file_object.write("\n")
        else:
            file_object.write("run 0\n")
            file_object.write("print \"{}: $(pe) $(fnorm) {}\" append ener_{}.dat screen no\n".format(
                params["filename"], params["supplements"], walker_index))
            file_object.write("\n")

        # write qeq parameter file
        with open(f"param_{walker_index}.qeq", "w") as param_object:
            param_object.write(
                "1 {} {} {}e-2 0 0.0\n".format(params["chi_Al"], 	params["eta_Al"], 	params["gamma_Al"]))
            param_object.write(
                "2 {} {} {}e-2 0 0.0\n".format(params["chi_O"], 	params["eta_O"], 	params["gamma_O"]))
            param_object.write(
                "3 {} {} {}e-2 0 0.0\n".format(params["chi_H"], 	params["eta_H"],	params["gamma_H"]))
            param_object.write(
                "4 {} {} {}e-2 0 0.0\n".format(params["chi_Na"], 	params["eta_Na"], 	params["gamma_Na"]))

        # write stillinger-weber parameter files
        with open(f"param_{walker_index}.sw", "w") as sw_object:
            sw_object.write("#j  i   k  eps_ijk   sigma_ij  a_ij  lambda_ijk  gamma_ij0.7617  cthet      A  B  p  q  tol\n")
            sw_object.write("O   H   H  {}    {}e-2   1     1           {}e-2       {}e-2  0  1  0  0  0.01\n".format(
                params["eps_HOH"],   params["swS_OH"],  params["swG_OH"],  params["ct_HOH"]))
            sw_object.write("H   O   O  0     {}e-2   1     0           {}e-2       0      0  1  0  0  0.01\n".format(
                params["swS_OH"],   params["swG_OH"]))
            sw_object.write("O   Al  H  {}  	 {}e-2   1     1           {}e-2       {}e-2  0  1  0  0  0.01\n".format(
                params["eps_AlOH"], params["swS_AlO"], params["swG_AlO"], params["ct_AlOH"]))
            sw_object.write("Al  O   O  0     {}e-2   1     1           {}e-2       0      0  1  0  0  0.01\n".format(
                params["swS_AlO"],  params["swG_AlO"]))
            sw_object.write("O   Al  Al 0     {}e-2   1     1           {}e-2       0      0  1  0  0  0.01\n".format(
                params["swS_AlO"],  params["swG_AlO"]))
            sw_object.write("Al  Al  Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  Al  H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("Al  H   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   O   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("O   H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   O   H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   Al  Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   Al  H  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   Al  O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   O   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   H   Al 0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   H   O  0     0       0     0           0           0      0  0  0  0  0.01\n")
            sw_object.write("H   H   H  0     0       0     0           0           0      0  0  0  0  0.01\n")


def walker_evaluate(errors: typing.List[float]) -> (int, float):
    return numpy.argmin(errors), errors[numpy.argmin(errors)]


def create_chunks(walkers: typing.List[Particle], nprocs: int, nwalkers: int) -> typing.List[typing.List[Particle]]:
    _num_chunks = nwalkers // nprocs  # get the number of chunks
    _rem_chunks = nwalkers % nprocs  # get the remainder

    # Compute start and end indices for each chunk
    indices = [
        (sum(_num_chunks + (1 if j < _rem_chunks else 0) for j in range(i)),
         sum(_num_chunks + (1 if j < _rem_chunks else 0) for j in range(i + 1)))
        for i in range(nprocs)
    ]

    # return the chunks
    return [walkers[start:end] for start, end in indices]


# create dictionaries of data
_energy_reference = create_params(keys=_reference_data_keys, values=_reference_data)
_constraints = create_params(keys=_constraints_data_keys, values=_constraints_data)
_counters = create_params(keys=_counter_data_keys, values=_counter_data)

# MPI init section
_comm = MPI.COMM_WORLD
_size = _comm.Get_size()
_rank = _comm.Get_rank()
_root = (_rank == 0)

# gather list of walkers generated on root
if _root:
    walkers = create_walkers(values=numpy.array(_values), margin=_margin, num_walkers=_num_walkers)
    walker_chunks = create_chunks(walkers=walkers, nprocs=_size, nwalkers=_num_walkers)
    num_chunks = len(walker_chunks)
else:
    walkers = None
    walker_chunks = None
    num_chunks = None

# Scatter walkers to all ranks
_this_ranks_walkers = _comm.scatter(walker_chunks, root=0)
_global_num_chunks = _comm.bcast(num_chunks, root=0)

# split into nchunks subcommunicators
_split = _comm.Split(_rank % _global_num_chunks, key=_rank)

# book keeping
_global_best_error = numpy.inf
_global_best_position = None

# Outer optimization loop
for _generation in range(1, _num_generations):
    _iter = 1

    # need to compute the first step separately
    # evaluate walkers in parallel
    for _walker in _this_ranks_walkers:
        _walker.create_lammps_param_set(constraints=_constraints, counters=_counters, index=_walker.index)
        _walker.gbest = _global_best_position if _global_best_position is not None else _walker.position
        _walker.compute(index=_walker.index, communicator=_split)
        _walker.evaluate(index=_walker.index)

    # now figure out which walker was the closest to minima
    # Gather updated errors
    _local_error = [_walker.error for _walker in _this_ranks_walkers]
    _gathered_errors = _comm.gather(_local_error, root=0)

    if _root:
        _all_errors = numpy.concatenate(_gathered_errors).tolist()
        _best_walker, _best_error = walker_evaluate(errors=_all_errors)
        if _best_error < _global_best_error:
            _global_best_error = _best_error
            _global_best_position = walkers[_best_walker].position
    else:
        _best_walker, _best_error = None, None

    # Broadcast best walker index and error to all ranks
    _best_walker = _comm.bcast(_best_walker, root=0)
    _best_error = _comm.bcast(_best_error, root=0)

    # Broadcast the global best position and error to all ranks
    _global_best_error = _comm.bcast(_global_best_error, root=0)
    _global_best_position = _comm.bcast(_global_best_position, root=0)

    _current_best_error = _best_error

    while (_current_best_error >= _best_error) and (_iter <= _max_iter):
        _iter += 1
        _current_best_error = min(_best_error, _current_best_error)

        # Broadcast best position to all ranks
        if _root:
            gbest_position = _global_best_position
        else:
            gbest_position = None

        gbest_position = _comm.bcast(gbest_position, root=0)

        # Each rank updates its walker
        for _walker in _this_ranks_walkers:
            _walker.gbest = gbest_position
            _walker.update(weight=_weight, local_rate=_local_rate, global_rate=_global_rate)
            _walker.create_lammps_param_set(constraints=_constraints, counters=_counters, index=_walker.index)
            _walker.compute(index=_walker.index, communicator=_split)
            _walker.evaluate(index=_walker.index)

        # Gather updated errors
        _local_error = [_walker.error for _walker in _this_ranks_walkers]
        _gathered_errors = _comm.gather(_local_error, root=0)

        if _root:
            _all_errors = numpy.concatenate(_gathered_errors).tolist()
            _best_walker, _best_error = walker_evaluate(errors=_all_errors)

            # Take care of the errors
            if _best_error < _global_best_error:
                _global_best_error = _best_error
                _global_best_position = walkers[_best_walker].position

            # Write the current best
            with open("pso-current-best.log", "w") as poslogger:
                numpy.savetxt(poslogger, _global_best_position, fmt="%10.6f", delimiter=" ")
                poslogger.write("\n")
        else:
            _best_walker, _best_error = None, None

        # Broadcast best walker index and error to all ranks
        _best_walker = _comm.bcast(_best_walker, root=0)
        _best_error = _comm.bcast(_best_error, root=0)

        # Broadcast the global best position and error to all ranks
        _global_best_error = _comm.bcast(_global_best_error, root=0)
        _global_best_position = _comm.bcast(_global_best_position, root=0)

        if _root:
            with open("pso.log", "a") as logger:
                logger.write("Generation: {:10.0f} Best Walker: {:10.0f} Best Error: {:10.3f} Current Error: {:10.3f}\n".format(
                    _generation, _best_walker, _best_error, _current_best_error))

    # Save best position
    if _root:
        with open("pso-best.log", "a") as poslogger:
            numpy.savetxt(poslogger, walkers[_best_walker].position, fmt="%10.6f", delimiter=" ")
            poslogger.write("\n")

    # Regenerate walkers from best position
    # also keep the best walker from the previous step
    if _root:
        walkers = create_walkers(values=numpy.array(gbest_position), margin=_margin,
                                 num_walkers=_num_walkers, gbest_pos=gbest_position)
        walker_chunks = create_chunks(walkers=walkers, nprocs=_size, nwalkers=len(walkers))
    else:
        walkers = None

    # Scatter new walkers
    _this_ranks_walkers = _comm.scatter(walker_chunks, root=0)
