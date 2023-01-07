# Import dependencies

# vehicle class declaration
class OpenVEHICLE:
    # constructor
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.name = ''
        self.factor_grip = 0
        self.factor_drive = 0
        self.factor_aero = 0
        self.driven_wheels = 4
        self.M = 0
        self.v_max = 0
        self.rho = 0
        self.factor_cl = 0
        self.Cl = 0
        self.A = 0
        self.sens_y = 0
        self.mu_y = 0
        self.mu_y_M = 0
        self.sens_x = 0
        self.mu_x = 0
        self.mu_x_M = 0
        self.factor_Cd = 0
        self.Cd = 0
        self.Cr = 0
        self.vehicle_speed = 0
        self.factor_power = 0
        self.fx_engine = 0
        self.beta = 0
        self.rack = 0
        self.tire_radius = 0
        self.engine_torque = 0
        self.engine_power = 0
        self.engine_speed = 0
        self.n_primary = 0
        self.n_gearbox = 0
        self.n_final = 0
        self.n_thermal = 0
        self.fuel_LHV = 0
        self.nog = 0