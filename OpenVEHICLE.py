# Import dependencies
import pandas as pd
import os
import datetime
import math
from tabulate import tabulate
import numpy as np
from scipy import interpolate

# vehicle class declaration
class OpenVEHICLE:
    # constructor
    def __init__(self, filename = 'Formula 1.xlsx'):
        self.filename = filename
        self.name = 'Formula 1'
        self.type = 'Open Wheel'
        self.M = 650 # kg
        self.df = 0.45 # %
        self.L = 3 # m
        self.rack = 10 # steering rack ratio
        self.Cl = -4.8 # lift coefficient
        self.Cd = -1.2 # drag coefficient
        self.factor_Cl = 1.0 # lift coefficient scale multiplier
        self.factor_Cd = 1.0 # drag coefficient scale multiplier
        self.da = 0.5 # %
        self.A = 1 # m^2
        self.rho = 1.225 # kg/m^3
        self.br_disc_d = 250 # mm
        self.br_pad_h = 40 # mm
        self.br_pad_mu = 0.45 # friction coefficient brake pads
        self.br_nop = 6 # number of pistons in caliper
        self.br_pist_d = 40 # mm
        self.br_mast_d = 25 # mm
        self.br_ped_r = 4 # pedal ratio
        self.factor_grip = 1
        self.tire_radius = 0.33 # mm
        self.Cr = -0.001 # rolling resistance coefficient
        self.mu_x = 2.0 # longitudinal friction coefficient
        self.mu_x_M = 250 # kg
        self.sens_x = 0.0001 # longitudinal friction sensitivity
        self.mu_y = 2.0 # lateral friction coefficient
        self.mu_y_M = 250 # kg
        self.sens_y = 0.0001 # lateral friction sensitivity
        self.CF = 800 # N/deg
        self.CR = 1000 # N/deg
        self.factor_power = 1 
        self.n_thermal = 0.35
        self.fuel_LHV = 47200000 # J/kg
        self.drive = 'RWD'
        self.shift_time = 0.01 # s
        self.n_primary = 1
        self.n_final = 0.92
        self.n_gearbox = 0.98
        self.ratio_primary = 1
        self.ratio_final = 7
        self.ratio_gearbox = [2.57, 2.11, 1.75, 1.46, 1.29, 1.13, 1]
        self.nog = len(self.ratio_gearbox)
        self.torque_curve = [125, 125, 125, 125, 125, 125, 150, 200, 240, 270, 300, 340, 350, 340, 330, 325, 312, 296.75]
        fill_in(self)

def fill_in(self):
    ## HUD

    if not os.path.exists('OpenVEHICLE Vehicles'):
        os.makedirs('OpenVEHICLE Vehicles')
    vehname = 'OpenVEHICLE Vehicles/OpenVEHICLE_' + self.name + '_' + self.type
    if os.path.exists(vehname+'.log'):
        os.remove(vehname+'.log')
    print('_______                    ___    ________________  ________________________________')
    print('__  __ \_____________________ |  / /__  ____/__  / / /___  _/_  ____/__  /___  ____/')
    print('_  / / /__  __ \  _ \_  __ \_ | / /__  __/  __  /_/ / __  / _  /    __  / __  __/   ')
    print('/ /_/ /__  /_/ /  __/  / / /_ |/ / _  /___  _  __  / __/ /  / /___  _  /___  /___   ')
    print('\____/ _  .___/\___//_/ /_/_____/  /_____/  /_/ /_/  /___/  \____/  /_____/_____/   ')
    print('       /_/                                                                          ')
    print('====================================================================================')
    print(self.filename)
    print('File read sucessfuly')
    print('====================================================================================')
    print('Name: ' + self.name)
    print('Type: ' + self.type)
    print('Date: ' + str(datetime.datetime.now()))
    print('====================================================================================')
    print('Vehicle generation started.')

    ## Brake Model

    self.br_pist_a = self.br_nop*np.pi*(self.br_pist_d/1000)**2/4 # [m^2]
    self.br_mast_a = np.pi*(self.br_mast_d/1000)**2/4 # [m^2]
    self.beta = self.tire_radius/(self.br_disc_d/2-self.br_pad_h/2)/self.br_pist_a/self.br_pad_mu/4 # [Pa/N] per wheel
    self.phi = self.br_mast_a/self.br_ped_r*2 # [-] for both systems
    # HUD
    print('Braking model generated successfully.')

    ## Steering Model

    a = round((1-self.df)*self.L, 4) # distance of front axle from center of mass [mm]
    b = -1*self.df*self.L # distance of rear axle from center of mass [mm]
    C = [[2*self.CF, 2*(self.CF+self.CR)],[int(2*self.CF*a), int(2*(self.CF*a+self.CR*b))]] # steering model matrix
    # HUD
    print('Steering model generated successfully.')

    ## Driveline Model

    # getching engine curves
    self.en_speed_curve = list(np.array([i * 1000 for i in range(1, len(self.torque_curve)+1)]))
    self.en_torque_curve = self.torque_curve
    self.en_power_curve = [num1 * num2 * 2 * np.pi / 60 for num1 in self.en_speed_curve for num2 in self.en_torque_curve]
    # memory preaalocation
    # wheel speed per gear for every engine speed value
    self.wheel_speed_gear = np.zeros((self.nog, len(self.en_speed_curve))).tolist()
    # vehicle speed per gear for every engine speed value
    self.vehicle_speed_gear = np.zeros((self.nog, len(self.en_speed_curve))).tolist()
    # wheel torque per gear for every engine speed value
    self.wheel_torque_gear = np.zeros((self.nog, len(self.en_torque_curve))).tolist()
    # calculating values for each gear and engine speed
    for i in range(0, self.nog):
        self.wheel_speed_gear[i] = [n/self.ratio_primary/self.ratio_gearbox[i]/self.ratio_final for n in self.en_speed_curve]
        self.vehicle_speed_gear[i] = [w * 2 * np.pi / 60 * self.tire_radius for w in self.wheel_speed_gear[i]]
        self.wheel_torque_gear[i] = [n * self.ratio_gearbox[i] * self.ratio_final * self.ratio_primary * self.n_primary * self.n_final * self.n_gearbox for n in self.en_torque_curve]
    # minimum and maximum vehicle speeds
    self.v_min = np.amin(self.vehicle_speed_gear).item()
    self.v_max = np.amax(self.vehicle_speed_gear).item()
    # new speed vector for fine meshing
    dv = 0.5/3.6
    self.vehicle_speed = np.linspace(self.v_min, self.v_max, math.floor((self.v_max-self.v_min)/dv)).tolist()
    # memory preallocation
    # gear
    gear = np.zeros((len(self.vehicle_speed))).tolist()
    # engine tractive force
    fx_engine = np.zeros(len(self.vehicle_speed)).tolist()
    # engine tractive force per gear
    fx = np.zeros((len(self.vehicle_speed), self.nog)).tolist()
    # optimizing gear selection and calculating tractive force
    for i in range(0, len(self.vehicle_speed)):
        # going through the gears
        for j in range(0, self.nog):
            x = self.vehicle_speed_gear[:][j]
            y = [wtg/self.tire_radius for wtg in self.wheel_torque_gear[:][j]]
            xq = self.vehicle_speed[i]
            f = interpolate.interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
            fx[i][j] = f(self.vehicle_speed[i]).tolist()
        # getting maximum tractive force and gear
        fx_engine[i] = max(fx[i])
        gear[i] = fx[i].index(fx_engine[i]) + 1
    # adding values for 0 speed to vectors for interpolation purposes at low speeds
    self.vehicle_speed = np.insert(self.vehicle_speed, 0, 0).tolist()
    gear.insert(0, gear[0])
    fx_engine.insert(0, fx_engine[0])
    # final vectors
    # engine speed
    ratios = [self.ratio_gearbox[g-1] for g in gear]
    ratio_mult = [self.ratio_final * self.ratio_primary * r for r in ratios]
    tire_speed = [vs / self.tire_radius for vs in self.vehicle_speed]
    engine_speed = [ratio_mult[i] * tire_speed[i] * 60 / 2 / np.pi for i in range(0, len(gear))]
    # wheel torque
    wheel_torque = [e * self.tire_radius for e in fx_engine]
    # engine torque
    engine_torque = [wheel_torque[i] / self.ratio_final / ratios[i] /self.ratio_primary/self.n_primary/self.n_gearbox/self.n_final for i in range(0, len(gear))]
    # engine power
    engine_power = [t * s * 2 * np.pi / 60 for t in engine_torque for s in engine_speed]
    # HUD
    print('Driveline model generated successfully.')

    ## Shifting points and Rev Drops

    # finding gear changes
    gear_change = np.diff(gear).tolist()
    # getting speed right before and after gear change
    indices = [ind for ind, ele in enumerate(gear_change) if ele == 1]
    for i in indices:
        gear_change[i+1] = 1
    gear_change.append(0)
    # getting engine speed at gear change
    engine_speed_gear_change = [engine_speed[i] for i,c in enumerate(gear_change) if c == 1]
    # getting shift points
    shift_points = engine_speed_gear_change[::2]
    # getting arrive points
    arrive_points = engine_speed_gear_change[1::2]
    # calculating revdrops
    rev_drops = [shift_points[i] - arrive_points[i] for i in range(0, len(shift_points))]
    # creating shifting table
    rownames = [''] * (self.nog-1)
    for i in range(0, self.nog-1):
        rownames[i] = str(i+1) + '-' + str(i+2)
    shifting = pd.DataFrame({'shift_points': shift_points, 'arrive_points': arrive_points, 'rev_drops': rev_drops}, index=rownames)
    # HUD
    print('Shift points calculated successfully.')

    ## Force model

    # gravitational constant
    g = 9.81 # [m/s^2]
    # drive to aero factors
    if self.drive == 'RWD':
        self.factor_drive = (1-self.df)
        self.factor_aero = (1-self.da)
        self.driven_wheels = 2
    elif self.drive == 'FWD':
        self.factor_drive = self.df
        self.factor_aero = self.da
        self.driven_wheels = 2
    else:
        self.factor_drive = 1
        self.factor_aero = 1
        self.driven_wheels = 4
    
    # Z axis
    fz_mass = -1*self.M*g
    fz_aero = [0.5*self.rho*self.factor_Cl*self.Cl*self.A*(s**2) for s in self.vehicle_speed]
    fz_total = [fz_mass + fa for fa in fz_aero]
    fz_tire = [(self.factor_drive*fz_mass+self.factor_aero*fa)/self.driven_wheels for fa in fz_aero]
    # x axis
    fx_aero = [0.5*self.rho*self.factor_Cd*self.Cd*self.A*(s**2) for s in self.vehicle_speed]
    fx_roll = (self.Cr*np.abs(fz_total)).tolist()
    fx_tire = (self.driven_wheels*(self.mu_x+self.sens_x*(self.mu_x_M*g-np.abs(fz_tire)))*np.abs(fz_tire)).tolist()
    # HUD
    print('Forces calculated successfully.')

    ## GGV Map

    # track data
    bank = 0
    incl = 0
    # lateral tire coefficients
    dmy = self.factor_grip * self.sens_y
    muy = self.factor_grip * self.mu_y
    Ny = self.mu_y_M * g
    # longitudinal tire coefficients
    dmx = self.factor_grip * self.sens_x
    mux = self.factor_grip * self.mu_x
    Nx = self.mu_x_M * g
    # normal load on all wheels
    Wz = self.M * g * np.cos(bank) * np.cos(incl)
    # induced weight from banking and inclination
    Wy = -1 * self.M*g*np.sin(bank)
    Wx = self.M*g*np.sin(incl)
    # speed map vector
    dv = 2
    v = [i for i in range(0, math.ceil(self.v_max),dv)]
    if v[-1] != self.v_max:
        v.append(self.v_max)
    # friction ellipse points
    N = 45
    # map preallocation
    GGV = np.zeros((len(v), 3, 2*N-1)).tolist()
    for i in range(0, len(v)):
        # aero forces
        Aero_Df = 0.5*self.rho*self.factor_Cl*self.Cl*self.A * v[i]**2
        Aero_Dr = 0.5*self.rho*self.factor_Cd*self.Cd*self.A * v[i]**2
        # rolling resistance
        Roll_Dr = self.Cr * np.abs(-1*Aero_Df+Wz)
        # normal load on driven wheels
        Wd = (self.factor_drive*Wz+(-1*self.factor_aero*Aero_Df))/self.driven_wheels
        # drag acceleration
        ax_drag = (Aero_Dr+Roll_Dr+Wx)/self.M
        # maximum lat acc available from tires
        ay_max = 1/self.M*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
        # max long acc available from tires
        ax_tire_max_acc = 1/self.M*(mux+dmx*(Nx-Wd))*Wd*self.driven_wheels
        # max long dec available from tires
        ax_tire_max_dec = -1/self.M * (mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
        # getting power limit from engine
        x = self.vehicle_speed
        y = self.factor_power * fx_engine
        f = interpolate.interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
        fx[i] = f(v[i]).tolist()
        ax_power_limit = 1/self.M * fx[i]
        ax_power_limit = (ax_power_limit * np.ones((N, 1))).tolist()
        # lat acc vector
        comp_conj = np.conj(np.cos(np.radians(np.linspace(0, 180, N)))).tolist()
        ay = [ay_max * c for c in comp_conj]
        # long acc vector
        ax_tire_acc = (ax_tire_max_acc * np.sqrt(1-(ay/ay_max)**2)).tolist()
        ax_acc = [min(ax_tire_acc[i], ax_power_limit[i][0])+ax_drag for i in range(0, N)]
        ax_dec = (ax_tire_max_dec * np.sqrt(1-(ay/ay_max)**2)+ax_drag).tolist()
        # saving GGV map
        GGV[i][0][:] = ax_acc + ax_dec[1:]
        GGV[i][1][:] = [ay, np.flipud(ay[2:])]
        GGV[i][2][:] = v[i] * np.ones((1, 2*N-1))
    # HUD
    print('GGV map generated successfully.')

    ## Saving vehicle

    # saving
    #save(vehname+'.mat')

    ## Plot

    # figure

veh = OpenVEHICLE('Formula 1.xlsx')