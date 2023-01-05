## OpenLAP Laptime Simulation Project
#
# OpenVEHICLE
#
# Racing vehicle model file creation for use in OpenLAP and OpenDRAG
# Instructions:
# 1) Select a vehicle excel file containing the vehicles information by 
#   assigning the full path to the variable 'filename'. Use the
#   "OpenVEHICLE tmp.xlsx" file to create a new vehicle excel file.
#2) Run the script.
# 3) The results will appear on the command window and inside the folder
#   "OpenVEHICLE Vehicles".
#
# More information can be found in the "OpenLAP Laptime Simulator"
# videos on Youtube.
#
# This software is license under the GPL V3 Open Source License.
#
# Python project created by:
#
# Daniel Harper
# University of Georgia MSAI
# University of Georgia BSCS
#
# Based on the MATLAB project from Michael Halkiopoulos
# github.com/mc12027/OpenLAP-Lap-Time-Simulator
#
# January 2023

## Imports
import os
import pandas as pd
import datetime
import math
import numpy as np
from scipy import interpolate

## Clearing Memory

os.system('cls')
#os.system('clear')

## Vehicle file selection

filename = 'Formula 1.xlsx'

## Reading vehicle file

info = pd.read_excel(filename, sheet_name = 'Info').values
data = pd.read_excel(filename, sheet_name = 'Torque Curve').values

## Getting variables

# info
name = info[0][2]
car_type = info[1][2]
# index
i = 3
#mass
M = info[i][2]; i = i+1
df = info[i][2]/100; i = i+1
# wheelbase
L = info[i][2]/1000; i = i+1
# steering rack ratio
rack = info[i][2]; i = i+1
# aerodynamics
Cl = info[i][2]; i = i+1
Cd = info[i][2]; i = i+1
factor_Cl = info[i][2]; i = i+1
factor_Cd = info[i][2]; i = i+1
da = info[i][2]/100; i = i+1
A = info[i][2]/100; i = i+1
rho = info[i][2]; i = i+1
# brakes
br_disc_d = info[i][2]/1000; i = i+1
br_pad_h = info[i][2]/1000; i = i+1
br_pad_mu = info[i][2]; i = i+1
br_nop = info[i][2]; i = i+1
br_pist_d = info[i][2]; i = i+1
br_mast_d = info[i][2]; i = i+1
br_ped_r = info[i][2]; i = i+1
# tires
factor_grip = info[i][2]; i = i+1
tire_radius = info[i][2]/1000; i = i+1
Cr = info[i][2]; i = i+1
mu_x = info[i][2]; i = i+1
mu_x_M = info[i][2]; i = i+1
sens_x = info[i][2]; i = i+1
mu_y = info[i][2]; i = i+1
mu_y_M = info[i][2]; i = i+1
sens_y = info[i][2]; i = i+1
CF = info[i][2]; i = i+1
CR = info[i][2]; i = i+1
# engine
factor_power = info[i][2]; i = i+1
n_thermal = info[i][2]; i = i+1
fuel_LHV = info[i][2]; i = i+1
# drivetrain
drive = info[i][2]; i = i+1
shift_time = info[i][2]; i = i+1
n_primary = info[i][2]; i = i+1
n_final = info[i][2]; i = i+1
n_gearbox = info[i][2]; i = i+1
ratio_primary = info[i][2]; i = i+1
ratio_final = info[i][2]; i = i+1
ratio_gearbox = info[i:][2]; i = i+1
nog = len(ratio_gearbox)

## HUD

if not os.path.exists('OpenVEHICLE Vehicles'): os.makedirs('OpenVEHICLE Vehicles')
vehname = 'OpenVEHICLE Vehicles/OpenVEHICLE_' + name + '_' + car_type
os.remove(vehname + '.log') if os.path.exists(vehname + '.log') else None
print('_______                    ___    ________________  ________________________________')
print('__  __ \_____________________ |  / /__  ____/__  / / /___  _/_  ____/__  /___  ____/')
print('_  / / /__  __ \  _ \_  __ \_ | / /__  __/  __  /_/ / __  / _  /    __  / __  __/   ')
print('/ /_/ /__  /_/ /  __/  / / /_ |/ / _  /___  _  __  / __/ /  / /___  _  /___  /___   ')
print('\____/ _  .___/\___//_/ /_/_____/  /_____/  /_/ /_/  /___/  \____/  /_____/_____/   ')
print('       /_/                                                                          ')
print('====================================================================================')
print(filename)
print('File read successfully')
print('====================================================================================')
print("Name: " + name)
print("Type: " + car_type)
now = datetime.datetime.now()
print("Date: " + str(now.month) + '/' + str(now.day) + '/' + str(now.year))
print("Time: " + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
print('====================================================================================')
print('Vehicle generation started.')

## Brake Model

br_pist_a = br_nop*math.pi*(br_pist_d/1000)**2/4 # [m2]
br_mast_a = math.pi*(br_mast_d/1000)**2/4 # [m2]
beta = tire_radius/(br_disc_d/2-br_pad_h/2)/br_pist_a/br_pad_mu/4 # [Pa/N] per wheel
phi = br_mast_a/br_ped_r*2 # [-] for both systems
# HUD
print('Braking model generated successfully.')

## Steering Model

a = (1-df)*L # distance of front axle from center of mass [mm]
b = -1*df*L # distance of rear axle from center of mass [mm]
C = [[(2*CF), (2*(CF+CR))],[(2*(CF*a)),(2*(CF*a+CR*b))]] # steering model matrix
# HUD
print('Steering model generated successfully.')

## Driveline Model

# fetching engine curves
en_speed_curve = data[:][1] # [rpm]
en_torque_curve = data[:][2] # [N*m]
en_power_curve = [(en_torque_curve[i] * en_speed_curve[i])*2*math.pi/60*tire_radius for i in range(len(en_torque_curve))] # [W]
# memory preallocation
# wheel speed per gear for every engine speed value
wheel_speed_gear = np.zeros([len(en_speed_curve),nog])
# vehicle speed per gear for every engine speed value
vehicle_speed_gear = np.zeros([len(en_speed_curve),nog])
# wheel torque per gear for every engine speed value
wheel_torque_gear = np.zeros([len(en_speed_curve),nog])
# calculating values for each gear and engine speed
for i in range(1, nog):
    wheel_speed_gear[:, i] = en_speed_curve/ratio_primary/ratio_gearbox[i]/ratio_final
    vehicle_speed_gear[:, i] = wheel_speed_gear[:, i]*2*math.pi/60*tire_radius
    wheel_torque_gear[:, i] = en_torque_curve*ratio_primary*ratio_gearbox[i]*ratio_final*n_primary*n_gearbox*n_final
# minimum and maximum vehicle speeds
v_min = min(vehicle_speed_gear[vehicle_speed_gear>0])
v_max = max(vehicle_speed_gear[vehicle_speed_gear>0])
# new speed vector for fine meshing
dv = 0.5/3.6
vehicle_speed = np.linspace(v_min, v_max, int((v_max-v_min)/dv))
# memory preallocation
# gear
gear = np.zeros(len(vehicle_speed))
# engine tractive force
fx_engine = np.zeros(len(vehicle_speed))
# engine tractive force per gear
fx = np.zeros([len(vehicle_speed),nog])
# optimizing gear selection and calculating tractive force
for i in range(1, len(vehicle_speed)):
    # going through the gears
    for j in range(1, nog):
        fx[i][j] = interpolate.interp1d(vehicle_speed_gear[:, j], wheel_torque_gear[:, j]/tire_radius, vehicle_speed[i], 'linear', 0)
    # getting maximum tractive force and gear
    [fx_engine[i], gear[i]] = max(fx[i:])
# adding values for 0 speed to vectors for interpolation purposes at low speeds
vehicle_speed = [[0][vehicle_speed]]
gear = [[gear[1]][gear]]
fx_engine = [[fx_engine[1]][fx_engine]]
# final vectors
# engine speed
engine_speed = ratio_final*ratio_gearbox[gear]*ratio_primary*vehicle_speed/tire_radius/2/math.pi*60
# wheel torque
wheel_torque = fx_engine*tire_radius
# engine torque
engine_torque = wheel_torque/ratio_final/ratio_gearbox[gear]/ratio_primary/n_primary/n_gearbox/n_final
# engine power
engine_power = engine_torque*engine_speed/60*2*math.pi
# HUD
print('Driveline model generated successfully.')

## Shifting Points and Rev Drops

# finding gear changes
gear_change = np.diff(gear)
# getting speed right before and after gear change
gear_change = np.asarray([[gear_change][0]] == [[0][gear_change]]).astype(np.int32)
#getting engine speed at gear change
engine_speed_gear_change = engine_speed[gear_change]
# getting shift points
shift_points = engine_speed_gear_change[0:2:len(engine_speed_gear_change)]
# getting arrive points
arrive_points = engine_speed_gear_change[1:2:len(engine_speed_gear_change)]
# calculating rev drops
rev_drops = shift_points - arrive_points
# creating shifting table
rownames = []
for i in range(1, nog-1):
    rownames[i] = str(i) + '-' + str(i+1)
shifting = np.zeros((len(rownames), 4))
for i in range(1, len(rownames)):
    shifting[i][0] = rownames[i]
shifting[0][0] = 'shift_points'
shifting[0][1] = 'arrive_points'
shifting[0][2] = 'rev_drops'
print(shifting)