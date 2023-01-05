## OpenLAP Laptime Simulation Project
# 
# OpenLAP
#
# Lap time simulation using a simple point mass model for a racing vehicle
# Instructions:
# 1) Select a vehicle file created by OpenVEHICLE by assigning the full
#    path to the variable "vehiclefile".
# 2) Select a track file created by OpenTRACK by assigning the full path to
#    the variable "trackfile".
# 3) Select an export frequency in [Hz] by setting the variable "freg" to
#    the desired value.
# 4) Run the script.
# 5) The results will appear on the command window and inside the folder
#    "OpenLAP Sims". You can choose to include the date and time of each
#    simulation in the result file name by changing the
#   "use_date_time_in_name" variable to true.
#
# More information can be found in the "OpenLAP Laptime Simulator"
# videos on Youtube.
#
# This software is licensed under the GPL V3 Open Source License.
#
# Python project created by:
#
# Daniel Harper
# University of Georgia MSAI
# University of Georgia BSCS
#
# Based on MATLAB project from:
# Michael Halkiopoulos
# Cranfield University MSc Advanced Motorsport Engineer
# National Technical University of Athens MEng Mechanical Engineer
#
# January 2023

## Import libraries
import os
import time
import numpy as np
from scipy import interpolate
from scipy import signal

## Functions

def vehicle_model_lat(veh, tr, p):

    ## Initialization
    # getting track data
    g = 9.81
    r = tr.r[p]
    incl = tr.incl[p]
    bank = tr.bank[p]
    factor_grip = tr.factor_grip[p]*veh.factor_grip
    # getting vehicle data
    factor_drive = veh.factor_drive
    factor_aero = veh.factor_aero
    driven_wheels = veh.driven_wheels
    # Mass
    M = veh.M
    # normal load on all wheels
    Wz = M*g*np.rad2deg(np.cos(bank))*np.rad2deg(np.cos(incl))
    # induced weight from banking and inclination
    Wy = -1*M*g*(np.rad2deg(np.sin(bank)))
    Wx = M*g*(np.rad2deg(np.sin(incl)))

    ## speed solution
    if r == 0: # straight (limited by engine speed limit or drag)
        v = veh.v_max
        tps = 1 # full throttle
        bps = 0 # no braking
    else: #corner (may be limited by engine, drag, or cornering ability)
        ## initial speed solution
        # downforce coefficient
        D = -0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A
        # longitudinal tire coefficients
        dmy = factor_grip*veh.sens_y
        muy = factor_grip*veh.mu_y
        Ny = veh.mu_y_M*g
        # longitudinal tire coefficients
        dmx = factor_grip*veh.sens_x
        mux = factor_grip*veh.mu_x
        Nx = veh.mu_x_M*g
        # 2nd degree polynomial coefficients (a*x^2 + b*x + c = 0)
        a = -1*np.sign(r)*dmy/4*(D**2)
        b = np.sign(r)*(muy*D+(dmy/4)*(Ny*4)*D-2*(dmy/4)*Wz*D)-M*r
        c = np.sign(r)*(muy*Wz+(dmy/4)*(Ny*4)*Wz-(dmy/4)*(Wy**2))+Wy
        # calculating 2nd degree polynomial roots
        if a==0:
            v = np.sqrt(-c/b)
        elif ((b**2)-4*a*c) >= 0:
            if (-b+np.sqrt((b**2)-4*a*c))/(2/a) >= 0:
                v = np.sqrt((-b+np.sqrt((b**2)-4*a*c))/(2/a)) #TODO isn't it 2*a?
            elif (-b-np.sqrt((b**2)-4*a*c))/(2/a) >= 0:
                v = np.sqrt((-b-np.sqrt((b**2)-4*a*c))/(2/a))
            else:
                print('No real roots at point index: ', p)
        else:
            print('Discriminant <0 at point index: ', p)
        # checking for engine speed limit
        c = min(v, veh.v_max)
        ## adjusting speed for drag force compensation
        adjust_speed = True
        while adjust_speed:
            # aero forces
            Aero_Df = 0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A*(v**2)
            Aero_Dr = 0.5*veh.rho*veh.factor_Cd*veh.Cd*veh.A*(v**2)
            # rolling resistance
            Roll_Dr = veh.Cr*(-Aero_Df+Wz)
            # normal load on driven wheels
            Wd = (factor_drive*Wz+(-factor_aero*Aero_Df))/driven_wheels
            # drag acceleration
            ax_drag = (Aero_Dr+Roll_Dr+Wx)/M
            # maximum lat acc available from tires
            ay_max = np.sign(r)/M*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
            # needed lat acc make turn
            ay_needed = (v**2)*r+g*np.rad2deg(np.sin(bank)) # circular motion and track banking
            # calculating driver inputs
            if ax_drag<=0: # need throttle to compensate for drag
                # max long acc available from tires
                ax_tire_max_acc = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels
                # getting power limit from engine
                ax_power_limit = 1/M*interpolate.interp1d(veh.vehicle_speed, veh.factor_power*veh.fx_engine,v)
                # available combined lat acc at ax_net==0 => ax_tire==-ax_drag
                ay = ay_max*np.sqrt(1-(ax_drag/ax_tire_max_acc)**2) # friction ellipse
                # available combined long acc at ay_needed
                ax_acc = ax_tire_max_acc*np.sqrt(1-(ay_needed/ay_max)**2) # friction ellipse
                # getting tps value
                scale = min(-ax_drag/ax_power_limit, -ax_acc/ax_power_limit)
                tps = max(0, min(1, scale)) # making sure its positive
                bps = 0 # setting brake pressure to 0
            else: # need to brake to compensate for drag
                # max long acc available from tires
                ax_tire_max_dec = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
                # available combined lat acc at ax_net==0 => ax_tire==-ax_drag
                ay = ay_max*np.sqrt(1-(ax_drag/ax_tire_max_dec)**2) # friction ellipse
                # available combined long acc at ay_needed
                ax_dec = ax_tire_max_dec*np.sqrt(1-(ay_needed/ay_max)**2) # friction ellipse
                # getting brake input
                fx_tire = max(ax_drag*M, -ax_dec*M)
                bps = max(fx_tire, 0)*veh.beta # making sure its positive
                tps = 0 # setting throttle to 0
            # checking if tires can produce the available combined lat acc
            if ay/ay_needed<1: # not enough grip
                v = np.sqrt((ay-g*np.rad2deg(np.sin(bank)))/r)-0.001 # the (-0.001 factor is there for convergence speed)
            else: # enough grip
                adjust_speed = False

def simulate(veh, tr, simname, logid):

    ## Initialization

    # solver time
    timer_solver_start = time.time()

    # HUD
    print('Simulation started')
    logid.write('Simulation started')

    ## maximum speed curve (assuming pure lateral condition)

    v_max = np.zeros(tr.n, 1)
    bps_v_max = np.zeros(tr.n, 1)
    tps_v_max = np.zeros(tr.n, 1)
    for i in range(1, tr.n):
        v_max[i], tps_v_max[i], bps_v_max[i] = vehicle_model_lat(veh, tr, i)

    # HUD
    print('Maximum speed calculated at all points.')
    logid.write('Maximum speed calculated at all points.')

    ## finding apexes

    v_apex, apex = signal.find_peaks(-v_max) # findpeaks works for maxima, so need to flip values
    v_apex = -v_apex # flipping to get positive values
    # setting up standing start for open track configuration
    
    ## I AM CURRENTLY ON LINE 274 of OPENLAP.m, RESUME FROM HERE

## Clearing memory

os.system('cls')

## Starting timer

start_time = time.time()

## Filnames

trackfile = 'OpenTRACK Tracks/OpenTRACK_Spa-Francorchamps_Closed_Forward.mat'
vehiclefile = 'OpenVEHICLE Vehicles/OpenVEHICLE_Formula 1_Open Wheel.mat'

## Loading circuit

tr = trackfile

## Loading car

veh = vehiclefile

## Export frequency

freq = 50 # [Hz]

## Simulation name

use_date_time_in_name = False
if use_date_time_in_name:
    date_time = '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
else:
    date_time = ''
simname = 'OpenLAP Sims/OpenLAP_' + veh.name + '_' + tr.name + date_time
logfile = simname + '.log'
logid = open(logfile, 'a')

# HUD

if not os.path.exists('OpenLAP Sims'): os.makedirs('OpenLAP Sims')
if os.path.exists(logfile): os.remove(logfile)
lg = '_______                    _____________________ \n__  __ \______________________  /___    |__  __ \\\n_  / / /__  __ \  _ \_  __ \_  / __  /| |_  /_/ /\n/ /_/ /__  /_/ /  __/  / / /  /___  ___ |  ____/\n\____/ _  .___/\___//_/ /_//_____/_/  |_/_/      \n       /_/                                       '
print(lg) # command window
print('=================================================')
print('Vehicle: ' + veh.name)
print('Track: ' + tr.name)
print('Date: ' + time.strftime("%Y-%m-%d"))
print('Time: ' + time.strftime("%H:%M:%S"))
print('=================================================')
logid.write('=================================================')
logid.write('Vehicle: ' + veh.name)
logid.write('Track: ' + tr.name)
logid.write('Date: ' + time.strftime("%Y-%m-%d"))
logid.write('Time: ' + time.strftime("%H:%M:%S"))
logid.write('=================================================')

## Lap Simulation

sim = simulate(veh, tr, simname)