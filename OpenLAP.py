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
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
from tqdm import tqdm
from OpenVEHICLE import OpenVEHICLE
from OpenTRACK import OpenTRACK
from OpenSIM import OpenSIM
import pickle
import warnings
from matplotlib.gridspec import GridSpec
from matplotlib import cm


## Functions

def vehicle_model_lat(veh, tr, p):

    warnings.filterwarnings("ignore")
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
    Wz = M*g*np.cos(bank)*np.cos(np.deg2rad(incl))
    # induced weight from banking and inclination
    Wy = -1*M*g*(np.sin(bank))
    Wx = M*g*(np.sin(np.deg2rad(incl)))

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
        c = np.sign(r)*(muy*Wz+(dmy/4)*(Ny*4)*Wz-(dmy/4)*(Wz**2))+Wy
        # calculating 2nd degree polynomial roots
        if a==0:
            v = np.sqrt(-c/b)
        elif ((b**2)-4*a*c) >= 0:
            if (-b+np.sqrt((b**2)-4*a*c))/(2/a) >= 0:
                v = np.sqrt((-b+np.sqrt((b**2)-4*a*c))/2/a)
            elif (-b-np.sqrt((b**2)-4*a*c))/(2/a) >= 0:
                v = np.sqrt((-b-np.sqrt((b**2)-4*a*c))/2/a)
            else:
                print('No real roots at point index: ', p)
        else:
            print('Discriminant <0 at point index: ', p)
        # checking for engine speed limit
        if 'v' not in locals():
            v = 0
        v = min(v, veh.v_max)
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
                f = interpolate.interp1d(veh.vehicle_speed, veh.factor_power*veh.fx_engine, kind='linear', fill_value='extrapolate')
                ax_power_limit = 1/M*f(v)
                # available combined lat acc at ax_net==0 => ax_tire==-ax_drag
                ay = ay_max*np.sqrt(1-(ax_drag/ax_tire_max_acc)**2) # friction ellipse
                # available combined long acc at ay_needed
                ax_acc = ax_tire_max_acc*np.sqrt(1-(ay_needed/ay_max)**2) # friction ellipse
                # getting tps value
                scale = min(-ax_drag, ax_acc)/ax_power_limit
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
    return v, tps, bps

def other_points(i, i_max):
    i_rest = list(range(0, i_max))
    i_rest.remove(i)
    return i_rest

def next_point(j, j_max, mode, tr_config):
    if mode == 1: # acceleration
        if tr_config == 'Closed':
            if j==j_max-1:
                j = j_max
                j_next = 0
            elif j==j_max:
                j = 0
                j_next = j+1
            else:
                j = j+1
                j_next = j+1
        elif tr_config == 'Open':
            j =  j+1
            j_next = j+1
    elif mode == -1: # deceleration
        if tr_config == 'Closed':
            if j==1:
                j = 0
                j_next = j_max
            elif j==0:
                j = j_max
                j_next = j-1
            else:
                j = j-1
                j_next = j-1
        elif tr_config == 'Open':
            j =  j-1
            j_next = j-1
    return j_next, j

def vehicle_model_comb(veh, tr, v, v_max_next, j, mode):
    
    warnings.filterwarnings("ignore")
    ## Initialization

    # assuming no overshoot
    overshoot = False
    # getting track data
    dx = tr.dx[j]
    r = tr.r[j]
    incl = tr.incl[j]
    bank = tr.bank[j]
    factor_grip = tr.factor_grip[j]*veh.factor_grip
    g = 9.81
    # getting vehicle data
    if mode == 1:
        factor_drive = veh.factor_drive
        factor_aero = veh.factor_aero
        driven_wheels = veh.driven_wheels
    else:
        factor_drive = 1
        factor_aero = 1
        driven_wheels = 4

    ## external forces

    # Mass
    M = veh.M
    # normal load on all wheels
    Wz = M*g*np.cos(np.deg2rad(bank))*np.cos(np.deg2rad(incl))
    # induced weight from banking and inclination
    Wy = -1*M*g*np.sin(np.deg2rad(bank))
    Wx = M*g*np.sin(np.deg2rad(incl))
    # aero forces
    Aero_Df = 0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v**2
    Aero_Dr = 0.5*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v**2
    # rolling resistance
    Roll_Dr = veh.Cr*(-1*Aero_Df+Wz)
    # normal load on driven wheels
    Wd = (factor_drive*Wz+(-1*factor_aero*Aero_Df))/driven_wheels

    ## overshoot acceleration

    # maximum allowed long acc to not overshoot at next point
    ax_max = mode*(v_max_next**2-v**2)/2/dx
    # drag acceleration
    ax_drag = (Aero_Dr+Roll_Dr+Wx)/M
    # overshoot acceleration limit
    ax_needed = ax_max-ax_drag

    ## current lat acc
    ay = v**2*r+g*np.sin(bank)

    ## tire forces

    # longitudinal tire coefficients
    dmy = factor_grip*veh.sens_y
    muy = factor_grip*veh.mu_y
    Ny = veh.mu_y_M*g
    # lateral tire coefficients
    dmx = factor_grip*veh.sens_x
    mux = factor_grip*veh.mu_x
    Nx = veh.mu_x_M*g
    # friction ellipse multiplier
    if not np.sign(ay) == 0: # in corner or compensating for banking
        # max lat acc available from tires
        ay_max = 1/M*(np.sign(ay)*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df)+Wy)
        # max combined long acc available from tires
        if np.abs(ay/ay_max) > 1: # checking if vehicle overshot (should not happen, but check exists to exclude complex numbers in solution from friction ellipse)
            ellipse_multi = 0
        else:
            ellipse_multi = np.sqrt(1-(ay/ay_max)**2) # friction ellipse
    else: # straight or no compensation for banking needed
        ellipse_multi = 1

    ## calculating driver inputs

    if ax_needed >= 0: # need tps
        # max pure long acc available from driven tires
        ax_tire_max = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels
        # max combinbed long acc available from driven tires
        ax_tire = ax_tire_max*ellipse_multi
        # getting power limit from engine
        f = interpolate.interp1d(veh.vehicle_speed, veh.factor_power*veh.fx_engine, kind='linear', fill_value='extrapolate')
        ax_power_limit = 1/M*f(v).tolist()
        # getting tps value
        scale = min(ax_tire/ax_power_limit, ax_needed/ax_power_limit)
        tps = max(0, min(1, scale)) # making sure its positive
        bps = 0 # setting brake pressure to 0
        # final long acc command
        ax_com = tps*ax_power_limit
    else: # need braking
        # max pure long acc available from all tires
        ax_tire_max = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
        # max comb long acc available from all tires
        ax_tire = ax_tire_max*ellipse_multi
        # tire braking force
        fx_tire = min(-ax_tire*M, -ax_needed*M)
        # getting brake input
        bps = max(fx_tire*veh.beta, 0) # making sure its positive
        tps = 0 # setting throttle to 0
        # final long acc command
        ax_com = -1*min(-ax_tire, -ax_needed)

    ## final results

    # total vehicle long acc
    ax = ax_com+ax_drag# next speed value
    v_next = np.sqrt(v**2+2*mode*ax*tr.dx[j])
    # correcting tps for full throttle when at v_max on straights
    if tps>0 and v/veh.v_max>=0.999:
        tps = 1
    
    ## checking for overshoot

    if v_next/v_max_next>1:
        # setting overshoot flag
        overshoot = True
        # resetting values for overshoot
        v_next = np.inf
        ax = 0
        ay = 0
        tps = -1
        bps = -1
    return v_next, ax, ay, tps, bps, overshoot

def progress_bar(flag, prg_size, logid, prg_pos, pbar):
    # current flag state
    try:
        p = np.sum(flag)/(flag.shape[0])/(flag.shape[1]) # progress percentage
    except:
        p = np.sum(flag)/(flag.shape[0])
    n = np.floor(p*prg_size) # new number of lines
    e = prg_size-n # number of spaces
    # updating progress bar in command window
    numpy.matlib.repmat('\b', 1, prg_size+1+8) # backspace to start of bar
    numpy.matlib.repmat('|', 1, int(n)) # writing lines
    np.matlib.repmat(' ', 1, int(e)) # writing spaces
    # updating progress bar in log file
    logid.seek(prg_pos) # going to start of bar
    pbar.update(p)
    ## CAN ADD THIS HERE BUT I DONT FEEL LIKE IT. MISSING LINES 982-988 of OpenLAP

def flag_update(flag, j, k, prg_size, logid, prg_pos, pbar):
    # current flag state
    p = np.sum(flag)/len(flag[0])/len(flag[1])
    n_old = np.floor(p*prg_size) # old number of lines
    # new flag state
    flag[j, k] = True
    p = np.sum(flag)/len(flag[0])/len(flag[2])
    n = np.floor(p*prg_size) # new number of lines
    # checking if state has changed enough to update progress bar
    if n>n_old:
        progress_bar(flag, prg_size, logid, prg_pos, pbar)
    return flag

def export_report(veh, tr, sim, freq, logid):
    # frequency
    freq = np.round(freq)
    # channel names
    ## CAN IMPLEMENT THIS LATER BUT DONT FEEL LIKE IT. MISSING LINES 1010-1099

def simulate(veh, tr, simname, logid):

    ## Initialization

    # solver time
    timer_solver_start = time.time()

    # HUD
    print('Simulation started')
    logid.write('Simulation started')

    ## maximum speed curve (assuming pure lateral condition)

    v_max = np.zeros(tr.n)
    bps_v_max = np.zeros(tr.n)
    tps_v_max = np.zeros(tr.n)
    for i in range(0, tr.n):
        v_max[i], tps_v_max[i], bps_v_max[i] = vehicle_model_lat(veh, tr, i)

    # HUD
    print('Maximum speed calculated at all points.')
    logid.write('Maximum speed calculated at all points.')

    ## finding apexes
    apex = signal.argrelmin(v_max)
    v_apex = [v_max[i] for i in apex]
    #v_apex = -v_apex # flipping to get positive values

    # setting up standing start for open track configuration
    if tr.config == 'Open':
        if not apex[0] == 1: # if index 1 is not already an apex
            apex = np.insert(apex, 0, 1) # inject index 1 as apex
            v_apex = np.insert(v_apex, 0, 0) # inject standing start
        else: # index 1 is already an apex
            v_apex[0] = 0 # set standing start at index 0
    # checking if no apexes found and adding one if needed
    if len(apex) == 0:
        print(len(v_max))
        v_apex = min(v_max)
        apex = v_max.tolist().index(v_apex)
    # reordering apexes for solver time optimization
    apex_table = sorted(np.column_stack((v_apex[0], apex[0])).tolist())
    v_apex = [apex_table[i][0] for i in range(0, len(apex_table))]
    apex = [apex_table[i][1] for i in range(0, len(apex_table))]
    # getting driver inputs at apexes
    tps_apex = [tps_v_max[int(a)] for a in apex]
    bps_apex = [bps_v_max[int(a)] for a in apex]

    # HUD
    print('Found all apexes on track.')
    logid.write('Found all apexes on track.')

    ## simulation

    # memory preallocation
    N = len(apex) # number of apexes
    flag = np.zeros((tr.n, 2)) # flag for checking that speed has been correctly evaluated
    # 1st matrix dimension equal to number of points in track mesh
    # 2nd matrix dimension equal to number of apexes
    # 3rd matrix dimension equal to 2 if needed (1 copy for acceleration and 1 copy for deceleration)
    v = np.inf*np.ones((tr.n, N, 2))
    ax = np.zeros((tr.n, N, 2))
    ay = np.zeros((tr.n, N, 2))
    tps = np.zeros((tr.n, N, 2))
    bps = np.zeros((tr.n, N, 2))

    # HUD
    print('Starting acceleration and deceleration.')
    pbar = tqdm(total=100, leave=False)
    logid.write('Starting acceleration and deceleration.')
    prg_size = 30
    prg_pos = logid.tell()
    logid.write('________________________________________________\n')
    logid.write('|_Apex__|_Point_|_Mode__|___x___|___v___|_vmax_|\n')

    same_reached = True

    # running simulation
    for i in range(0, N): # apex number
        for k in range(0, 2): # mode number
            if k == 0: # acceleration
                mode = 1
                k_rest = 1
            if k == 1: # deceleration
                mode = -1
                k_rest = 0
            if not (tr.config == 'Open' and mode==-1 and i == 0): # does not run in decel mode at standing start in open track
                # getting other apex for later checking
                i_rest = other_points(i, N)
                if len(i_rest) == 0:
                    i_rest = i
                # getting apex index
                j = int(apex[i])
                # saving speed and latacc and driver inputs from presolved apex
                v[j, i, k] = v_apex[i]
                ay[j, i, k] = v_apex[i]**2*tr.r[j]
                tps[j, :, 0] = tps_apex[i]*np.ones((1, N))
                bps[j, :, 0] = bps_apex[i]*np.ones((1, N))
                tps[j, :, 1] = tps_apex[i]*np.ones((1, N))
                bps[j, :, 1] = bps_apex[i]*np.ones((1, N))
                # setting apex flag
                flag[j, k] = True
                # getting next point index
                j_next = next_point(j, tr.n-1, mode, tr.config)[1]
                if not (tr.info['Config'] == 'Open' and mode == 1 and i == 0): # if not in standing start
                    # assume same speed right after apex
                    v[j_next, i, k] = v[j, i, k]
                    # moving to the next point index
                    j_next, j = next_point(j, tr.n-1, mode, tr.config)
                same_reached = True
                while same_reached:
                    # writing to log file
                    logid.write('%7d\t%7d\t%7d\t%7.1f\t%7.2f\t%7.2f\n' % (i, j, k, tr.x[j], v[j, i, k], v_max[j]))
                    # calculating speed, accelerations and driver inputs from vehicle model
                    v[j_next, i, k], ax[j, i, k], ay[j, i, k], tps[j, i, k], bps[j, i, k], overshoot = vehicle_model_comb(veh, tr, v[j, i, k], v_max[j_next], j, mode)
                    # checking for limit
                    if overshoot:
                        same_reached = False
                    # checking if point is already solved in other apex iteration
                    if flag[j, k] or flag[j, k_rest]:
                        if max(v[j_next, i, k]>=v[j_next, i_rest, k]) or max(v[j_next, i, k]>v[j_next, i_rest, k_rest]):
                            same_reached = False
                    # updating flag and progress bar
                    flag = flag_update(flag, j, k, prg_size, logid, prg_pos, pbar)
                    # moving to next point index
                    j_next, j = next_point(j, tr.n-1, mode, tr.config)
                    # checking if lap is completed
                    if tr.config == 'Closed':
                        if j == apex[i]: # made it to the same apex
                            same_reached = False
                    elif tr.config == 'Open':
                        if j == tr.n: # mad it to the end
                            flag = flag_update(flag, j, k, prg_size, logid, prg_pos, pbar)
                        if j==1: # made it to the start
                            same_reached = False
    # HUD
    progress_bar(np.amax(flag, 1), prg_size, logid, prg_pos, pbar)
    pbar.close()
    logid.write('\n')
    print('Velocity profile calculated.')
    print('Solver time is: ' + str(time.time() - start_time) + ' seconds.')
    print('Post-processing initialized.')
    logid.write('________________________________________________\n')
    if np.sum(flag)<(flag.shape[0])/(flag.shape[1]):
        logid.write('Velocity profile calculation error.\n')
        logid.write('Points not calculated.\n')
        p = list(range(1, tr.n))
        #TODO fix this later
        # logid.write(p[np.amin(flag, 1).astype(int)])
    else:
        logid.write('Velocity profile calculated successfully.\n')
    logid.write('Solver time is: ')
    logid.write(str(time.time() - start_time))
    logid.write(' seconds.\n')
    logid.write('Post-processing initialized.\n')

    ## post processing results

    # result preallocation
    V = np.zeros((tr.n, 1))
    AX = np.zeros((tr.n, 1))
    AY = np.zeros((tr.n, 1))
    TPS = np.zeros((tr.n, 1))
    BPS = np.zeros((tr.n, 1))
    # solution selection
    for i in range(0, tr.n):
        IDX = len(v[i, :, 0])
        V[i] = np.min(v[i]) # order of k in v[i,:,k] inside min() must be the same order to not miss correct values
        idx = np.where(v[i, :] == V[i])[0][0]
        if idx<=IDX: # solved in acceleration
            AX[i] = ax[i, idx, 0]
            AY[i] = ay[i, idx, 0]
            TPS[i] = tps[i, idx, 0]
            BPS[i] = bps[i, idx, 0]
        else: # solved in deceleration
            AX[i] = ax[i, idx-IDX, 1]
            AY[i] = ay[i, idx-IDX, 1]
            TPS[i] = tps[i, idx-IDX, 1]
            BPS[i] = bps[i, idx-IDX, 1]
    
    # HUD
    print('Correct solution selected from modes.')
    logid.write('Correct solution selected from modes.\n')

    # laptime calculation
    if tr.config == 'Open':
        timer = np.cumsum([tr.dx[2]/V[2], tr.dx[2:]/V[2:]])
    else:
        timer = np.cumsum(np.divide(np.array(tr.dx), np.rot90(V)[0]))
    sector_time = np.zeros((int(max(tr.sector)), 1))
    indexes = np.where(np.roll(tr.sector, 1) != tr.sector)[0][1:]
    indexes = np.append(indexes, tr.n-1)
    sector_times = [timer[j] for j in indexes]
    for i in range(0, int(max(tr.sector))):
        sector_time[i] = timer[indexes[i]] - np.sum(sector_time[:i])
    laptime = timer[-1]

    # HUD
    print('Laptime calculated.')
    logid.write('Laptime calculated.\n')

    # calculating forces
    M = veh.M
    g = 9.81
    A = np.sqrt(AX**2+AY**2)
    Fz_mass = -1*M*g*np.cos(tr.bank)*np.cos(tr.incl)
    Fz_aero = 0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A*V**2
    Fz_total = Fz_mass+Fz_aero
    Fx_aero = 0.5*veh.rho*veh.factor_Cd*veh.Cd*veh.A*V**2
    Fx_roll = veh.Cr*np.abs(Fz_total)
    # HUD
    print('Forces calculated.')
    logid.write('Forces calculated.\n')

    # calculating yaw motion, vehicle slip angle and steering input
    yaw_rate = V*tr.r
    delta = np.zeros((tr.n, 1))
    beta = np.zeros((tr.n, 1))
    for i in range(0, tr.n):
        B = np.array(M*V[i]**2*tr.r[i]+M*g*np.sin(tr.bank[i]))
        B = np.append(B, 0.0)
        sol = np.linalg.lstsq(np.array(veh.C).T, B.T)[0].T
        delta[i] = sol[0]+np.arctan(veh.L*tr.r[i])
        beta[i] = sol[1]
    steer = delta*veh.rack
    # HUD
    print('Yaw motion calculated.')
    print('Steering angles calculated.')
    print('Vehicle slip angles calculated.')
    logid.write('Yaw motion calculated.\n')
    logid.write('Steering angles calculated.\n')
    logid.write('Vehicle slip angles calculated.\n')

    # calculating engine metrics
    f = interpolate.interp1d(veh.vehicle_speed, veh.wheel_torque, kind='linear', fill_value='extrapolate')
    wheel_torque = TPS*f(V)
    
    Fx_eng = wheel_torque*veh.tire_radius
    f = interpolate.interp1d(veh.vehicle_speed, veh.engine_torque, kind='linear', fill_value='extrapolate')
    engine_torque = TPS*f(V)
    
    f = interpolate.interp1d(veh.vehicle_speed, veh.engine_power, kind='linear', fill_value='extrapolate')
    engine_power = TPS*f(V)
    
    f = interpolate.interp1d(veh.vehicle_speed, veh.engine_speed, kind='linear', fill_value='extrapolate')
    engine_speed = f(V)
    
    f = interpolate.interp1d(veh.vehicle_speed, veh.gear, kind='linear', fill_value='extrapolate')
    gear = f(V)
    
    fuel_cons = np.cumsum(wheel_torque/veh.tire_radius*tr.dx/veh.n_primary/veh.n_gearbox/veh.n_final/veh.n_thermal/veh.fuel_LHV)
    fuel_cons_total = fuel_cons[-1]
    # HUD
    print('Engine metrics calculated.')
    logid.write('Engine metrics calculated.\n')

    # calculating kpis
    percent_in_corners = np.sum(tr.r != 0)/tr.n*100
    percent_in_accel = np.sum(TPS>0)/tr.n*100
    percent_in_decel = np.sum(BPS>0)/tr.n*100
    percent_in_coast = np.sum(np.logical_and(not BPS.all(), not TPS.all()))/tr.n*100
    percent_in_full_tps = np.sum(TPS==1)/tr.n*100
    percent_in_gear = np.zeros((veh.nog, 1))
    for i in range(1, veh.nog):
        percent_in_gear[i] = np.sum(gear==i)/tr.n*100
    energy_spent_fuel = fuel_cons*veh.fuel_LHV
    energy_spent_mech = energy_spent_fuel*veh.n_thermal
    gear_shifts = np.sum(np.abs(np.diff(gear)))
    i = max(np.abs(AY))
    ay_max = AY[int(i)]
    ax_max = max(AX)
    ax_min = min(AX)
    sector_v_max = np.zeros((int(max(tr.sector)), 1))
    sector_v_min = np.zeros((int(max(tr.sector)), 1))
    for i in range(0, int(max(tr.sector))):
        sector_V = np.array([V[j].tolist()[0] for j in range(len(V)) if tr.sector[j] == i+1.0])
        sector_v_max[i] = np.max(sector_V)
        sector_v_min[i] = np.min(sector_V)
    # HUD
    print('KPIs calculated.')
    print('Post-processing finished.')
    logid.write('KPIs calculated.\n')
    logid.write('Post-processing finished.\n')

    sim = OpenSIM(simname, tr, time, N, apex, v_max, v, ax, ay, tps, bps, V, yaw_rate, AX, AY, A, \
                    TPS, BPS, veh, steer, delta, beta, Fz_aero, Fx_aero, Fx_eng, Fx_roll, Fz_mass, Fz_total, \
                    wheel_torque, engine_torque, engine_power, engine_speed, gear, fuel_cons, fuel_cons_total, \
                        laptime, sector_time, percent_in_corners, percent_in_accel, percent_in_decel, percent_in_coast, \
                            percent_in_full_tps, percent_in_gear, energy_spent_fuel, energy_spent_mech, gear_shifts, \
                                ay_max, ax_max, ax_min, sector_v_min, sector_v_max, flag)

    # HUD
    print('Simulation results saved.')
    print('Simulation completed.')
    logid.write('Simulation results saved.')
    logid.write('Simulation completed.')
    return sim

## Clearing memory

os.system('cls')

## Starting timer

start_time = time.time()

## Filnames

#trackfile = 'OpenTRACK Tracks/OpenTRACK_Spa-Francorchamps_Closed_Forward.mat'
#vehiclefile = 'OpenVEHICLE Vehicles/OpenVEHICLE_Formula 1_Open Wheel.mat'
trackfile = 'Spa-Francorchamps.xlsx'
vehiclefile = 'Formula 1.xlsx'


## Loading circuit

# tr = OpenTRACK(trackfile)
with open('OpenTRACK Tracks/OpenTRACK_Spa-Francorchamps_Closed_Forward.pkl', 'rb') as f:
    tr = pickle.load(f)

## Loading car

# veh = OpenVEHICLE(vehiclefile)
with open('OpenVEHICLE Vehicles/OpenVEHICLE_Formula 1_Open Wheel.pkl', 'rb') as f:
    veh = pickle.load(f)

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
#if os.path.exists(logfile): 
#    os.remove(logfile)
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

sim = simulate(veh, tr, simname, logid)

## Displaying laptime

print('Laptime: ' + str(sim.laptime_data) + ' s')
logid.write('Laptime: ' + str(sim.laptime_data) + ' s')
for i in range(0, int(max(tr.sector))):
    print('Sector ' + str(i+1) + ': ' + str(sim.sector_time_data[i]) + ' s')
    logid.write('Sector ' + str(i+1) + ': ' + str(sim.sector_time_data[i]) + ' s')

## Plotting results

# figure window
px = 1/plt.rcParams['figure.dpi']
H = 900-90
W = 1200
f = plt.figure()
f.set_size_inches(W*px, H*px, forward=True)
f.suptitle(simname, fontsize=16)

# setting rows & columns
gs = GridSpec(nrows=9, ncols=2)
# x axis limits
xlimit = [tr.x[0], tr.x[-1]]
# setting legend location
loc = 'right'

# speed
ax0 = f.add_subplot(gs[0, :])
color = 'tab:blue'
ax0.plot(tr.x, sim.speed_data*3.6, color = color)
ax0.legend({'Speed'}, loc=loc, facecolor='white', framealpha=1)
ax0.set_xlabel('Distance [m]', fontsize=8)
ax0.set_xlim(xlimit)
ax0.set_ylabel('Speed [km/h]', fontsize=8)
ax0.tick_params(axis='both', labelsize=8)
ax0.grid(True)

# elevation and curvature
ax1 = f.add_subplot(gs[1, :])
lns1 = ax1.plot(tr.x, tr.Z, color = color, label='Elevation')
ax1.set_xlabel('Distance [m]', fontsize=8)
ax1.set_xlim(xlimit)
ax1.set_yticks([380, 400, 420, 440, 460])
ax1.tick_params(axis='y', colors=color, labelsize=8)
ax1.set_ylabel('Elevation [m]', color=color, fontsize=8)
ax1.grid(True)
ax2 = ax1.twinx()
color = 'tab:red'
lns2 = ax2.plot(tr.x, tr.r, color = color, label='Curvature')
ax2.set_ylabel('Curvature [m^-1]', color=color, fontsize=8)
ax2.set_yticks([-0.1, -0.05, 0])
ax2.tick_params(axis='y', colors=color, labelsize=8)
ax1.tick_params(axis='x', labelsize=8)
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=loc, facecolor='white', framealpha=1)

# accelerations
ax3 = f.add_subplot(gs[2, :])
color = 'tab:blue'
lns1 = ax3.plot(tr.x, sim.long_acc_data, color = color, label='LonAcc')
ax3.set_xlabel('Distance [m]', fontsize=8)
ax3.set_xlim(xlimit)
ax3.set_yticks([-100, 0, 100])
ax3.set_ylabel('Acceleration [m/s^2]', fontsize=8)
color = 'tab:red'
lns2 = ax3.plot(tr.x, sim.lat_acc_data, color = color, label='LatAcc')
color = 'black'
lns3 = ax3.plot(tr.x, sim.sum_acc_data, color = color, linestyle='dashed', label='GSum')
ax3.set_ylabel('Acceleration [m/s^2]', fontsize=8)
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc=loc, facecolor='white', framealpha=1)
ax3.tick_params(axis='both', labelsize=8)
ax3.grid(True)

# drive inputs
ax4 = f.add_subplot(gs[3, :])
color = 'tab:blue'
tps = [t*100 for t in sim.throttle_data]
ax4.plot(tr.x, tps, color = color)
color = 'tab:red'
bps = [b/(10**5) for b in sim.brake_pres_data]
ax4.plot(tr.x, bps, color = color)
ax4.legend({'tps', 'bps'}, loc=loc, facecolor='white', framealpha=1)
ax4.set_xlabel('Distance [m]', fontsize=8)
ax4.set_xlim(xlimit)
ax4.set_yticks([0, 50, 100])
ax4.set_ylabel('input [%]', fontsize=8)
ax4.tick_params(axis='both', labelsize=8)
ax4.grid(True)
ax4.set_ylim([-10, 110])

# steering inputs
ax5 = f.add_subplot(gs[4, :])
color = 'tab:blue'
ax5.plot(tr.x, sim.steering_data, color = color)
color = 'tab:red'
ax5.plot(tr.x, sim.delta_data, color = color)
color = 'yellow'
ax5.plot(tr.x, sim.beta_data, color = color)
ax5.legend({'Steering wheel', 'Steering delta', 'Vehicle slip angle beta'}, loc=loc, facecolor='white', framealpha=1)
ax5.set_xlabel('Distance [m]', fontsize=8)
ax5.set_xlim(xlimit)
ax5.set_yticks([-100, -50, 0, 50])
ax5.set_ylabel('angle [deg]', fontsize=8)
ax5.tick_params(axis='both', labelsize=8)
ax5.grid(True)

# ggv circle
ax6 = f.add_subplot(gs[5:, 0], projection='3d')
ax6.scatter3D(sim.lat_acc_data, sim.long_acc_data, sim.speed_data*3.6, color='red')
x = []
y = []
z = []
for i in range(0, len(veh.v)):
    x.append(veh.GGV[i][1])
    y.append(veh.GGV[i][0])
    z.append(veh.GGV[i][2][0])
x = np.array(x)
y = np.array(y)
z = np.array(z)
ax6.plot_surface(x, y, z)
my_col = cm.viridis(z/np.amax(z))
ax6.plot_surface(x, y, z, rstride=1, cstride=1, facecolors = my_col,
    linewidth=0, antialiased=True)
ax6.set_xlabel('Lat acc [m/s^2]')
ax6.set_ylabel('Long acc [m/s^2]')
ax6.set_xticks([-100, -50, 0, 50, 100])
ax6.set_yticks([-100, -50, 0, 50])
ax6.w_zaxis.line.set_lw(0.)
ax6.set_zticks([])
# ax6.set_zlabel('Speed [m/s]')
ax6.view_init(azim=-90, elev=90)
ax6.locator_params(axis='x', nbins=5)

# track map
ax7 = f.add_subplot(gs[5:, 1])
ax7.scatter(tr.X, tr.Y, c=sim.speed_data*3.6, marker='.', linewidth=0.05)
# im = ax7.imshow(sim.speed_data)
# cbar = plt.colorbar(im, ax=ax7)
ax7.plot(tr.arrow[0], tr.arrow[1], color='black')
ax7.legend({'Track Map'})
ax7.set_xlabel('X [m]')
ax7.set_ylabel('Y [m]')
ax7.set_xticks([-1500, -1000, -500, 0])
ax7.set_yticks([0, 500, 1000])
ax7.grid(True)

plt.tight_layout()
plt.show()