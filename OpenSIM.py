import numpy as np

class OpenSIM:
    def __init__(self, simname, tr, time, N, apex, v_max, v, ax, ay, tps, bps, V, yaw_rate, AX, AY, A, \
                    TPS, BPS, veh, steer, delta, beta, Fz_aero, Fx_aero, Fx_eng, Fx_roll, Fz_mass, Fz_total, \
                    wheel_torque, engine_torque, engine_power, engine_speed, gear, fuel_cons, fuel_cons_total, \
                        laptime, sector_time, percent_in_corners, percent_in_accel, percent_in_decel, percent_in_coast, \
                            percent_in_full_tps, percent_in_gear, energy_spent_fuel, energy_spent_mech, gear_shifts, \
                                ay_max, ax_max, ax_min, sector_v_min, sector_v_max, flag):
        ## saving results in sim structure
        self.sim_name_data = simname 
        self.distance_data = tr.x 
        self.distance_unit = 'm' 
        self.time_data = time 
        self.time_unit = 's' 
        self.N_data = N 
        self.N_unit = [] 
        self.apex_data = apex 
        self.apex_unit = [] 
        self.speed_max_data = v_max 
        self.speed_max_unit = 'm/s' 
        self.flag_data = flag 
        self.flag_unit = [] 
        self.v_data = v 
        self.v_unit = 'm/s' 
        self.Ax_data = ax 
        self.Ax_unit = 'm/s/s' 
        self.Ay_data = ay 
        self.Ay_unit = 'm/s/s' 
        self.tps_data = tps 
        self.tps_unit = [] 
        self.bps_data = bps 
        self.bps_unit = [] 
        self.elevation_data = tr.Z 
        self.elevation_unit = 'm' 
        self.speed_data = V 
        self.speed_unit = 'm/s' 
        self.yaw_rate_data = yaw_rate 
        self.yaw_rate_unit = 'rad/s' 
        self.long_acc_data = AX 
        self.long_acc_unit = 'm/s/s' 
        self.lat_acc_data = AY 
        self.lat_acc_unit = 'm/s/s' 
        self.sum_acc_data = A 
        self.sum_acc_unit = 'm/s/s' 
        self.throttle_data = TPS 
        self.throttle_unit = 'ratio' 
        self.brake_pres_data = BPS 
        self.brake_pres_unit = 'Pa' 
        self.brake_force_data = BPS*veh.phi 
        self.brake_force_unit = 'N' 
        self.steering_data = steer 
        self.steering_unit = 'deg' 
        self.delta_data = delta 
        self.delta_unit = 'deg' 
        self.beta_data = beta 
        self.beta_unit = 'deg' 
        self.Fz_aero_data = Fz_aero 
        self.Fz_aero_unit = 'N' 
        self.Fx_aero_data = Fx_aero 
        self.Fx_aero_unit = 'N' 
        self.Fx_eng_data = Fx_eng 
        self.Fx_eng_unit = 'N' 
        self.Fx_roll_data = Fx_roll 
        self.Fx_roll_unit = 'N' 
        self.Fz_mass_data = Fz_mass 
        self.Fz_mass_unit = 'N' 
        self.Fz_total_data = Fz_total 
        self.Fz_total_unit = 'N' 
        self.wheel_torque_data = wheel_torque 
        self.wheel_torque_unit = 'N.m' 
        self.engine_torque_data = engine_torque 
        self.engine_torque_unit = 'N.m' 
        self.engine_power_data = engine_power 
        self.engine_power_unit = 'W' 
        self.engine_speed_data = engine_speed 
        self.engine_speed_unit = 'rpm' 
        self.gear_data = gear 
        self.gear_unit = [] 
        self.fuel_cons_data = fuel_cons 
        self.fuel_cons_unit = 'kg' 
        self.fuel_cons_total_data = fuel_cons_total 
        self.fuel_cons_total_unit = 'kg' 
        self.laptime_data = laptime 
        self.laptime_unit = 's' 
        self.sector_time_data = sector_time 
        self.sector_time_unit = 's' 
        self.percent_in_corners_data = percent_in_corners 
        self.percent_in_corners_unit = '%' 
        self.percent_in_accel_data = percent_in_accel 
        self.percent_in_accel_unit = '%' 
        self.percent_in_decel_data = percent_in_decel 
        self.percent_in_decel_unit = '%' 
        self.percent_in_coast_data = percent_in_coast 
        self.percent_in_coast_unit = '%' 
        self.percent_in_full_tps_data = percent_in_full_tps 
        self.percent_in_full_tps_unit = '%' 
        self.percent_in_gear_data = percent_in_gear 
        self.percent_in_gear_unit = '%' 
        self.v_min_data = min(V) 
        self.v_min_unit = 'm/s' 
        self.v_max_data = max(V) 
        self.v_max_unit = 'm/s' 
        self.v_ave_data = np.mean(V) 
        self.v_ave_unit = 'm/s' 
        self.energy_spent_fuel_data = energy_spent_fuel 
        self.energy_spent_fuel_unit = 'J' 
        self.energy_spent_mech_data = energy_spent_mech 
        self.energy_spent_mech_unit = 'J' 
        self.gear_shifts_data = gear_shifts 
        self.gear_shifts_unit = [] 
        self.lat_acc_max_data = ay_max 
        self.lat_acc_max_unit = 'm/s/s' 
        self.long_acc_max_data = ax_max 
        self.long_acc_max_unit = 'm/s/s' 
        self.long_acc_min_data = ax_min 
        self.long_acc_min_unit = 'm/s/s' 
        self.sector_v_max_data = sector_v_max 
        self.sector_v_max_unit = 'm/s' 
        self.sector_v_min_data = sector_v_min 
        self.sector_v_min_unit = 'm/s' 