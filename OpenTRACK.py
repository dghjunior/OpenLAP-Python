# Import dependencies
import numpy as np
import os
import datetime
from scipy.signal import savgol_filter

# vehicle class declaration
class OpenTRACK:
    # constructor
    def __init__(self, csv_filename='Spa-Francorchamps.xlsx'):
        self.filename = csv_filename
        
        ## Mode selection

        # mode = 'logged data'
        mode = 'shape data'
        # log_mode = 'speed & yaw'
        self.log_mode = 'speed & latacc'

        ## Settings

        # meshing
        mesh_size = 1 # [m]
        #filtering for logged data mode
        filter_dt = 0.1 # [s]
        # track map rotation angle
        rotation = 0 # [deg]
        # track map shape adjuster
        lambd = 1 # [-]
        # long corner adjuster
        kappa = 1000 # [deg]

        ## Reading file

        # HUD
        print('Reading track file: ' + self.filename)
        if mode == 'logged data':
            ## from logged data

            head, data = read_logged_data(self.filename)
            self.name = head[1, 1]
            self.country = head[2, 1]
            self.city = head[3, 1]
            self.type = head[4, 1]
            self.config = head[5, 1]
            self.direction = head[6, 1]
            self.mirror = head[6, 1]
            # channels
            self.channels = head[10, :]
            self.units = head[12, :]
            # frequency
            self.freq = str(head[8, 1])
            # data columns
            col_dist = 1
            col_vel = 2
            col_yaw = 3
            col_ay = 4
            col_el = 5
            col_bk = 6
            col_gf = 7
            col_sc = 8
            # extracting data
            self.x = data[:, col_dist]
            self.v = data[:, col_vel]
            self.w = data[:, col_yaw]
            self.ay = data[:, col_ay]
            self.el = data[:, col_el]
            self.bk = data[:, col_bk]
            self.gf = data[:, col_gf]
            self.sc = data[:, col_sc]
            # converting units
            if self.units[col_dist] != 'm':
                if self.units[col_dist] == 'km':
                    self.x = self.x*1000
                elif self.units[col_dist] == 'miles':
                    self.x = self.x*1609.34
                elif self.units[col_dist] == 'ft':
                    self.x = self.x*0.3048
                else:
                    print('Check distance units.')
            if self.units[col_vel] != 'm/s':
                if self.units[col_vel] == 'km/h':
                    self.v = self.v/3.6
                elif self.units[col_vel] == 'mph':
                    self.v = self.v*0.44704
                else:
                    print('Checkk speed units.')
            if self.units[col_yaw] != 'rad/s':
                if self.units[col_yaw] == 'deg/s':
                    self.w = self.w*2*np.pi/360
                elif self.units[col_yaw] == 'rpm':
                    self.w = self.w*2*np.pi/60
                elif self.units[col_yaw] == 'rps':
                    self.w = self.w*2*np.pi
                else:
                    print('Check yaw velocity units.')
            if self.units[col_ay] != 'm/s/s':
                if self.units[col_ay] == 'G':
                    self.ay = self.ay*9.81
                elif self.units[col_ay] == 'ft/s/s':
                    self.ay = self.ay*0.3048
                else:
                    print('Check lateral acceleration units.')
            if self.units[col_el] != 'm':
                if self.units[col_el] == 'km':
                    self.el = self.el*1000
                elif self.units[col_el] == 'miles':
                    self.el = self.el*1609.34
                elif self.units[col_el] == 'ft':
                    self.el = self.el*0.3048
                else:
                    print('Check elevation units.')
            if self.units[col_bk] != 'deg':
                if self.units[col_bk] == 'rad':
                    self.bk = self.bk/2/np.pi*360
                else:
                    print('Check banking units')
            ## from shape data

            info = read_info(self.filename, 'Info')
            table_shape = read_shape_data(self.filename, 'Shape')
            table_el = read_data(self.filename, 'Elevation')
            table_bk = read_data(self.filename, 'Banking')
            table_gf = read_data(self.filename, 'Grip Factors')
            table_sc = read_data(self.filename, 'Sectors')

def fill_in(self):

    ## Track model name

    ## HUD

        if not os.path.exists('OpenTRACK Tracks'):
            os.makedirs('OpenTRACK Tracks')
        trackname = 'OpenTRACK Tracks/OpenTRACK_' + self.name + '_' + self.config + '_' + self.direction
        if self.mirror == 'On':
            trackname = trackname + '_Mirrored'
        if os.path.exists(trackname+'.log'):
            os.remove(trackname+'.log')
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
        print('Date: ' + datetime.datetime.now())
        print('====================================================================================')
        print('Track generation started.')

        ## Pre-processing

        if self.mode == 'logged data':
            # getting unique points
            self.x, rows_to_keep = unique(x)
            self.v = savgol_filter(self.v(rows_to_keep), round(self.freq*self.filter_dt))
            self.w = savgol_filter(self.w(rows_to_keep), round(self.freq*self.filter_dt))
            self.ay = savgol_filter(self.ay(rows_to_keep), round(self.freq*self.filter_dt))
            self.el = savgol_filter(self.el(rows_to_keep), round(self.freq*self.filter_dt))
            self.bk = savgol_filter(self.bk(rows_to_keep), round(self.freq*self.filter_dt))
            self.gf = self.gf(rows_to_keep)
            self.sc = self.sc(rows_to_keep)
            # shifting position vector for 0 value at start
            self.x = self.x-self.x[0]
            # curvature
            if self.log_mode == 'speed & yaw':
                self.r = self.lambd*self.w/self.v
            elif self.log_mode == 'speed & latacc':
                self.r = self.lambd*self.ay/(self.v ** 2)
            self.r = savgol_filter(self.r, round(self.freq*self.filter_dt))
            # mirroring if needed
            if self.mirror == 'On':
                self.r = -1*self.r
            # track length
            self.L = self.x[-1]
            # saving coarse position vectors
            self.xx = self.x
            self.xe = self.x
            self.xb = self.x
            self.xg = self.x
            self.xs = self.x
        else: # shape data
            # turning radius
            self.R = self.table_shape[:, 2]
            # segment length
            self.l = self.table_shape[:, 1]
            # segment type
            self.type_tmp = self.table_shape[:, 0]
            # correcting straight segment radius
            self.R[self.R==0] = np.inf
            # total length
            self.L = sum(self.l)
            # segment type variable conversion to number
            self.type = np.zeros(len(1), 1)
            ## TODO
            ## Lines 289-291
            if self.mirror == 'On':
                self.type = -1*self.type
            # removing segments with zero length
            self.R[self.l==0] = []
            self.type[self.l==0] = []
            self.l[self.l==0] = []
            
