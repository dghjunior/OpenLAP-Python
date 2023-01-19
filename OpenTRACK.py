# Import dependencies
import numpy as np
import os
import datetime
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from scipy import signal
import pandas as pd
from scipy import interpolate

# vehicle class declaration
class OpenTRACK:
    # constructor
    def __init__(self, csv_filename='Spa-Francorchamps.xlsx'):
        self.filename = csv_filename
        
        ## Mode selection

        # self.mode = 'logged data'
        self.mode = 'shape data'
        # log_mode = 'speed & yaw'
        self.log_mode = 'speed & latacc'

        ## Settings

        # meshing
        self.mesh_size = 1 # [m]
        #filtering for logged data mode
        self.filter_dt = 0.1 # [s]
        # track map rotation angle
        self.rotation = 0 # [deg]
        # track map shape adjuster
        self.lambd = 1 # [-]
        # long corner adjuster
        self.kappa = 1000 # [deg]

        ## Reading file
        ## TODO
        # Using defaults from Spa for now
        self.name = 'Spa-Francorchamps'
        self.country = 'Belgium'
        self.city = 'Francorchamps'
        self.type = 'Permanent'
        self.config = 'Closed'
        self.direction = 'Forward'
        self.mirror = 'Off'
        self.n = 6955


        # HUD
        print('Reading track file: ' + self.filename)
        if self.mode == 'logged data':
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
        fill_in(self)

def fill_in(self):

    self.info = read_info(self, self.filename, 'Info')
    self.table_shape = read_shape_data(self, self.filename, 'Shape')
    self.table_el = read_data(self, self.filename, 'Elevation')
    self.table_bk = read_data(self, self.filename, 'Banking')
    self.table_gf = read_data(self, self.filename, 'Grip Factors')
    self.table_sc = read_data(self, self.filename, 'Sectors')
    ## Track model name

    ## HUD

    if not os.path.exists('OpenTRACK Tracks'):
        os.makedirs('OpenTRACK Tracks')
    trackname = 'OpenTRACK Tracks/OpenTRACK_' + self.name + '_' + self.config + '_' + self.direction
    if self.mirror == 'On':
        trackname = trackname + '_Mirrored'
    if os.path.exists(trackname+'.log'):
        os.remove(trackname+'.log')
    print('_______                    ____________________________________ __')
    print('__  __ \______________________  __/__  __ \__    |_  ____/__  //_/')
    print('_  / / /__  __ \  _ \_  __ \_  /  __  /_/ /_  /| |  /    __  ,<   ')
    print('/ /_/ /__  /_/ /  __/  / / /  /   _  _, _/_  ___ / /___  _  /| |  ')
    print('\____/ _  .___/\___//_/ /_//_/    /_/ |_| /_/  |_\____/  /_/ |_|  ')
    print('       /_/                                                        ')
    print('====================================================================================')
    print(self.filename)
    print('File read sucessfuly')
    print('====================================================================================')
    print('Name: ' + self.name)
    print('Type: ' + self.type)
    print('Date: ' + str(datetime.datetime.now()))
    print('====================================================================================')
    print('Track generation started.')

    ## Pre-processing

    if self.mode == 'logged data':
        # getting unique points
        self.x, rows_to_keep = np.unique(self.x)
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
        self.R = self.table_shape['Corner Radius'].values
        # segment length
        self.l = self.table_shape['Section Length'].values
        # segment type
        self.type_tmp = self.table_shape['Type'].values
        # correcting straight segment radius
        self.R[self.R==0] = np.inf
        # total length
        self.L = sum(self.l)
        # segment type variable conversion to number
        self.type = np.zeros((len(self.l), 1))
        self.type[self.type_tmp == 'Straight'] = 0
        self.type[self.type_tmp == 'Left'] = 1
        self.type[self.type_tmp == 'Right'] = -1
        if self.mirror == 'On':
            self.type = -1*self.type
        # removing segments with zero length
        for i, v in enumerate(self.l):
             if v == 0:
                self.R = np.delete(self.R, i)
                self.l = np.delete(self.l, i)
                self.type = np.delete(self.type, i)
        # injecting points at long corners
        self.angle_seg = self.l/self.R
        j = 1
        self.RR = self.R
        self.ll = self.l
        self.tt = self.type
        for i in range(0, len(self.l)):
            if self.angle_seg[i] > self.kappa:
                l_inj = min([self.ll[j]/3, self.kappa*self.R[i]])
                self.ll = [self.ll[1:j-1], l_inj, self.ll[j]*l_inj, l_inj, self.ll[j+1:]]
                self.RR = [self.RR[1:j-1], self.RR[j], self.RR[j], self.RR[j], self.RR[j+1:]]
                self.tt = [self.tt[1:j-1], self.tt[j], self.tt[j], self.tt[j], self.tt[j+1:]]
                j = j+3
            else:
                j = j+1
        self.R = self.RR
        self.l = self.ll
        self.type = self.tt
        # replacing consecutive straights
        for i in range(0, len(self.l)-1):
            j = 0
            while True:
                if self.type[i+j] == 0 and self.type[i] == 0 and self.l[i] != -1:
                    self.l[i] = self.l[i]+self.l[i+j]
                    self.l[i+j] = -1
                else:
                    break
                j = j+1
        for i, v in enumerate(self.l):
             if v == -1:
                self.R = np.delete(self.R, i)
                self.l = np.delete(self.l, i)
                self.type = np.delete(self.type, i)
        # final segment point calculation
        self.X = np.cumsum(self.l) # end position of each segment
        self.XC = np.cumsum(self.l)-self.l/2 # center position of each segment
        j = 0 # index
        self.x = np.zeros((len(self.X)+sum(self.R==np.inf), 1)) # preallocation
        self.r = np.zeros((len(self.X)+sum(self.R==np.inf), 1)) # preallocation
        for i in range(0, len(self.X)):
            if self.R[i] == np.inf: # end of straight point injection
                self.x[j] = self.X[i]-self.l[i]
                self.x[j+1] = self.X[i]
                j = j+2
            else: # circular segment center
                self.x[j] = self.XC[i]
                self.r[j] = self.type[i]/self.R[i]
                j = j+1
        # getting data from tables and ignoring points with x>L
        self.el = self.table_el['Elevation [m]'].values
        self.el = self.el[self.el < self.L]
        self.bk = self.table_bk['Banking [deg]'].values
        self.bk = self.bk[self.bk < self.L]
        self.gf = self.table_gf['Grip Factor [-]'].values
        self.gf = self.gf[self.gf < self.L]
        self.sc = self.table_sc['Sector'].values
        self.sc = self.sc[self.sc < self.L]
        self.sc[-1] = self.sc[-2]
        # saving coarse position vectors
        self.xx = self.x
        self.xe = self.table_el['Point [m]'].values
        self.xb = self.table_bk['Point [m]'].values
        self.xg = self.table_gf['Start Point [m]'].values
        self.xs = self.table_sc['Start Point [m]'].values
        # saving coarse topology
    # HUD
    print('Pro-processing completed')

    ## Meshing

    # new fine position vector
    if np.floor(self.L)<self.L: # check for injecting last point
        self.x = [np.arange(0, self.mesh_size, np.floor(self.L)), self.L]
    else:
        self.x = np.arange(0, self.mesh_size, np.floor(self.L))
    # distance step vector
    self.dx = np.diff(self.x)
    self.dx = [self.dx, self.dx[-1]]
    # number of mesh points
    self.n = len(self.x)
    # fine curvature vector
    self.r = interpolate.PchipInterpolator(self.xx, self.r, self.x, True)
    # elevation
    self.Z = interpolate.interp1d(self.xe, self.el, self.x, 'linear', 'extrap')
    # banking
    self.bank = interpolate.interp1d(self.xb, self.bk, self.x, 'linear', 'extrap')
    # inclination
    self.incl = -1*np.arctan(np.diff(self.Z)/np.diff(self.x))
    self.incl = [self.incl, self.incl[-1]]
    # grip factor
    self.factor_grip = interpolate.interp1d(self.xg, self.gf, self.x, 'linear', 'extrap')
    # sector
    self.sector = interpolate.interp1d(self.xs, self.sc, self.x, 'previous', 'extrap')
    # HUD
    print('Fine meshing completed with mesh size: ' + self.mesh_size + '[m]')

    ## Map generation

    # coordinate vector preallocation
    self.X = np.zeros((self.n, 1))
    self.Y = np.zeros((self.n, 1))
    # segment angles
    self.angle_seg = np.cumsum(self.r*self.dx)
    # heading angles
    self.angle_head = np.cumsum(self.angle_seg)
    if self.config == 'Closed': # tangency correction for closed track
        self.dh = [np.mod(self.angle_head[-1], np.sign(self.angle_head[-1])*360), self.angle_head[-1]-np.sign(self.angle_head[-1]*360)]
        self.idx = min(np.abs(self.dh))
        self.dh = self.dh[self.idx]
        self.angle_head = self.angle_head-self.x/self.L*self.dh
        self.angle_seg = [self.angle_head[0], np.diff(self.angle_head)]
    self.angle_head = self.angle_head-self.angle_head[0]
    # map generation
    for i in range(2, self.n):
        # previous point
        p = [self.X[i-1], self.Y[i-1], 0]
        # next point
        xyz = Rotation.from_matrix(self.angle_head[i-1]) * [self.dx[i-1], 0, 0] + p
        # saving point coordinates of next point
        self.X[i] = xyz[0]
        self.Y[i] = xyz[1]
    ## Apexes

    # finding Apexes
    apex = signal.find_peaks(np.abs(self.r))
    # correcting corner type
    self.r_apex = self.r[apex]
    # HUD
    print('Apex calculation completed')

    ## Map edit
    # track direction
    if self.direction == 'Backward':
        self.x = self.x[-1]-np.flipud(self.x)
        self.r = -1*np.flipud(self.r)
        self.apex = len(self.x)-np.flipud(self.apex)
        self.r_apex = -1*np.flipud(self.r_apex)
        self.incl = -1*np.flipud(self.incl)
        self.bank = -1*np.flipud(self.bank)
        self.factor_frip = np.flipud(self.factor_grip)
        self.sector = np.flipud(self.sector)
        self.X = np.flipud(self.X)
        self.Y = np.flipud(self.Y)
        self.Z = np.flipud(self.Z)
    
    # track rotation
    # rotating track map
    xyz = Rotation.from_matrix(self.rotation) * [self.X, self.Y, self.Z]
    self.X = xyz[0, :]
    self.Y = xyz[1, :]
    self.Z = xyz[2, :]
    # HUD
    print('Track rotated')

    # closing map if necessary
    if self.config == 'Closed': # closed track
        # HUD
        print('Closing fine mesh map')
        # linear correction vectors
        self.DX = self.x/self.L * (self.X[0]-self.X[-1])
        self.DY = self.x/self.L * (self.Y[0]-self.Y[-1])
        self.DZ = self.x/self.L * (self.Z[0]-self.Z[-1])
        self.db = self.x/self.L * (self.bank[0]-self.bank[-1])
        # adding correction
        self.X = self.X + self.DX
        self.Y = self.Y + self.DY
        self.Z = self.Z + self.DZ
        self.bank = self.bank + self.db
        # recalculating inclination
        self.incl = -1*np.arctan(np.diff(self.Z)/np.diff(self.x))
        self.incl = [self.incl, self.incl[-2]+self.incl[0]/2]
        # HUD
        print('Fine mesh map closed')
    # smooth track inclination
    self.incl = savgol_filter(self.incl)
    # HUD
    print('Fine mesh map created')

    ## Plotting Results
    ## TODO
    # Add plot stuff from lines 515-622

    ## Saving circuit

    # saving
    ## TODO
    # add lines 626-628

    ## ASCII map
    self.charh = 15 # font height [pixels]
    self.charw = 8 # font width [pixels]
    self.linew = 66 # log file character width
    self.mapw = max(self.X)-min(self.X) # map width
    self.YY = np.round(self.Y/(self.charh/self.charw)/self.mapw*self.linew) # scales y values
    self.XX = np.round(self.X/self.mapw*self.linew) # scales x values
    self.YY = -1* self.YY - min(self.YY * -1) # flipping y and shifting to positive space
    self.XX = self.XX - min(self.XX) # shifting x to positive space
    self.p = np.unique([self.XX, self.YY], 'rows') # getting unique points
    self.XX = self.p[:, 0]+1 # saving x
    self.YY = self.p[:, 1]+1 # saving y
    self.maph = max(self.YY) # getting new map height [lines]
    self.mapw = max(self.XX) # getting new map width [columns]
    self.map = np.chararray(self.maph, self.mapw) # preallocating map
    # looping through characters
    for i in range(0, self.maph):
        for j in range(0, self.mapw):
            self.check = [self.XX, self.YY] == [j, i] # checking if pixel is on
            self.check = self.check[:, 0] * self.check[:, 1] # combining truth table
            if max(self.check):
                self.map[i, j] = 'o' # turning pixel on
            else:
                self.map[i, j] = ' ' # turning pixel off
    print('Map: ')
    print(self.map)

## Functions
## Line 665 to end
def read_info(self, workbookFile, sheetName=1, startRow=1, endRow=7):
    # Input handling
    # If no sheet is specified, read first sheet
    data = pd.read_excel(workbookFile, 'Info', header=None)
    # Convert to output type
    self.name = data.at[0, 1]
    self.country = data.at[1, 1]
    self.city = data.at[2, 1]
    self.type = data.at[3, 1]
    self.config = data.at[4, 1]
    self.direction = data.at[5, 1]
    self.mirror = data.at[6, 1]
        
def read_shape_data(self, workbookFile, sheetName=1, startRow=2, endRow=10000):
    # Input handling
    return pd.read_excel(workbookFile, 'Shape', header=0)
            
def read_data(self, workbookFile, sheetName=1, startRow=2, endRow=10000):
    # Input Handling
    # If no sheet is specified, read first sheet
    # If row start and end points are not specified, define defaults
    # Setup the Import Options
    return pd.read_excel(workbookFile, sheetName, header=0)
            
def read_logged_data(self, filename, header_startRow=1, header_endRow=12, data_startRow=14, data_endRow=np.inf):
    delimiter = ','
    fileID = open(filename, 'r')

    # Header array
    # Read columns of data according to the format
    headerArray = np.loadtxt(fileID, delimiter=delimiter)
    for block in range(2, len(header_startRow)):
        fileID.seek(0)
        dataArrayBlock = np.loadtxt(fileID, header_endRow(block)-header_startRow(block)+1, delimiter=delimiter)
        for col in range(0,len(headerArray)):
            headerArray[col] = [headerArray[col], dataArrayBlock[col]]
    # Create output variable
    header = headerArray[1, -2]

    # Data array
    # Pointer to start of file
    fileID.seek(0)
    # Format for each line of text
    data_formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]'
    # Read columns of data according to the format
    dataArray = np.loadtxt(fileID, data_formatSpec, data_endRow[0]-data_startRow[0]+1, delimiter=delimiter)
    for block in range(2, len(data_startRow)):
        fileID.seek(0)
        dataArrayBlock = np.loadtxt(fileID, data_formatSpec, data_endRow(block)-data_startRow(block)+1, delimiter=delimiter)
        for col in range(0,len(dataArray)):
            dataArray[col] = [dataArray[col], dataArrayBlock[col]]
    # Create output variable
    data = dataArray[1, -2]

    # Close the text file
    fileID.close()