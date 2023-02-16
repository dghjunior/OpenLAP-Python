# Import dependencies
import numpy as np
import os
import datetime
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from scipy import signal
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

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
        indices = np.where(self.l == 0)
        self.R = np.delete(self.R, indices)
        self.l = np.delete(self.l, indices)
        self.type = np.delete(self.type, indices)
        # injecting points at long corners
        self.angle_seg = np.rad2deg(self.l/self.R)
        j = 0
        self.RR = self.R
        self.ll = self.l
        self.tt = self.type
        for i in range(0, len(self.l)):
            if self.angle_seg[i] > self.kappa:
                l_inj = min([self.ll[j]/3, self.kappa*self.R[i]])
                self.ll = [self.ll[0:j-1], l_inj, self.ll[j]*l_inj, l_inj, self.ll[j+1:]]
                self.RR = [self.RR[0:j-1], self.RR[j], self.RR[j], self.RR[j], self.RR[j+1:]]
                self.tt = [self.tt[0:j-1], self.tt[j], self.tt[j], self.tt[j], self.tt[j+1:]]
                j = j+3
            else:
                j = j+1
        self.R = self.RR
        self.l = self.ll
        self.type = self.tt
        # replacing consecutive straights
        for i in range(0, len(self.l)-1):
            j = 1
            while True:
                if self.type[i+j] == 0 and self.type[i] == 0 and self.l[i] != -1:
                    self.l[i] = self.l[i]+self.l[i+j]
                    self.l[i+j] = -1
                else:
                    break
                j = j+1
        indices = np.where(self.l == -1)
        self.R = np.delete(self.R, indices)
        self.l = np.delete(self.l, indices)
        self.type = np.delete(self.type, indices)
        # final segment point calculation
        self.X = np.cumsum(self.l) # end position of each segment
        self.XC = np.cumsum(self.l)-self.l/2 # center position of each segment
        j = 0 # index
        self.x = np.zeros((len(self.X)+sum(self.R==np.inf), 1)) # preallocation
        self.r = np.zeros((len(self.X)+sum(self.R==np.inf), 1)).tolist() # preallocation
        for i in range(0, len(self.X)):
            if self.R[i] == np.inf: # end of straight point injection
                self.x[j] = self.X[i]-self.l[i]
                self.r[j] = 0.0
                self.r[j+1] = 0.0
                self.x[j+1] = self.X[i]
                j = j+2
            else: # circular segment center
                self.x[j] = self.XC[i]
                self.r[j] = self.type[i]/self.R[i]
                j = j+1
        # getting data from tables and ignoring points with x>L
        self.el = self.table_el[['Point [m]', 'Elevation [m]']].values
        self.el = self.el[~(self.el[:, 0] > self.L),:]
        self.bk = self.table_bk[['Point [m]', 'Banking [deg]']].values
        self.bk = self.bk[~(self.bk[:, 0] > self.L),:]
        self.gf = self.table_gf[['Start Point [m]', 'Grip Factor [-]']].values
        self.gf = self.gf[~(self.gf[:, 0] > self.L),:]
        self.sc = self.table_sc[['Start Point [m]', 'Sector']].values
        self.sc = self.sc[~(self.sc[:, 0] > self.L),:].tolist()
        self.sc.append([self.L, self.sc[-1][1]])
        # saving coarse position vectors
        self.xx = self.x
        self.xx = np.sort(self.xx.T.flatten()).tolist()
        self.xe = self.el[:, 0].tolist()
        self.xb = self.bk[:, 0].tolist()
        self.xg = self.gf[:, 0].tolist()
        self.xs = np.array(self.sc)[:, 0].tolist()
        # saving coarse topology
        self.el = self.el[:, 1].tolist()
        self.bk = self.bk[:, 1].tolist()
        self.gf = self.gf[:, 1].tolist()
        self.sc = np.array(self.sc)[:, 1].tolist()
    # HUD
    print('Pro-processing completed')

    ## Meshing

    # new fine position vector
    self.x = list(np.arange(0, np.floor(self.L)+1, self.mesh_size))
    if np.floor(self.L)<self.L: # check for injecting last point
        self.x.append(self.L)
    # distance step vector
    self.dx = np.diff(self.x).tolist()
    self.dx.append(self.dx[-1])
    # number of mesh points
    self.n = len(self.x)
    # fine curvature vector
    self.r = interpolate.pchip_interpolate(self.xx, self.r, self.x)
    # elevation
    z = interpolate.interp1d(self.xe, self.el, kind='linear', fill_value='extrapolate')
    self.Z = []
    for i in range(0, len(self.x)):
        self.Z.append(z(self.x[i]).tolist())
    # banking
    f = interpolate.interp1d(self.xb, self.bk, kind='linear', fill_value='extrapolate')
    self.bank = []
    for i in range(0, len(self.x)):
        self.bank.append(f(self.x[i]).tolist())
    # inclination
    self.incl = -1*np.rad2deg(np.arctan(np.diff(self.Z)/np.diff(self.x)))
    self.incl = np.append(self.incl, self.incl[-1])
    # grip factor
    f = interpolate.interp1d(self.xg, self.gf, kind='linear', fill_value='extrapolate')
    self.factor_grip = []
    for i in range(0, len(self.x)):
        self.factor_grip.append(f(self.x[i]).tolist())
    # sector
    f = interpolate.interp1d(self.xs, self.sc, kind='previous', fill_value='extrapolate')
    self.sector = []
    for i in range(0, len(self.x)):
        self.sector.append(f(self.x[i]).tolist())
    # HUD
    print('Fine meshing completed with mesh size: ' + str(self.mesh_size) + '[m]')

    ## Map generation

    # coordinate vector preallocation
    self.X = np.zeros((self.n)).tolist()
    self.Y = np.zeros((self.n)).tolist()
    # segment angles
    self.r = [arr.tolist() for arr in self.r.T.flatten()]
    self.angle_seg = np.rad2deg(np.multiply(self.dx,self.r)).tolist()
    # heading angles
    self.angle_head = np.cumsum(self.angle_seg).tolist()
    if self.config == 'Closed': # tangency correction for closed track
        self.dh = [
            np.mod(self.angle_head[-1], np.sign(self.angle_head[-1])*360),
            self.angle_head[-1]-np.sign(self.angle_head[-1])*360
        ]
        self.idx = self.dh.index(-1*(min(np.abs(self.dh))))
        self.dh = self.dh[self.idx-1]
        self.angle_head = self.angle_head-self.x/self.L*self.dh
        self.angle_seg = np.insert(np.diff(self.angle_head), 0, self.angle_seg[0])
    self.angle_head = self.angle_head-self.angle_head[0]
    # map generation
    for i in range(1, self.n):
        # previous point
        p = np.matrix([[self.X[i-1]], [self.Y[i-1]], [0.0]])
        # next point
        rotz = Rotation.from_euler('z', self.angle_head[i-1], degrees=True).as_matrix().astype(np.float64)
        temp = np.matrix([[self.dx[i-1]], [0.0], [0.0]])
        xyz = np.add(np.matmul(rotz, temp), p)
        # saving point coordinates of next point
        self.X[i] = np.around(xyz.item((0, 0)), 6)
        self.Y[i] = xyz.item((1, 0))
    ## Apexes

    # finding Apexes
    apex = signal.find_peaks(np.abs(self.r))[0] + 1
    # correcting corner type
    self.r_apex = [self.r[i-1] for i in apex]
    self.X = np.array(self.X)
    self.Y = np.array(self.Y)
    self.Z = np.array(self.Z)

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
    rotz = Rotation.from_euler('z', self.rotation, degrees=True).as_matrix().astype(int)
    primes = np.array([self.X.conj(), self.Y.conj(), self.Z.conj()])
    xyz = np.dot(rotz, primes)
    self.X = xyz[0, :].conj()
    self.Y = xyz[1, :].conj()
    self.Z = xyz[2, :].conj()
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
        self.incl = -1*np.rad2deg(np.arctan(np.diff(self.Z)/np.diff(self.x)))
        self.incl = np.append(self.incl, self.incl[-2]+self.incl[0]/2)
        # HUD
        print('Fine mesh map closed')
    # smooth track inclination
    self.incl = savgol_filter(self.incl, int(self.n*0.11), 1)
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
    self.p = np.unique([self.XX, self.YY], axis=1) # getting unique points
    self.XX = (self.p[:][0]+1).astype(int) # saving x
    self.YY = (self.p[:][1]+1).astype(int) # saving y
    self.maph = max(self.YY) # getting new map height [lines]
    self.mapw = max(self.XX) # getting new map width [columns]
    self.map = [[' ' for x  in range(self.mapw)] for y in range(self.maph)] # preallocating map
    # looping through characters
    for i in range(0, self.maph+1):
        for j in range(0, self.mapw):
            # create array of all rows of [self.XX, self.YY] that match [j, i]
            self.check = np.array([self.XX, self.YY]).T == [j, i]
            self.check = self.check[:,0] * self.check[:,1] # combining truth table
            if max(self.check):
                self.map[i-1][j-1] = 'o' # turning pixel on
    print('Map: ')
    print(*(''.join(row) for row in self.map), sep='\n')

## Functions
## Line 665 to end
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def read_info(self, workbookFile, sheetName=1, startRow=1, endRow=7):
    # Input handling
    # If no sheet is specified, read first sheet
    data = pd.read_excel(workbookFile, 'Info', header=None)
    info = {
        'Name': data.at[0, 1],
        'Country': data.at[1, 1],
        'City': data.at[2, 1],
        'Type': data.at[3, 1],
        'Config': data.at[4, 1],
        'Direction': data.at[5, 1],
        'Mirror': data.at[6, 1]
    }
    # Convert to output type
    self.name = data.at[0, 1]
    self.country = data.at[1, 1]
    self.city = data.at[2, 1]
    self.type = data.at[3, 1]
    self.config = data.at[4, 1]
    self.direction = data.at[5, 1]
    self.mirror = data.at[6, 1]
    return info
        
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

# tr = OpenTRACK('Spa-Francorchamps.xlsx')