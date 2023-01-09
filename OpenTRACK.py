# Import dependencies

# vehicle class declaration
class OpenTRACK:
    # constructor
    def __init__(self, csv_filename='Spa-Francorchamps.xlsx'):
        self.filename = csv_filename
        
        ## Mode selection

        # mode = 'logged data'
        mode = 'shape data'
        # log_mode = 'speed & yaw'
        log_mode = 'speed & latacc'

        ## Settings

        # meshing
        mesh_size = 1 # [m]
        #filtering for logged data mode
        filter_dt = 0.1 # [s]
        # track map rotation angle
        rotation = 0 # [deg]
        # track map shape adjuster
        lambda = 1 # [-] CHANGE THIS VARIABLE NAME
        # long corner adjuster
        kappa = 1000 # [deg]

        ## Reading file

        # HUD
        print('Reading track file: ' + self.filename)
        if mode == 'logged data':
            ## from logged data

            head, data = read_logged_data(self.filename)
            self.name = head[1, 1]
            self.country = head[3, 2]
            self.city = head
        ## STOPPED ON LINE 102 OF OPENTRACK.m
        