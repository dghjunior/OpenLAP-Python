# Import dependencies

# vehicle class declaration
class OpenTRACK:
    # constructor
    def __init__(self, csv_filename):
        self.name = ''
        self.n = 0
        self.r = 0
        self.incl = 0
        self.bank = 0
        self.factor_grip = 0
        self.config = ''
        sector = 0