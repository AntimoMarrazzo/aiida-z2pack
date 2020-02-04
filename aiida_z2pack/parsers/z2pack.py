# -*- coding: utf-8 -*-
import json
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory, CalculationFactory

Dict              = DataFactory('dict')
KpointsData       = DataFactory('array.kpoints')
BandsData         = DataFactory('array.bands')
Z2packCalculation = CalculationFactory('z2pack')


class Z2packParser(Parser):
    """
    Z2packParser
    """    
    _outarray_name = 'output_data'
    

    def __init__(self, calculation):
        """
        Initialize the instance of Z2pack parser
        """
        # check for valid input
        if not isinstance(calculation, Z2packCalculation):
            raise exceptions.OutputParsingError(
                "Input must calc must be a Z2packCalculation")
        super(Z2packParser, self).__init__(calculation)

    def parse(self, **kwargs):
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_FOLDER)

        # Checks for error output files
        if self.node._ERROR_W90_FILE in out_folder.list_object_names():
            return self.exit(self.exit_codes.ERROR_UNEXPECTED_FAILURE)

        try:
            filpath = out_folder.get_abs_path(
                self.node._OUTPUT_RESULTS_FILE)
            with open(filpath, 'r') as fil:
                    data = json.load(fil)
        except OSError:
            return self.exit(self.exit_codes.ERROR_MISSING_RESULTS_FILE)

        gap_f   = len(data['convergence_report']['GapCheck']['FAILED'])
        move_f  = len(data['convergence_report']['MoveCheck']['FAILED'])        
        pos_f   = len(data['convergence_report']['PosCheck']['FAILED'])
        pos_m   = len(data['convergence_report']['PosCheck']['MISSING'])
        success = not any([gap_f, move_f, pos_f, pos_m])        
        
        data['Tests_passed'] = success
        
        try:
            filpath = out_folder.get_abs_path(self.node._OUTPUT_Z2PACK_FILE)
            with open(filpath, 'r') as fil:
                    out_file = fil.readlines()
        except OSError:
            return self.exit(self.exit_codes.ERROR_MISSING_Z2PACK_OUTFILE)

        #out_file = out_file.split("\n")
        out_file = [i.strip('\n') for i in out_file]

        time = [i for i in out_file if 'Calculation finished' in i][0].split()[4:7]
        wall_time_seconds = int(time[0].strip('h')) * 3600 + \
                        int(time[1].strip('m')) * 60 + \
                        int(time[2].strip('s'))

        z2pack_version = [i for i in out_file if "running Z2Pack version" in i][0].split()[3]
        data['wall_time_seconds'] =  wall_time_seconds
        data['z2pack_version'] =  z2pack_version 

        self.out('output_parameters', Dict(dict=data))




