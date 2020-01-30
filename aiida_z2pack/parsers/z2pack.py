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
        successful = True
        new_nodes_list = []

        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            self.logger.error("No retrieved folder found")
            successful = False
            return
            # return successful, new_nodes_list
            # return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_FOLDER)


        # Checks for error output files
        if self.node._ERROR_FILE_NAME in out_folder.list_object_names():
            self.logger.error('Errors were found please check the retrieved '
                              '{} file'.format(self.node._ERROR_FILE_NAME))
            return
            successful = False
            # return successful, new_nodes_list

        try:
            filpath = out_folder.get_abs_path(
                self.node._DEFAULT_OUTPUT_RESULTS_Z2PACK )
            with open(filpath, 'r') as fil:
                    #out_file = fil.readlines()
                    data = json.load(fil)
        except OSError:
            self.logger.error("Standard output file could not be found.")
            successful = False
            return successful, new_nodes_list

        gap_f   = len(data['convergence_report']['GapCheck']['FAILED'])
        move_f  = len(data['convergence_report']['MoveCheck']['FAILED'])        
        pos_f   = len(data['convergence_report']['PosCheck']['FAILED'])
        pos_m   = len(data['convergence_report']['PosCheck']['MISSING'])
        success = not any([gap_f, move_f, pos_f, pos_m])        
        
        data['Tests_passed'] = success
        
        try:
            filpath = out_folder.get_abs_path( self.node._DEFAULT_OUTPUT_FILE)
            with open(filpath, 'r') as fil:
                    out_file = fil.readlines()
        except OSError:
            self.logger.error("Standard output file {} could not be found.".format(self.node._DEFAULT_OUTPUT_FILE))
            successful = False
            return
            # return successful, new_nodes_list
        #out_file = out_file.split("\n")
        out_file = [i.strip('\n') for i in out_file]
        if successful:
            time = [i for i in out_file if 'Calculation finished' in i][0].split()[4:7]
            wall_time_seconds = int(time[0].strip('h')) * 3600 + \
                            int(time[1].strip('m')) * 60 + \
                            int(time[2].strip('s'))
        else:
            wall_time_seconds = None 
        z2pack_version = [i for i in out_file if "running Z2Pack version" in i][0].split()[3]
        data['wall_time_seconds'] =  wall_time_seconds
        data['z2pack_version'] =  z2pack_version 

        self.out('output_parameters', Dict(dict=data))
        # save the arrays
        # output_data = ParameterData(dict=data)
        # linkname = 'output_parameters'
        # new_nodes_list += [(linkname,output_data)]
        
        # return successful,new_nodes_list





