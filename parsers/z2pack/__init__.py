# -*- coding: utf-8 -*-
from aiida.orm.calculation.job.z2pack import Z2packCalculation
from aiida.parsers.parser import Parser
#from aiida.common.datastructures import calc_states
from aiida.parsers.exceptions import OutputParsingError
import json
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.bands import BandsData
from aiida.orm.data.array.kpoints import KpointsData
from __builtin__ import True

__authors__ = "Antimo Marrazzo and The AiiDA team."
__copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/.. All rights reserved."
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file"
__version__ = "0.6.0"

class Z2packParser(Parser):
    """
    Z2packParser
    """    
    _outarray_name = 'output_data'
    

    def __init__(self,calculation):
        """
        Initialize the instance of Z2pack parser
        """
        # check for valid input
        if not isinstance(calculation,Z2packCalculation):
            raise OutputParsingError("Input must calc must be a "
                                     "Z2packCalculation")
        super(Z2packParser, self).__init__(calculation)
          
            
    def parse_with_retrieved(self, retrieved):
        """
        Parses the datafolder, stores results.
        """
        successful = True
        new_nodes_list = []
        # select the folder object
        # Check that the retrieved folder is there 
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error("No retrieved folder found")
            successful = False
            return successful, new_nodes_list

        # Checks for error output files
        if self._calc._ERROR_FILE_NAME in out_folder.get_folder_list():
            self.logger.error('Errors were found please check the retrieved '
                              '{} file'.format(self._calc._ERROR_FILE_NAME))
            successful = False
            return successful, new_nodes_list

        try:
            filpath = out_folder.get_abs_path(
                self._calc._DEFAULT_OUTPUT_RESULTS_Z2PACK )
            with open(filpath, 'r') as fil:
                    #out_file = fil.readlines()
                    data = json.load(fil)
        except OSError:
            self.logger.error("Standard output file could not be found.")
            successful = False
            return successful, new_nodes_list

        gap_f = len(data['convergence_report']['GapCheck']['FAILED'])
        move_f = len(data['convergence_report']['MoveCheck']['FAILED'])        
        pos_f = len(data['convergence_report']['PosCheck']['FAILED'])
        pos_m = len(data['convergence_report']['PosCheck']['MISSING'])
        if gap_f==0 and move_f==0 and pos_f==0 and pos_m==0:
            success = True
        else:
            success = False
        
        
        data['Tests_passed'] = success
        
        try:
            filpath = out_folder.get_abs_path(
                self._calc._DEFAULT_OUTPUT_FILE)
            with open(filpath, 'r') as fil:
                    out_file = fil.readlines()
        except OSError:
            self.logger.error("Standard output file {} could not be found.".format(self._calc._DEFAULT_OUTPUT_FILE))
            successful = False
            return successful, new_nodes_list
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

        # save the arrays
        output_data = ParameterData(dict=data)
        linkname = 'output_parameters'
        new_nodes_list += [(linkname,output_data)]
        
        return successful,new_nodes_list





