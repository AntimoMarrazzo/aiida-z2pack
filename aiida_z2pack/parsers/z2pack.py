# -*- coding: utf-8 -*-
import json
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.plugins import DataFactory, CalculationFactory

Dict              = DataFactory('dict')
Z2packCalculation = CalculationFactory('z2pack.z2pack')


class Z2packParser(Parser):
    """
    Z2packParser
    """    
    def parse(self, **kwargs):
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # Checks for error output files
        retrieved_names = out_folder.list_object_names()
        if not Z2packCalculation._OUTPUT_Z2PACK_FILE in retrieved_names:
            return self.exit_codes.ERROR_UNEXPECTED_FAILURE
        if Z2packCalculation._ERROR_W90_FILE in retrieved_names:
            return self.exit_codes.ERROR_W90_CRASH
        if Z2packCalculation._ERROR_PW_FILE in retrieved_names:
            return self.exit_codes.ERROR_PW0_CRASH
        if not Z2packCalculation._OUTPUT_RESULT_FILE in retrieved_names:
            return self.exit_codes.ERROR_MISSING_RESULTS_FILE

        with out_folder.open(Z2packCalculation._OUTPUT_RESULT_FILE) as f:
            data = json.load(f)
        with out_folder.open(Z2packCalculation._OUTPUT_Z2PACK_FILE) as f:
            out_file = f.readlines()

        gap_f   = len(data['convergence_report']['GapCheck']['FAILED'])
        move_f  = len(data['convergence_report']['MoveCheck']['FAILED'])        
        pos_f   = len(data['convergence_report']['PosCheck']['FAILED'])
        pos_m   = len(data['convergence_report']['PosCheck']['MISSING'])
        success = not any([gap_f, move_f, pos_f, pos_m])        
        
        data['Tests_passed'] = success
        

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




