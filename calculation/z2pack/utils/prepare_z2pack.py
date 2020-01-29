from aiida.common import exceptions
from aiida_quantumespresso.calculations import _uppercase_dict

def prepare_z2pack(self, folder):
    input_filename = folder.get_abs_path(self._DEFAULT_INPUT_FILE)
    try:
        nscf_code = self.inputs.nscf_code
    except KeyError:
        raise exceptions.InputValidationError("No nscf code specified for this calculation")
    try:
        overlap_code = self.inputs.overlap_code
    except KeyError:
        raise exceptions.InputValidationError("No overlap code specified for this calculation")
    try:
        wannier90_code = self.inputs.wannier90_code
    except KeyError:
        raise exceptions.InputValidationError("No Wannier90 code specified for this calculation")
    

    if 'settings' in self.inputs:
        settings_dict = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        settings_dict = self.inputs.settings.get_dict()
    else:
        raise exceptions.InputValidationError("No settings code specified for this calculation")          

    try:
        mpi_command = settings_dict['mpi_command']
    except KeyError:
        raise exceptions.InputValidationError("No mpi_command code specified for this calculation")

    if not 'npools' in settings_dict:
        pools_cmd = ''
    else:
        npools = settings_dict['npools']
        if isinstance(npools, int):
            pools_cmd = ' -nk ' + str(npools) + ' '
        else:
            raise exceptions.InputValidationError("npools must be an integer.")


    try:
        dim_mode = settings_dict['dimension_mode']
    except KeyError:
        raise exceptions.InputValidationError("No dimension_mode specified for this calculation")

    try:
        invariant = settings_dict['invariant']
    except KeyError:
        raise exceptions.InputValidationError("No invariant specified for this calculation")

    pos_tol            = settings_dict.get('pos_tol', self._DEFAULT_POS_TOLERANCE)
    gap_tol            = settings_dict.get('gap_tol', self._DEFAULT_GAP_TOLERANCE)
    move_tol           = settings_dict.get('move_tol', self._DEFAULT_MOVE_TOLERANCE)
    num_lines          = settings_dict.get('num_lines', self._DEFAULT_NUM_LINES)
    min_neighbour_dist = settings_dict.get('min_neighbour_dist', self._DEFAULT_MIN_NEIGHBOUR_DISTANCE)
    iterator           = settings_dict.get('iterator', self._DEFAULT_ITERATOR)
    restart_mode       = settings_dict.get('restart_mode', False)
    prepend_code       = settings_dict.get('prepend_code', '')



    if dim_mode == '3D':
        try:
            surface = settings_dict.get_dict()['surface']
        except KeyError:
            raise exceptions.InputValidationError("A surface must be specified for dim_mode==3D ")
    
    input_file_lines=[]
    input_file_lines.append('#!/usr/bin/env python3')
    input_file_lines.append('import sys')
    #input_file_lines.append('sys.path.append('+"'"+'./Z2pack'+"'"+')')
    input_file_lines.append('import z2pack')
    input_file_lines.append('import os')
    input_file_lines.append('import shutil')
    input_file_lines.append('import subprocess')
    input_file_lines.append('import xml.etree.ElementTree as ET')
    input_file_lines.append('import json')

    nscf_cmd      = ' {} {}'.format(mpi_command, nscf_code.get_execname())
    overlap_cmd   = ' {} {}'.format(mpi_command, overlap_code.get_execname())
    wannier90_cmd = ' {} {}'.format(mpi_command, wannier90_code.get_execname())
    
    z2cmd = ("('"+wannier90_cmd + ' aiida '+' -pp;' +"'"+  "+"+'\n'+
             "'"+nscf_cmd + pools_cmd + '< ' +self._INPUT_NSCF_FILE_NAME + '>& pw.log;' +"'"+ "+" +'\n'+
             "'"+overlap_cmd + '< ' + self._INPUT_OVERLAP_FILE_NAME+' >& pw2wan.log;' + "')")

    input_file_lines.append('z2cmd =' +  z2cmd)
    input_files = [ self._INPUT_NSCF_FILE_NAME, self._INPUT_OVERLAP_FILE_NAME,self._INPUT_W90_FILE_NAME]
    input_file_lines.append('input_files='+str(input_files))
    input_file_lines.append('system=z2pack.fp.System(input_files=input_files,')
    input_file_lines.append('\t kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90_full],')
    # input_file_lines.append('\t kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90],')
    input_file_lines.append('\t'+'kpt_path='+str([ self._INPUT_NSCF_FILE_NAME, self._INPUT_W90_FILE_NAME])+',')
    input_file_lines.append('\t command=z2cmd'+',')
    input_file_lines.append("\t executable='/bin/bash'"+',')
    input_file_lines.append("\t mmn_path='aiida.mmn')")            
    input_file_lines.append("gap_check={}")
    input_file_lines.append("move_check={}")
    input_file_lines.append("pos_check={}")
    input_file_lines.append("res_dict={'convergence_report':{'GapCheck':{},"
                                                             "'MoveCheck':{},"
                                                             "'PosCheck':{}"
                                                             "}"
                                        ",'invariant':{}}")
    if(prepend_code!=''):
        input_file_lines.append("\t"+prepend_code)
    if (dim_mode=='2D' or dim_mode=='3D'):       
        input_file_lines.append('result = z2pack.surface.run(')
        input_file_lines.append('\t system=system,')
        if(dim_mode=='2D'):
            if (invariant=='Z2'):
                input_file_lines.append('\t surface=lambda t1,t2: [t1/2, t2, 0],')
            elif (invariant=='Chern'):
                input_file_lines.append('\t surface=lambda t1,t2: [t1, t2, 0],')
        elif(dim_mode=='3D'):
            input_file_lines.append('\t surface='+surface+',')
        input_file_lines.append('\t pos_tol='+str(pos_tol)+',')
        input_file_lines.append('\t gap_tol='+str(gap_tol)+',')
        input_file_lines.append('\t move_tol='+str(move_tol)+',')
        input_file_lines.append('\t num_lines='+str(num_lines)+',')
        input_file_lines.append('\t min_neighbour_dist='+ str(min_neighbour_dist)+',')
        input_file_lines.append('\t iterator='+ str(iterator)+',')
        input_file_lines.append('\t save_file='+"'"+self._DEFAULT_OUTPUT_Z2PACK +"'"+',')
        if restart_mode:
            input_file_lines.append('\t load=True')
        input_file_lines.append('\t )')
        if (invariant=='Z2'):
            input_file_lines.append('Z2=z2pack.invariant.z2(result)')
            input_file_lines.append("res_dict['invariant'].update({'Z2':Z2})")
        elif (invariant=='Chern'):
            input_file_lines.append('Chern=z2pack.invariant.chern(result)')
            input_file_lines.append("res_dict['invariant'].update({'Chern':Chern})")
    else:
        raise TypeError("Only dimension_mode 2D and 3D"
                    "are currently implemented.")
    
    input_file_lines.append("gap_check['PASSED']="
                            "result.convergence_report['surface']['GapCheck']['PASSED']")
    input_file_lines.append("gap_check['FAILED']="
                            "result.convergence_report['surface']['GapCheck']['FAILED']")
    input_file_lines.append("move_check['PASSED']="
                            "result.convergence_report['surface']['MoveCheck']['PASSED']")
    input_file_lines.append("move_check['FAILED']="
                            "result.convergence_report['surface']['MoveCheck']['FAILED']")
    input_file_lines.append("pos_check['PASSED']="
                            "result.convergence_report['line']['PosCheck']['PASSED']")
    input_file_lines.append("pos_check['FAILED']="
                            "result.convergence_report['line']['PosCheck']['FAILED']")
    input_file_lines.append("pos_check['MISSING']="
                            "result.convergence_report['line']['PosCheck']['MISSING']")

    input_file_lines.append("res_dict['convergence_report']['GapCheck'].update(gap_check)")
    input_file_lines.append("res_dict['convergence_report']['MoveCheck'].update(move_check)")
    input_file_lines.append("res_dict['convergence_report']['PosCheck'].update(pos_check)")  
    
    input_file_lines.append("with open('results.json', 'w') as fp:")
    input_file_lines.append("\t json.dump(res_dict, fp)")
    with open(input_filename, 'w') as file_input:
        file_input.write( "\n".join(input_file_lines) )
        file_input.write( "\n" )