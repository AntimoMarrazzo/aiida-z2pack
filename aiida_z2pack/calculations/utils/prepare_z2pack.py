from aiida.common import exceptions
from aiida_quantumespresso.calculations import _lowercase_dict

def prepare_z2pack(cls, folder):
    input_filename = folder.get_abs_path(cls._INPUT_Z2PACK_FILE)
    try:
        pw_code = cls.inputs.pw_code
    except AttributeError:
        raise exceptions.InputValidationError("No nscf code specified for this calculation")
    try:
        overlap_code = cls.inputs.overlap_code
    except AttributeError:
        raise exceptions.InputValidationError("No overlap code specified for this calculation")
    try:
        wannier90_code = cls.inputs.wannier90_code
    except AttributeError:
        raise exceptions.InputValidationError("No Wannier90 code specified for this calculation")
    
    try:
        settings_dict = _lowercase_dict(cls.inputs.z2pack_settings.get_dict(), dict_name='z2pack_settings')
    except:
        raise exceptions.InputValidationError("No settings specified for this calculation")          

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

    computer           = cls.inputs.pw_code.computer
    proc_per_machine   = computer.get_default_mpiprocs_per_machine()
    n_machines         = cls.inputs.metadata.options.resources['num_machines']
    mpi_procs          = proc_per_machine * n_machines
    mpi_command        = computer.get_mpirun_command()
    mpi_command        = ' '.join(mpi_command).format(tot_num_mpiprocs=mpi_procs)

    pos_tol            = settings_dict.get('pos_tol', cls._DEFAULT_POS_TOLERANCE)
    gap_tol            = settings_dict.get('gap_tol', cls._DEFAULT_GAP_TOLERANCE)
    move_tol           = settings_dict.get('move_tol', cls._DEFAULT_MOVE_TOLERANCE)
    num_lines          = settings_dict.get('num_lines', cls._DEFAULT_NUM_LINES)
    min_neighbour_dist = settings_dict.get('min_neighbour_dist', cls._DEFAULT_MIN_NEIGHBOUR_DISTANCE)
    iterator           = settings_dict.get('iterator', cls._DEFAULT_ITERATOR)
    # restart_mode       = settings_dict.get('restart_mode', False)
    prepend_code       = settings_dict.get('prepend_code', '')



    if dim_mode == '3D':
        try:
            surface = settings_dict['surface']
            # surface = settings_dict.get_dict()['surface']
        except KeyError:
            raise exceptions.InputValidationError("A surface must be specified for dim_mode==3D ")
    
    input_file_lines=[]
    input_file_lines.append('#!/usr/bin/env python')
    input_file_lines.append('import z2pack')
    input_file_lines.append('import json')

    nscf_cmd      = ' {} {}'.format(mpi_command, pw_code.get_execname())
    overlap_cmd   = ' {} {}'.format(mpi_command, overlap_code.get_execname())
    wannier90_cmd = ' {}'.format(wannier90_code.get_execname())
    
    z2cmd = (
        "(\n    '" +
        "ln -s ../out .; ln -s ../pseudo .;'\n    '" +
        wannier90_cmd + ' ' + cls._SEEDNAME + ' -pp;' + "' +\n    '" +
        nscf_cmd + pools_cmd + ' < ' + cls._INPUT_PW_NSCF_FILE + ' >& ' + cls._OUTPUT_PW_NSCF_FILE + ";' +\n    '" +
        overlap_cmd + ' < ' + cls._INPUT_OVERLAP_FILE + '  >& ' + cls._OUTPUT_OVERLAP_FILE + ";'\n" +
        ")"
        # yapf: disable
        )

    # yapf: disable
    input_file_lines.append("")
    input_file_lines.append('z2cmd =' +  z2cmd)

    input_file_lines.append("")
    input_files = [cls._INPUT_PW_NSCF_FILE, cls._INPUT_OVERLAP_FILE,cls._INPUT_W90_FILE]
    input_file_lines.append('input_files = ' + str(input_files))
    input_file_lines.append('system = z2pack.fp.System(')
    input_file_lines.append('    input_files = input_files,')
    input_file_lines.append('    kpt_fct     = [z2pack.fp.kpoint.qe_explicit, z2pack.fp.kpoint.wannier90_full],')
    # input_file_lines.append('    build_folder= \'.\',')
    # input_file_lines.append('\t kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90],')
    input_file_lines.append('    kpt_path    = ' + str([cls._INPUT_PW_NSCF_FILE, cls._INPUT_W90_FILE]) + ',')
    input_file_lines.append('    command     = z2cmd,')
    input_file_lines.append("    executable  = '/bin/bash',")
    input_file_lines.append("    mmn_path    = '{}.mmn'".format(cls._SEEDNAME))
    input_file_lines.append(')')

    input_file_lines.append('')
    input_file_lines.append('gap_check={}')
    input_file_lines.append('move_check={}')
    input_file_lines.append('pos_check={}')
    input_file_lines.append(
        "res_dict={'convergence_report':{'GapCheck':{}, 'MoveCheck':{}, 'PosCheck':{}}, 'invariant':{}}"
        )
    # yapf: enable

    input_file_lines.append('')
    if prepend_code != '':
        input_file_lines.append('\t' + prepend_code)
    if dim_mode == '2D' or dim_mode == '3D':
        input_file_lines.append('result = z2pack.surface.run(')
        input_file_lines.append('    system             = system,')
        if dim_mode == '2D':
            if invariant == 'Z2':
                input_file_lines.append(
                    '    surface            = lambda t1,t2: [t2, t1/2, 0],')
            elif invariant == 'Chern':
                input_file_lines.append(
                    '    surface            = lambda t1,t2: [t1, t2, 0],')
        elif dim_mode == '3D':
            input_file_lines.append('    surface            = ' + surface +
                                    ',')
        input_file_lines.append('    pos_tol            = ' + str(pos_tol) +
                                ',')
        input_file_lines.append('    gap_tol            = ' + str(gap_tol) +
                                ',')
        input_file_lines.append('    move_tol           = ' + str(move_tol) +
                                ',')
        input_file_lines.append('    num_lines          = ' + str(num_lines) +
                                ',')
        input_file_lines.append('    min_neighbour_dist = ' +
                                str(min_neighbour_dist) + ',')
        input_file_lines.append('    iterator           = ' + str(iterator) +
                                ',')
        input_file_lines.append('    save_file          = ' + "'" +
                                cls._OUTPUT_SAVE_FILE + "'" + ',')
        if cls.restart_mode:
            input_file_lines.append('    load               = True')
        input_file_lines.append('    )')

        if invariant.lower() == 'z2':
            input_file_lines.append('Z2 = z2pack.invariant.z2(result)')
            input_file_lines.append("res_dict['invariant'].update({'Z2':Z2})")
        elif invariant.lower() == 'chern':
            input_file_lines.append('Chern = z2pack.invariant.chern(result)')
            input_file_lines.append(
                "res_dict['invariant'].update({'Chern':Chern})")
    else:
        raise exceptions.InputValidationError(
            'Only dimension_mode 2D and 3D are currently implemented.')

    input_file_lines.append('')
    input_file_lines.append(
        "gap_check['PASSED']  = "
        "result.convergence_report['surface']['GapCheck']['PASSED']")
    input_file_lines.append(
        "gap_check['FAILED']  = "
        "result.convergence_report['surface']['GapCheck']['FAILED']")
    input_file_lines.append(
        "move_check['PASSED'] = "
        "result.convergence_report['surface']['MoveCheck']['PASSED']")
    input_file_lines.append(
        "move_check['FAILED'] = "
        "result.convergence_report['surface']['MoveCheck']['FAILED']")
    input_file_lines.append(
        "pos_check['PASSED']  = "
        "result.convergence_report['line']['PosCheck']['PASSED']")
    input_file_lines.append(
        "pos_check['FAILED']  = "
        "result.convergence_report['line']['PosCheck']['FAILED']")
    input_file_lines.append(
        "pos_check['MISSING'] = "
        "result.convergence_report['line']['PosCheck']['MISSING']")

    input_file_lines.append('')
    input_file_lines.append(
        "res_dict['convergence_report']['GapCheck'].update(gap_check)")
    input_file_lines.append(
        "res_dict['convergence_report']['MoveCheck'].update(move_check)")
    input_file_lines.append(
        "res_dict['convergence_report']['PosCheck'].update(pos_check)")

    input_file_lines.append('')
    input_file_lines.append("with open('" + cls._OUTPUT_RESULT_FILE +
                            "', 'w') as fp:")
    input_file_lines.append('    json.dump(res_dict, fp)')
    input_file_lines.append('')
    input_file_lines.append('')
    input_file_lines.append('')
    with open(input_filename, 'w') as file_input:
        file_input.write('\n'.join(input_file_lines))
        file_input.write('\n')
