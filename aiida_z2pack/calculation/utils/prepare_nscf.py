import os
from aiida import orm
from aiida.common import exceptions
from aiida_quantumespresso.calculations import _uppercase_dict, _lowercase_dict

def prepare_nscf(self,tempfolder,inputdict_nscf):
    """ 
    This methods generate the input data for the nscf part of the calculation
    """
    inputdict = inputdict_nscf
    def _generate_PWCPinputdata(self,parameters,settings_dict,pseudos,structure,kpoints=None):
        """
        This method creates the content of an input file
        in the PW/CP format.
        :
        """
        from aiida.common.utils import get_unique_filename, get_suggestion
        import re
        local_copy_list_to_append = []

        # I put the first-level keys as uppercase (i.e., namelist and card names)
        # and the second-level keys as lowercase
        # (deeper levels are unchanged)

        input_params = _uppercase_dict(parameters.get_dict(),
                                       dict_name='parameters')
        input_params = {k: _lowercase_dict(v, dict_name=k)
                        for k, v in input_params.iteritems()}

        # I remove unwanted elements (for the moment, instead, I stop; to change when
        # we setup a reasonable logging)
        for blocked in self._blocked_keywords:
            nl = blocked[0].upper()
            flag = blocked[1].lower()
            defaultvalue = None
            if len(blocked) >= 3:
                defaultvalue = blocked[2]
            if nl in input_params:
                # The following lines is meant to avoid putting in input the
                # parameters like celldm(*)
                stripped_inparams = [re.sub("[(0-9)]", "", _)
                                     for _ in input_params[nl].keys()]
                if flag in stripped_inparams:
                    raise exceptions.InputValidationError(
                        "You cannot specify explicitly the '{}' flag in the '{}' "
                        "namelist or card.".format(flag, nl))
                if defaultvalue is not None:
                    if nl not in input_params:
                        input_params[nl] = {}
                    input_params[nl][flag] = defaultvalue

        # Set some variables (look out at the case! NAMELISTS should be uppercase,
        # internal flag names must be lowercase)
        if 'CONTROL' not in input_params:
            input_params['CONTROL'] = {}
        input_params['CONTROL']['pseudo_dir'] = '.'+self._PSEUDO_SUBFOLDER
        input_params['CONTROL']['outdir'] = '.'+self._OUTPUT_SUBFOLDER
        input_params['CONTROL']['prefix'] = self._PREFIX

        input_params['CONTROL']['verbosity'] = input_params['CONTROL'].get(
            'verbosity', self._default_verbosity)  # Set to high if not specified

        # ============ I prepare the input site data =============
        # ------------ CELL_PARAMETERS -----------
        cell_parameters_card = "CELL_PARAMETERS angstrom\n"
        for vector in structure.cell:
            cell_parameters_card += ("{0:18.10f} {1:18.10f} {2:18.10f}"
                                     "\n".format(*vector))

        # ------------- ATOMIC_SPECIES ------------
        atomic_species_card_list = []

        # Keep track of the filenames to avoid to overwrite files
        # I use a dictionary where the key is the pseudo PK and the value
        # is the filename I used. In this way, I also use the same filename
        # if more than one kind uses the same pseudo.
        pseudo_filenames = {}

        # I keep track of the order of species
        kind_names = []
        # I add the pseudopotential files to the list of files to be copied
        for kind in structure.kinds:
            # This should not give errors, I already checked before that
            # the list of keys of pseudos and kinds coincides
            ps = pseudos[kind.name]
            if kind.is_alloy() or kind.has_vacancies():
                raise exceptions.InputValidationError("Kind '{}' is an alloy or has "
                                           "vacancies. This is not allowed for pw.x input structures."
                                           "".format(kind.name))

            try:
                # It it is the same pseudopotential file, use the same filename
                filename = pseudo_filenames[ps.pk]
            except KeyError:
                # The pseudo was not encountered yet; use a new name and
                # also add it to the local copy list
                filename = get_unique_filename(ps.filename,
                                               pseudo_filenames.values())
                pseudo_filenames[ps.pk] = filename
                # I add this pseudo file to the list of files to copy
                local_copy_list_to_append.append((ps.get_file_abs_path(),
                                                  os.path.join(self._PSEUDO_SUBFOLDER,
                                                               filename)))
            kind_names.append(kind.name)
            atomic_species_card_list.append("{} {} {}\n".format(
                kind.name.ljust(6), kind.mass, filename))

        # I join the lines, but I resort them using the alphabetical order of
        # species, given by the kind_names list. I also store the mapping_species
        # list, with the order of species used in the file
        mapping_species, sorted_atomic_species_card_list = zip(
            *sorted(zip(kind_names, atomic_species_card_list)))
        # The format of mapping_species required later is a dictionary, whose
        # values are the indices, so I convert to this format
        # Note the (idx+1) to convert to fortran 1-based lists
        mapping_species = {sp_name: (idx + 1) for idx, sp_name
                           in enumerate(mapping_species)}
        # I add the first line
        sorted_atomic_species_card_list = (["ATOMIC_SPECIES\n"] +
                                           list(sorted_atomic_species_card_list))
        atomic_species_card = "".join(sorted_atomic_species_card_list)
        # Free memory
        del sorted_atomic_species_card_list
        del atomic_species_card_list

        # ------------ ATOMIC_POSITIONS -----------
        atomic_positions_card_list = ["ATOMIC_POSITIONS angstrom\n"]

        # Check on validity of FIXED_COORDS
        fixed_coords_strings = []
        fixed_coords = settings_dict.pop('FIXED_COORDS', None)
        if fixed_coords is None:
            # No fixed_coords specified: I store a list of empty strings
            fixed_coords_strings = [""] * len(structure.sites)
        else:
            if len(fixed_coords) != len(structure.sites):
                raise exceptions.InputValidationError(
                    "Input structure contains {:d} sites, but "
                    "fixed_coords has length {:d}".format(len(structure.sites),
                                                          len(fixed_coords)))

            for i, this_atom_fix in enumerate(fixed_coords):
                if len(this_atom_fix) != 3:
                    raise exceptions.InputValidationError("fixed_coords({:d}) has not length three"
                                               "".format(i + 1))
                for fixed_c in this_atom_fix:
                    if not isinstance(fixed_c, bool):
                        raise exceptions.InputValidationError("fixed_coords({:d}) has non-boolean "
                                                   "elements".format(i + 1))

                if_pos_values = [self._if_pos(_) for _ in this_atom_fix]
                fixed_coords_strings.append("  {:d} {:d} {:d}".format(*if_pos_values))

        for site, fixed_coords_string in zip(
                structure.sites, fixed_coords_strings):
            atomic_positions_card_list.append(
                "{0} {1:18.10f} {2:18.10f} {3:18.10f} {4}\n".format(
                    site.kind_name.ljust(6), site.position[0], site.position[1],
                    site.position[2], fixed_coords_string))
        atomic_positions_card = "".join(atomic_positions_card_list)
        del atomic_positions_card_list  # Free memory

        # I set the variables that must be specified, related to the system
        # Set some variables (look out at the case! NAMELISTS should be
        # uppercase, internal flag names must be lowercase)
        if 'SYSTEM' not in input_params:
            input_params['SYSTEM'] = {}
        input_params['SYSTEM']['ibrav'] = 0
        input_params['SYSTEM']['nat'] = len(structure.sites)
        input_params['SYSTEM']['ntyp'] = len(structure.kinds)
    
        # ============ I prepare the k-points =============
        if self._use_kpoints:
            try:
                mesh, offset = kpoints.get_kpoints_mesh()
                has_mesh = True
                force_kpoints_list = settings_dict.pop('FORCE_KPOINTS_LIST', False)
                if force_kpoints_list:
                    kpoints_list = kpoints.get_kpoints_mesh(print_list=True)
                    num_kpoints = len(kpoints_list)
                    has_mesh = False
                    weights = [1.] * num_kpoints

            except AttributeError:

                try:
                    kpoints_list = kpoints.get_kpoints()
                    num_kpoints = len(kpoints_list)
                    has_mesh = False
                    if num_kpoints == 0:
                        raise exceptions.InputValidationError("At least one k point must be "
                                                   "provided for non-gamma calculations")
                except AttributeError:
                    raise exceptions.InputValidationError("No valid kpoints have been found")

                try:
                    _, weights = kpoints.get_kpoints(also_weights=True)
                except AttributeError:
                    weights = [1.] * num_kpoints

            gamma_only = settings_dict.pop("GAMMA_ONLY", False)

            if gamma_only:
                if has_mesh:
                    if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0., 0., 0.):
                        raise exceptions.InputValidationError(
                            "If a gamma_only calculation is requested, the "
                            "kpoint mesh must be (1,1,1),offset=(0.,0.,0.)")

                else:
                    if ( len(kpoints_list) != 1 or
                                 tuple(kpoints_list[0]) != tuple(0., 0., 0.) ):
                        raise exceptions.InputValidationError(
                            "If a gamma_only calculation is requested, the "
                            "kpoints coordinates must only be (0.,0.,0.)")

                kpoints_type = "gamma"

            elif has_mesh:
                kpoints_type = "automatic"

            else:
                kpoints_type = "crystal"

            kpoints_card_list = ["K_POINTS {}\n".format(kpoints_type)]

            if kpoints_type == "automatic":
                if any([(i != 0. and i != 0.5) for i in offset]):
                    raise exceptions.InputValidationError("offset list must only be made "
                                               "of 0 or 0.5 floats")
                the_offset = [0 if i == 0. else 1 for i in offset]
                the_6_integers = list(mesh) + the_offset
                kpoints_card_list.append("{:d} {:d} {:d} {:d} {:d} {:d}\n"
                                         "".format(*the_6_integers))

            elif kpoints_type == "gamma":
                # nothing to be written in this case
                pass
            else:
                kpoints_card_list.append("{:d}\n".format(num_kpoints))
                for kpoint, weight in zip(kpoints_list, weights):
                    kpoints_card_list.append(
                        "  {:18.10f} {:18.10f} {:18.10f} {:18.10f}"
                        "\n".format(kpoint[0], kpoint[1], kpoint[2], weight))

            kpoints_card = "".join(kpoints_card_list)
            del kpoints_card_list

        # =================== NAMELISTS AND CARDS ========================
        try:
            namelists_toprint = settings_dict.pop('NAMELISTS')
            if not isinstance(namelists_toprint, list):
                raise exceptions.InputValidationError(
                    "The 'NAMELISTS' value, if specified in the settings input "
                    "node, must be a list of strings")
        except KeyError:  # list of namelists not specified; do automatic detection
            try:
                control_nl = input_params['CONTROL']
                calculation_type = control_nl['calculation']
            except KeyError:
                raise exceptions.InputValidationError(
                    "No 'calculation' in CONTROL namelist."
                    "It is required for automatic detection of the valid list "
                    "of namelists. Otherwise, specify the list of namelists "
                    "using the NAMELISTS key inside the 'settings' input node")

            try:
                namelists_toprint = self._automatic_namelists[calculation_type]
            except KeyError:
                sugg_string = get_suggestion(calculation_type,
                                             self._automatic_namelists.keys())
                raise exceptions.InputValidationError("Unknown 'calculation' value in "
                                           "CONTROL namelist {}. Otherwise, specify the list of "
                                           "namelists using the NAMELISTS inside the 'settings' input "
                                           "node".format(sugg_string))

        inputfile = ""
        for namelist_name in namelists_toprint:
            inputfile += "&{0}\n".format(namelist_name)
            # namelist content; set to {} if not present, so that we leave an 
            # empty namelist
            namelist = input_params.pop(namelist_name, {})
            for k, v in sorted(namelist.iteritems()):
                inputfile += get_input_data_text(k, v, mapping=mapping_species)
            inputfile += "/\n"
    
        # Write cards now
        inputfile += atomic_species_card
        inputfile += atomic_positions_card
        if self._use_kpoints:
            inputfile += kpoints_card
        inputfile += cell_parameters_card
        #TODO: write CONSTRAINTS
        #TODO: write OCCUPATIONS
            
        if input_params:            
            raise exceptions.InputValidationError(
                "The following namelists are specified in input_params, but are "
                "not valid namelists for the current type of calculation: "
                "{}".format(",".join(input_params.keys())))

        return inputfile, local_copy_list_to_append

    def _prepare_for_submission_nscf(self, tempfolder,inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputs_dict (without the orm.Code!)
        """
        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        try:
            parameters = inputdict.pop(self.get_linkname('nscf_parameters'))
        except KeyError:
            raise exceptions.InputValidationError("No parameters specified for this calculation")
        if not isinstance(parameters, orm.Dict):
            raise exceptions.InputValidationError("parameters is not of type orm.Dict")

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise exceptions.InputValidationError("No structure specified for this calculation")
        if not isinstance(structure, orm.StructureData):
            raise exceptions.InputValidationError("structure is not of type orm.StructureData")

        kpoints = None

        # Settings can be undefined, and defaults to an empty dictionary
        settings = inputdict.pop(self.get_linkname('settings'), None)
        if settings is None:
            settings_dict = {}
        else:
            if not isinstance(settings, orm.Dict):
                raise exceptions.InputValidationError("settings, if specified, must be of "
                                           "type orm.Dict")
            # Settings converted to uppercase
            settings_dict = _uppercase_dict(settings.get_dict(),
                                            dict_name='settings')

        pseudos = {}
        # I create here a dictionary that associates each kind name to a pseudo
        for link in inputdict.keys():
            if link.startswith(self._get_linkname_pseudo_prefix()):
                kindstring = link[len(self._get_linkname_pseudo_prefix()):]
                kinds = kindstring.split('_')
                the_pseudo = inputdict.pop(link)
                if not isinstance(the_pseudo, orm.UpfData):
                    raise exceptions.InputValidationError("Pseudo for kind(s) {} is not of "
                                               "type orm.UpfData".format(",".join(kinds)))
                for kind in kinds:
                    if kind in pseudos:
                        raise exceptions.InputValidationError("Pseudo for kind {} passed "
                                                   "more than one time".format(kind))
                    pseudos[kind] = the_pseudo

        parent_calc_folder = inputdict.pop(self.get_linkname('parent_folder'), None)
        if parent_calc_folder is not None:
            if not isinstance(parent_calc_folder, orm.RemoteData):
                raise exceptions.InputValidationError("parent_calc_folder, if specified, "
                                           "must be of type orm.RemoteData")


        try:
            code = inputdict.pop(self.get_linkname('nscf_code'))
        except KeyError:
            raise exceptions.InputValidationError("No code specified for this calculation")

        # Here, there should be no more parameters...
        # if inputdict:
        #     raise exceptions.InputValidationError("The following input data nodes are "
        #                               "unrecognized: {}".format(inputdict.keys()))

        # Check structure, get species, check peudos
        kindnames = [k.name for k in structure.kinds]
        if set(kindnames) != set(pseudos.keys()):
            err_msg = ("Mismatch between the defined pseudos and the list of "
                       "kinds of the structure. Pseudos: {}; kinds: {}".format(
                ",".join(pseudos.keys()), ",".join(list(kindnames))))
            raise exceptions.InputValidationError(err_msg)

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################
        # I create the subfolder that will contain the pseudopotentials
        tempfolder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)
        # I create the subfolder with the output data (sometimes Quantum
        # Espresso codes crash if an empty folder is not already there
        tempfolder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)

        # If present, add also the Van der Waals table to the pseudo dir
        # Note that the name of the table is not checked but should be the
        # one expected by QE.
        # if vdw_table:
        #     local_copy_list.append(
        #         (
        #         vdw_table.get_file_abs_path(),
        #         os.path.join(self._PSEUDO_SUBFOLDER,
        #             os.path.split(vdw_table.get_file_abs_path())[1])
        #         )
        #         )
        input_filecontent, local_copy_pseudo_list = _generate_PWCPinputdata(self,parameters,settings_dict,pseudos,
                                                                                 structure,kpoints)
        local_copy_list += local_copy_pseudo_list
        #calcinfo = CalcInfo()
        input_filename = tempfolder.get_abs_path(self._INPUT_NSCF_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_filecontent)

        # operations for restart
        symlink = settings_dict.pop('PARENT_FOLDER_SYMLINK', self._default_symlink_usage)  # a boolean
        if symlink:
            if parent_calc_folder is not None:
                # I put the symlink to the old parent ./out folder
                remote_symlink_list.append(
                    (parent_calc_folder.get_computer().uuid,
                     os.path.join(parent_calc_folder.get_remote_path(),
                                  self._restart_copy_from),
                     self._restart_copy_to
                    ))
        else:
            # copy remote output dir, if specified
            if parent_calc_folder is not None:
                remote_copy_list.append(
                    (parent_calc_folder.get_computer().uuid,
                     os.path.join(parent_calc_folder.get_remote_path(),
                                  self._restart_copy_from),
                     self._restart_copy_to
                    ))

        #return calcinfo
        return local_copy_list,remote_copy_list,remote_symlink_list


    #Call to the _prepare_for_submission_nscf
    #print inputdict
    lc,rc,sy=_prepare_for_submission_nscf(self, tempfolder,inputdict=inputdict)
    return lc,rc,sy
