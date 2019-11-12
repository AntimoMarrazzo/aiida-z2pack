# -*- coding: utf-8 -*-
from aiida.orm.calculation.job.quantumespresso.bands import BandsCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.folder import FolderData
from aiida.parsers.parser import Parser
from aiida.common.datastructures import calc_states
from aiida_quantumespresso.parsers.basicpw import QEOutputParsingError
from aiida.common.exceptions import UniquenessError
from aiida.orm.data.array.bands import BandsData
from aiida.orm.data.array.kpoints import KpointsData


__copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/.. All rights reserved."
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file"
__version__ = "0.6.0"
__authors__ = "The AiiDA team."


class BandsParser(Parser):
    """
    This class is the implementation of the basic Parser class for Bands.
    """
    _bands_name = 'output_bands'
    _rap_name = 'output_representations'
    _parity_name = 'output_parities'
    _sigma1_name = 'output_sigma1'
    _sigma2_name = 'output_sigma2'
    _sigma3_name = 'output_sigma3'


    _bands_parameters_name = 'output_bands_parameters'
    _rap_parameters_name = 'output_representations_parameters'
    _parity_parameters_name = 'output_parities_parameters'
    _sigma1_parameters_name = 'output_sigma1_parameters'
    _sigma2_parameters_name = 'output_sigma2_parameters'
    _sigma3_parameters_name = 'output_sigma3_parameters'


    def __init__(self, calculation):
        """
        Initialize the instance of BandsParser
        """
        # check for valid input
        if not isinstance(calculation, BandsCalculation):
            raise QEOutputParsingError("Input calc must be a BandsCalculation")

        self._calc = calculation

        super(BandsParser, self).__init__(calculation)

    def get_linkname_bands(self):
        """
        Returns the name of the link of postprocessed bands
        """
        return self._bands_name

    def get_linkname_rap(self):
        """
        Returns the name of the link of symmetry groups
        """
        return self._rap_name

    def get_linkname_parity(self):
        """
        Returns the name of the link of symmetry groups
        """
        return self._parity_name

    def get_linkname_sigma1(self):
        """
        Returns the name of the link of symmetry groups
        """
        return self._sigma1_name

    def get_linkname_sigma2(self):
        """
        Returns the name of the link of symmetry groups
        """
        return self._sigma2_name

    def get_linkname_sigma3(self):
        """
        Returns the name of the link of symmetry groups
        """
        return self._sigma3_name



    def get_linkname_bands_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._bands_parameters_name

    def get_linkname_rap_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._rap_parameters_name

    def get_linkname_parity_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._parity_parameters_name

    def get_linkname_sigma1_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._sigma1_parameters_name

    def get_linkname_sigma2_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._sigma2_parameters_name


    def get_linkname_sigma3_parameters(self):
        """
        Returns the name of the link of units
        """
        return self._sigma3_parameters_name





    def parse_with_retrieved(self, retrieved):
        """
        Parses the datafolder, stores results.
        Retrieves bands output, symmetry output and some basic information from the
        out_file, such as warnings and wall_time
        """
        from aiida.common.exceptions import InvalidOperation

        # suppose at the start that the job is successful
        successful = True
        new_nodes_list = []

        # check if I'm not to overwrite anything
        state = self._calc.get_state()
        if state != calc_states.PARSING:
           raise InvalidOperation("Calculation not in {} state")

        try:
            out_folder = self._calc.get_retrieved_node()
        except KeyError:
            self.logger.error("No retrieved folder found")
            return successful, new_nodes_list

        # Read standard out
        try:
            filpath = out_folder.get_abs_path(self._calc._OUTPUT_FILE_NAME)
            with open(filpath, 'r') as fil:
                    out_file = fil.readlines()
        except OSError:
            self.logger.error("Standard output file could not be found.")
            successful = False
            return successful, new_nodes_list

        #TO IMPLEMENT:
        #catching errors from outfile

        #successful = False
        #for i in range(len(out_file)):
        #    line = out_file[-i]
        #    if "JOB DONE" in line:
        #        successful = True
        #        break
        #if not successful:
        #    self.logger.error("Computation did not finish properly")
        #    return successful, new_nodes_list

        #everything standard till there


        input_params=self._calc.get_inputs_dict()['parameters'].get_dict()

        # check that the dos file is present, if it is, read it
        try:
            bands_path = out_folder.get_abs_path(self._calc._BANDS_FILENAME)
            #with open(bands_path, 'r') as fil:
            #        bands_file = fil.readlines()
        except OSError:
            successful = False
            self.logger.error("Bands output file could not found")
            return successful, new_nodes_list


        #--parsing bands
        parsed_bands_data = parse_raw_bands_file(bands_path)  
         
        # extract kpoints bands (and take out from output dictionary) 
        array_kpoints = parsed_bands_data.pop('array_kpoints')
        kpointsdata_for_bands = KpointsData()
        kpointsdata_for_bands.set_kpoints(array_kpoints)

        # extract bands (and take out from output dictionary)
        postprocessed_bands = parsed_bands_data.pop('array_bands')
        
        # save bands into BandsData
        output_bands = BandsData()
        output_bands.set_kpointsdata(kpointsdata_for_bands)
        output_bands.set_bands(postprocessed_bands,units='eV')

        output_bands_params = ParameterData(dict=parsed_bands_data)

        for message in parsed_bands_data['warnings']:
            self.logger.error(message)

        new_nodes_list = []
        new_nodes_list.append( (self.get_linkname_bands_parameters(),output_bands_params) )
        new_nodes_list.append( (self.get_linkname_bands(),output_bands) )



        if ('lsym' in input_params['BANDS'].keys()):
	    if input_params['BANDS']['lsym'] == True:

                try:
                    rap_path = out_folder.get_abs_path(self._calc._RAP_FILENAME)
            #with open(simmetries_path, 'r') as filS:
            #        simmetries_file = filS.readlines()
                except OSError:
                    successful = False
                    self.logger.error("Rap output file could not found")
                    return successful, new_nodes_list



                #--parsing symmetries
                parsed_rap_data = parse_raw_bands_file(rap_path)  
         
                # extract kpoints bands (and take out from output dictionary) 
                array_kpoints = parsed_rap_data.pop('array_kpoints')
                kpointsdata_for_rap = KpointsData()
                kpointsdata_for_rap.set_kpoints(array_kpoints)

                # extract bands symmetry (and take out from output dictionary)
                postprocessed_rap = parsed_rap_data.pop('array_bands')
        
                # save bands symmetry into BandsData
                output_rap = BandsData()
                output_rap.set_kpointsdata(kpointsdata_for_rap)
                output_rap.set_bands(postprocessed_rap,units='None')

                output_rap_params = ParameterData(dict=parsed_rap_data)

                for message in parsed_rap_data['warnings']:
                    self.logger.error(message)
        

                new_nodes_list.append( (self.get_linkname_rap_parameters(),output_rap_params) )
                new_nodes_list.append( (self.get_linkname_rap(),output_rap) )



        if ('parity' in input_params['BANDS'].keys()):
	    if input_params['BANDS']['parity'] == True:

                try:
                    par_path = out_folder.get_abs_path(self._calc._PARITY_FILENAME)
            #with open(simmetries_path, 'r') as filS:
            #        simmetries_file = filS.readlines()
                except OSError:
                    successful = False
                    self.logger.error("Parity output file could not found")
                    return successful, new_nodes_list



                #--parsing symmetries
                parsed_par_data = parse_raw_bands_file(par_path)  
         
                # extract kpoints bands (and take out from output dictionary) 
                array_kpoints = parsed_par_data.pop('array_kpoints')
                kpointsdata_for_par = KpointsData()
                kpointsdata_for_par.set_kpoints(array_kpoints)

                # extract bands symmetry (and take out from output dictionary)
                postprocessed_par = parsed_par_data.pop('array_bands')
        
                # save bands symmetry into BandsData
                output_par = BandsData()
                output_par.set_kpointsdata(kpointsdata_for_par)
                output_par.set_bands(postprocessed_par,units='None')

                output_par_params = ParameterData(dict=parsed_par_data)

                for message in parsed_par_data['warnings']:
                    self.logger.error(message)
        

                new_nodes_list.append( (self.get_linkname_parity_parameters(),output_par_params) )
                new_nodes_list.append( (self.get_linkname_parity(),output_par) )



        if ('lsigma(1)' in input_params['BANDS'].keys()):
	    if input_params['BANDS']['lsigma(1)'] == True:

                try:
                    sigma1_path = out_folder.get_abs_path(self._calc._SIGMA1_FILENAME)
            #with open(simmetries_path, 'r') as filS:
            #        simmetries_file = filS.readlines()
                except OSError:
                    successful = False
                    self.logger.error("sigma1 output file could not found")
                    return successful, new_nodes_list



                #--parsing symmetries
                parsed_sigma1_data = parse_raw_bands_file(sigma1_path)  
         
                # extract kpoints bands (and take out from output dictionary) 
                array_kpoints = parsed_sigma1_data.pop('array_kpoints')
                kpointsdata_for_sigma1 = KpointsData()
                kpointsdata_for_sigma1.set_kpoints(array_kpoints)

                # extract bands symmetry (and take out from output dictionary)
                postprocessed_sigma1 = parsed_sigma1_data.pop('array_bands')
        
                # save bands symmetry into BandsData
                output_sigma1 = BandsData()
                output_sigma1.set_kpointsdata(kpointsdata_for_sigma1)
                output_sigma1.set_bands(postprocessed_sigma1,units='None')

                output_sigma1_params = ParameterData(dict=parsed_sigma1_data)

                for message in parsed_sigma1_data['warnings']:
                    self.logger.error(message)
        

                new_nodes_list.append( (self.get_linkname_sigma1_parameters(),output_sigma1_params) )
                new_nodes_list.append( (self.get_linkname_sigma1(),output_sigma1) )

        if ('lsigma(2)' in input_params['BANDS'].keys()):
	    if input_params['BANDS']['lsigma(2)'] == True:

                try:
                    sigma2_path = out_folder.get_abs_path(self._calc._SIGMA2_FILENAME)
            #with open(simmetries_path, 'r') as filS:
            #        simmetries_file = filS.readlines()
                except OSError:
                    successful = False
                    self.logger.error("sigma2 output file could not found")
                    return successful, new_nodes_list



                #--parsing symmetries
                parsed_sigma2_data = parse_raw_bands_file(sigma2_path)  
         
                # extract kpoints bands (and take out from output dictionary) 
                array_kpoints = parsed_sigma2_data.pop('array_kpoints')
                kpointsdata_for_sigma2 = KpointsData()
                kpointsdata_for_sigma2.set_kpoints(array_kpoints)

                # extract bands symmetry (and take out from output dictionary)
                postprocessed_sigma2 = parsed_sigma2_data.pop('array_bands')
        
                # save bands symmetry into BandsData
                output_sigma2 = BandsData()
                output_sigma2.set_kpointsdata(kpointsdata_for_sigma2)
                output_sigma2.set_bands(postprocessed_sigma2,units='None')

                output_sigma2_params = ParameterData(dict=parsed_sigma2_data)

                for message in parsed_sigma2_data['warnings']:
                    self.logger.error(message)
        

                new_nodes_list.append( (self.get_linkname_sigma2_parameters(),output_sigma2_params) )
                new_nodes_list.append( (self.get_linkname_sigma2(),output_sigma2) )


        if ('lsigma(3)' in input_params['BANDS'].keys()):
	    if input_params['BANDS']['lsigma(3)'] == True:

                try:
                    sigma3_path = out_folder.get_abs_path(self._calc._SIGMA3_FILENAME)
            #with open(simmetries_path, 'r') as filS:
            #        simmetries_file = filS.readlines()
                except OSError:
                    successful = False
                    self.logger.error("sigma3 output file could not found")
                    return successful, new_nodes_list



                #--parsing symmetries
                parsed_sigma3_data = parse_raw_bands_file(sigma3_path)  
         
                # extract kpoints bands (and take out from output dictionary) 
                array_kpoints = parsed_sigma3_data.pop('array_kpoints')
                kpointsdata_for_sigma3 = KpointsData()
                kpointsdata_for_sigma3.set_kpoints(array_kpoints)

                # extract bands symmetry (and take out from output dictionary)
                postprocessed_sigma3 = parsed_sigma3_data.pop('array_bands')
        
                # save bands symmetry into BandsData
                output_sigma3 = BandsData()
                output_sigma3.set_kpointsdata(kpointsdata_for_sigma3)
                output_sigma3.set_bands(postprocessed_sigma3,units='None')

                output_sigma3_params = ParameterData(dict=parsed_sigma3_data)

                for message in parsed_sigma3_data['warnings']:
                    self.logger.error(message)
        

                new_nodes_list.append( (self.get_linkname_sigma3_parameters(),output_sigma3_params) )
                new_nodes_list.append( (self.get_linkname_sigma3(),output_sigma3) )




            
        return successful,new_nodes_list

def parse_raw_bands_file(path_bands):
    """
    Parses the bands frequencies file
    :param bands_file: bands or symmetry data
    
    :return dict parsed_data: keys:
         * warnings: parser warnings raised
         * num_kpoints: number of kpoints read from the file
         * num_bands: number of bands for each kpoint
         * array_kpoints: numpy array with the k-points
         * array_bands: numpy array with the post-processesd bands or symmetry group for each kpoint
    """
    import numpy
    import re
    
    parsed_data = {}
    parsed_data['warnings'] = []

    # read file
    with open(path_bands,'r') as f:
        lines = f.read()
    
    # extract numbere of bands and kpoints
    try:
        num_bands = int( lines.split("=")[1].split(',')[0] )
        num_kpoints = int( lines.split("=")[2].split('/')[0] )
        parsed_data['num_kpoints'] = num_kpoints
        parsed_data['num_bands'] = num_bands
    except (ValueError,IndexError):
        parsed_data['warnings'].append("Number of bands or kpoints unreadable "
                                       "in phonon frequencies file")
        return parsed_data

    # initialize array of frequencies
    freq_matrix = numpy.zeros((num_kpoints,num_bands))

    # initialize array of kpoints
    kpoints_matrix = numpy.zeros((num_kpoints,3))

    split_data = lines.split()
    # discard the header of the file
    raw_data = split_data[split_data.index('/')+1:]
    

    corrected_data = []
    for b in raw_data:
        try:
            corrected_data.append(float(b))
        except:
	    pass

            # ValueError:
            # parsed_data["warnings"].append("Bad formatting")
            # return parsed_data

    counter = 3
    for i in range(num_kpoints):
        for j in range(num_bands):
            try:
                freq_matrix[i,j] = corrected_data[counter]
            except ValueError:
                parsed_data["warnings"].append("Error while parsing the "
                                               "energies") 
            except IndexError:
                parsed_data["warnings"].append("Error while parsing the "
                                               "energies, dimension exceeded")
                return parsed_data
            counter += 1
        counter += 3 # move past the kpoint coordinates
    
            
    parsed_data['array_bands'] = freq_matrix

    counter = 0
    for i in range(num_kpoints):
        for j in range(3):
            try:
                kpoints_matrix[i,j] = corrected_data[counter]
            except ValueError:
                parsed_data["warnings"].append("Error while parsing the "
                                               "kpoints") 
            except IndexError:
                parsed_data["warnings"].append("Error while parsing the "
                                               "kpoints, dimension exceeded")
                return parsed_data
            counter += 1
        counter += num_bands # move past the kpoint coordinates
    
            
    parsed_data['array_kpoints'] = kpoints_matrix

    #returining dictionary
    return parsed_data
























