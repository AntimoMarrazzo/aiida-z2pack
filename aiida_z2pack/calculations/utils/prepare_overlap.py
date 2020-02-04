from aiida_quantumespresso.calculations.namelists import NamelistsCalculation


def prepare_overlap(cls,folder):
	parameters = cls.inputs.overlap_parameters.get_dict()
	blocked    = cls._blocked_keywords_overlap
	# namelists  = NamelistsCalculation._default_namelists

	NamelistsCalculation._blocked_keywords = blocked

	parameters = NamelistsCalculation._set_blocked_keywords(parameters)

	content = NamelistsCalculation._generate_input_file(parameters) + '\n\n'

	input_filename = cls._INPUT_OVERLAP_FILE

	with folder.open(input_filename, 'w') as infile:
		infile.write(content)
