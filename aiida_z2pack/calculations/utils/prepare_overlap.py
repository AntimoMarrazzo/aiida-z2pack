from __future__ import absolute_import
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation


class Temp(NamelistsCalculation):
    pass


def prepare_overlap(cls, folder):
    parameters = cls.inputs.overlap_parameters.get_dict()
    blocked = cls._blocked_keywords_overlap

    Temp._blocked_keywords = blocked

    parameters = Temp.set_blocked_keywords(parameters)

    content = Temp.generate_input_file(parameters) + '\n\n'

    input_filename = cls._INPUT_OVERLAP_FILE

    with folder.open(input_filename, 'w') as infile:
        infile.write(content)
