import os
from Sample import Sample


def read_input_file():
    """
    Reads the input file and builds the corresponding Sample objects.
    :return: A list of Sample objects.
    """
    input_dir = os.getcwd()
    input_path = os.path.join(input_dir, "input.csv")
    samples = []
    line_counter = 0
    with open(input_path) as input_file:
        for line in input_file:
            split_line = line.split(",")
            # handle the first line containing healthy/sick labels
            if line_counter == 0:
                for i in range(len(split_line[2:])):
                    sample_idx = i
                    if split_line[i] == "healthy":
                        sample_obj = Sample(sample_idx, True)
                    else:
                        sample_obj = Sample(sample_idx, False)
                    samples.append(sample_obj)
                line_counter += 1
                continue
            # ignore the second line
            elif line_counter == 1:
                line_counter += 1
                continue
            # handle all other lines of the input file, containing species index,
            # strain index, and strain values over all samples.
            species_int = int(split_line[0][len("species"):])
            strain = split_line[1]
            if strain.startswith("strain"):
                strain_int = int(strain[len("strain"):])
            for i in range(2, len(split_line)):
                sample = samples[i-2]
                sample.add_value(strain_int, species_int-1, float(split_line[i]))
            line_counter += 1
    return samples
