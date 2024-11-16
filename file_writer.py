# Define input and output file paths
input_file_paths = ['data/train-consonant-examples.txt']
output_file_path = 'data/folder_15/first_10_train-consonant-examples.txt'

# Open the output file for writing
with open(output_file_path, 'w') as output_file:
    # Loop through each input file
    for input_file_path in input_file_paths:
        # Open the current input file for reading
        with open(input_file_path, 'r') as input_file:
            # Read each line, extract the first 15 characters, and write to the output file
            for line in input_file:
                output_file.write(line[:10] + '\n')  # Take first 15 characters and write to output

print(f"Extracted first 15 characters written to {output_file_path}")
