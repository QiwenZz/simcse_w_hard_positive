with open('data/wiki1m_for_simcse.txt', 'r') as original, open('data/modified_file.txt', 'w') as modified:
    # Read the first three lines from the original file
    first_three_lines = [next(original) for _ in range(50)]
    # Write these lines to the new file
    modified.writelines(first_three_lines)