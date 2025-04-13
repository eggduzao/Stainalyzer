
import os
import sys

def sort_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sorted_lines = sorted([line.strip() for line in lines if line.strip()])

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sorted_lines:
            f.write(line + '\n')

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python sort_lines.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sort_lines(input_file, output_file)
    print(f"Sorted lines written to {output_file}")