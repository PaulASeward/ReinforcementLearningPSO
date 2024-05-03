import os

def remove_computecanada_suffix(requirements_file):
    with open(requirements_file, 'r') as file:
        lines = file.readlines()

    # Remove the "+computecanada" suffix from each line
    updated_lines = [line.replace('+computecanada', '') for line in lines]

    # Write the updated lines back to the requirements file
    with open(requirements_file, 'w') as file:
        file.writelines(updated_lines)

if __name__ == "__main__":
    requirements_file = "currentversions.txt"
    remove_computecanada_suffix(requirements_file)
    print(f"'+computecanada' suffix removed from '{requirements_file}'")
