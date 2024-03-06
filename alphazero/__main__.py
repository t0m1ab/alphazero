import sys
import os

import alphazero
from alphazero.utils import DEFAULT_DOCS_PATH
from alphazero.tests import run_tests
from alphazero.utils import download_all_models_from_hf_hub

def print_help():
    """ Print the help message for the cs3arl package """
    filepath = os.path.join(DEFAULT_DOCS_PATH, "help.txt")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as file:
        print(file.read())


def main():
    """ 
    Entry point for the application script.
    (sys.argv = list of arguments given to the program as strings separated by spaces)
    """

    if ("--help" in sys.argv) or ("-h" in sys.argv):
        print_help()

    elif ("--test" in sys.argv) or ("-t" in sys.argv):
        run_tests()
    
    elif ("--download" in sys.argv) or ("-d" in sys.argv):
        download_all_models_from_hf_hub(verbose=True)

    else:
        print("Available commands:")
        print("--help | -h : print help docs")
        print("--test | -t : run simple tests to check the package")
        print("--download | -d : download all models from the HF Hub linked to the default credentials")


if __name__ == "__main__":
    main()