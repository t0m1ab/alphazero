import sys
import os

import alphazero
from alphazero.tests import run_tests
from alphazero.utils import download_all_models_from_hf_hub


def print_help(help_msg_relative_path: str = "docs/help.txt"):
    """ Print the help message for the cs3arl package """
    filepath = os.path.join(alphazero.__path__[0], help_msg_relative_path)
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
        print("command 'alphazero' is working: try --help or --test")


if __name__ == "__main__":
    main()