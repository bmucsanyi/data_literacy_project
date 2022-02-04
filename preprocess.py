"""Interface to our preprocessing routines."""
import argparse
from src.merge_convert import merge_convert

def main():
    """ Merge all DataFrames and convert currencies.""" 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', 
                        type=str, 
                        default="../dat/data_clean.csv", 
                        help="File or filename to which the data is saved.")

    args = parser.parse_args()

    merge_convert(args.file)


if __name__ == '__main__':
    main()