import argparse
import pandas as pd
import numpy as np
import hitran_functions

# Mapping of isotopologue numbers to new names
isotopologue_map = {
    "1": 626,
    "2": 636,
    "3": 628,
    "4": 627,
    "5": 638,
    "6": 637,
    "7": 828,
    "8": 827,
    "9": 727,
    "0": 838,
    'A': 837,
    'B': 737
}


def main():
    """
    Initializes the script to process a hitran file into individual MARVEL input files by isotopologue.
    This function sets up an argument parser to handle the following command-line arguments:
    - hitran_filepath (str): Filepath to the hitran file.
    - isotopologues (int): Isotopologues to be extracted.
    - save_folder (str): Path to the folder to save the output files.
    The function performs the following steps:
    1. Parses the command-line arguments.
    2. Converts the hitran file to a DataFrame using hitran_functions.hitran_to_dataframe.
    3. Extracts the specified isotopologues or all unique isotopologues if none are specified.
    4. Creates individual DataFrames for each isotopologue.
    5. Formats each DataFrame to match the MARVEL input requirements.
    6. Saves each formatted DataFrame as a CSV file in the specified save folder.
    """

    parser = argparse.ArgumentParser(
        description='Process hitran file into individual MARVEL input files by isotopologue.')
    parser.add_argument('hitran_filepath', type=str,
                        help='Filepath to the hitran file.')
    parser.add_argument('save_folder', type=str,
                        help='Path to folder to save files.')
    parser.add_argument('molecule', type=str,
                        help='Formula of molecule for naming files.')
    parser.add_argument('--isotopologues', type=int, nargs='*', default=None,
                        help='Isotopologues to be extracted. If not specified, all unique isotopologues will be extracted.')

    args = parser.parse_args()
    print(args)

    print("Script started...")

    hitran_full_df = hitran_functions.hitran_to_dataframe(args.hitran_filepath)
    print("HITRAN file loaded...")

    isotopologues = args.isotopologues
    if isotopologues is None:
        isotopologues = hitran_full_df['I'].unique().tolist()
    print(f"Isotopologues selected: {isotopologues}")

    full_single_isotopologue_dfs = []
    for iso in isotopologues:
        single_iso_df = hitran_functions.single_isotopologue(
            hitran_full_df, iso)
        full_single_isotopologue_dfs.append(single_iso_df)
    print("Single isotopologue dataframes created...")

    marvel_input_isotopologue_dfs = []
    for iso in full_single_isotopologue_dfs:
        marvel_input = iso[["v", "v1'", "v2'", "l2'", "v3'", "r'", "J'", "sym'",
                            "v1''", "v2''", "l2''", "v3''", "r''", "J''", "sym''"]]
        marvel_input.insert(1, 'unc', 0.0001)
        marvel_input = marvel_input.copy()
        marvel_input['tag'] = ['tag' + str(i + 1)
                               for i in range(len(marvel_input))]
        marvel_input_isotopologue_dfs.append(marvel_input)
    print("MARVEL input dataframes created...")

    # Saving the files with the new naming convention
    for i, df in enumerate(marvel_input_isotopologue_dfs):
        # Get the isotopologue number
        iso = isotopologues[i]

        # Map the isotopologue number to the new name using the isotopologue_map dictionary
        # Default to iso if not found in the map
        iso_new_name = isotopologue_map.get(iso, iso)

        output_filepath = f"{args.save_folder}/MARVEL_input_HITRAN_{args.molecule}_{iso_new_name}.txt"
        df.to_csv(output_filepath, sep='\t', index=False)
        print(f"Saved {output_filepath}")

    print("Script finished")


if __name__ == "__main__":
    main()
    # Example usage for two isotopologues:
    # python hitran_to_marvel_inp.py /path/to/hitran_file.par /path/to/save_folder CO2 --isotopologues 626 636
    # Example usage for all isotopologues:
    # python hitran_to_marvel_inp.py /path/to/hitran_file.par /path/to/save_folder CO2
