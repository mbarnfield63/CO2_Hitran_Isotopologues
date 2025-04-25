import pandas as pd
import numpy as np

'''
This file deals with all the necessary functions for importing a HITRAN 160 file.
'''


def hitran_to_dataframe(filepath):
    """
    Converts a HITRAN formatted file to a pandas DataFrame.
    Parameters:
    filepath (str): The path to the HITRAN formatted file.
    Returns:
    pandas.DataFrame: A DataFrame containing the data from the HITRAN file with columns:
        - 'M': Integer, 2 characters
        - 'I': Integer, 1 character
        - 'v': Float, 12 characters
        - 'S': Exponential, 10 characters
        - 'A': Exponential, 10 characters
        - 'gamma_air': Float, 5 characters
        - 'gamma_self': Float, 5 characters
        - 'E\'\'': Float, 10 characters
        - 'n_air': Float, 4 characters
        - 'sigma_air': Float, 8 characters
        - 'V\'': String, 15 characters
        - 'V\'\'': String, 15 characters
        - 'Q\'': String, 15 characters
        - 'Q\'\'': String, 15 characters
        - 'I_error': Integer, 12 characters (6 integers, 2 chars each)
        - 'I_ref': Integer, 12 characters (6 integers, 2 chars each)
        - 'flag': String, 1 character
        - 'g\'': Float, 7 characters
        - 'g\'\'': Float, 7 characters
    """

    # Define the format specification
    format_spec = [('M', 2), ('I', 1), ('v', 12), ('S', 10), ('A', 10),
                   ('gamma_air', 5), ('gamma_self', 5),
                   ('E\'\'', 10), ('n_air', 4),
                   ('sigma_air', 8), ('V\'', 15), ('V\'\'', 15),
                   ('Q\'', 15), ('Q\'\'', 15),
                   ('I_error', 6), ('I_ref', 12),
                   ('flag', 1), ('g\'', 7), ('g\'\'', 7)]

    # Calculate the column widths
    colspecs = [(sum([width for _, width in format_spec[:i]]), sum(
        [width for _, width in format_spec[:i+1]])) for i in range(len(format_spec))]

    # Read the file using pandas
    HITEMP_ALL = pd.read_fwf(filepath, colspecs=colspecs, names=[
                             name for name, _ in format_spec])
    return HITEMP_ALL


def infer_parity(J, l2):
    """
    Infers the e/f parity of a rotational-vibrational state.

    Parameters:
        J (int): Rotational quantum number
        l2 (str or int): Vibrational angular momentum quantum number for bending mode
    Returns:
        str: 'e' or 'f' or '' if parity not applicable
    """
    try:
        l2 = int(l2)
        J = int(J)
    except:
        return ''

    # Only apply parity splitting if l2 != 0 (i.e. in degenerate bending mode)
    if l2 == 0:
        return ''

    # For bending modes, use rule based on J
    if J % 2 == 0:
        return 'e'
    else:
        return 'f'


def calculate_J(row, branches):
    """
    Calculate the rotational quantum number J based on the branch type.
    Parameters:
    row (dict): A dictionary containing the spectral line data. It must have keys 'Q\'\'' and 'J\'\''.
    Returns:
    int: The calculated rotational quantum number J.
    Notes:
    - The function assumes that 'J\'' is a string where the first character indicates the branch type.
    - The branch types are 'O', 'P', 'Q', 'R', and 'S'.
    - The function adjusts the value of 'J\'\'' based on the branch type:
        - 'O': J = J'' - 2
        - 'P': J = J'' - 1
        - 'Q': J = J''
        - 'R': J = J'' + 1
        - 'S': J = J'' + 2
    - If the branch type is not recognized, the function returns the value of 'J\'\''.
    """

    if row['J\''][0] == 'O':
        branches.append("O")
        return int(row['J\'\'']) - 2
    elif row['J\''][0] == 'P':
        branches.append("P")
        return int(row['J\'\'']) - 1
    elif row['J\''][0] == 'Q':
        branches.append("Q")
        return int(row['J\'\''])
    elif row['J\''][0] == 'R':
        branches.append("R")
        return int(row['J\'\'']) + 1
    elif row['J\''][0] == 'S':
        branches.append("S")
        return int(row['J\'\'']) + 2
    else:
        return int(row['J\'\''])


def single_isotopologue(df_all, isotopologue_number, branches=False):
    """
    Filters a DataFrame to include only rows corresponding to a specific isotopologue and processes the data.
    Args:
        df_all (pd.DataFrame): The full DataFrame containing all isotopologues.
        isotopologue_number (int): The isotopologue number to filter by.
        branches (bool, optional): A flag to indicate whether to print branches. Defaults to False.
    Returns:
        pd.DataFrame: A DataFrame containing the filtered and processed isotopologue data.
    """

    branches = []

    # Filter full df to just isotopologue required & create copy
    df_isotopologue = pd.DataFrame()
    df_isotopologue = df_all.loc[df_all['I'] == isotopologue_number].copy()

    # Split Q'' into correct columns
    df_isotopologue['J\''] = df_isotopologue['Q\'\''].str.extract(
        r'([A-Za-z]+)', expand=False)
    df_isotopologue['J\'\''] = df_isotopologue['Q\'\''].str.extract(
        r'(\d+)', expand=False).astype(int)

    # See calculate_J function doc
    # for index, row in df_isotopologue.iterrows():
    df_isotopologue['J\''] = df_isotopologue.apply(
        lambda row: calculate_J(row, branches), axis=1)

    # Drop unnecessary columns for this work
    df_isotopologue.drop(columns=['M', 'I', 'gamma_air', 'gamma_self', 'n_air',
                         'sigma_air', 'I_error', 'I_ref', 'flag', 'Q\'', 'Q\'\''], inplace=True)

    # Reorder columns for legibility
    df_isotopologue = df_isotopologue[[
        "v", "S", "A", "E\'\'", "V\'", "V\'\'", "J\'", "J\'\'", "g\'", "g\'\'"]]

    if branches == True:
        print("Branches:")
        o_count = sum(1 for item in branches if item == "O")
        print(f"\tO = {o_count}")
        p_count = sum(1 for item in branches if item == "P")
        print(f"\tP = {p_count}")
        q_count = sum(1 for item in branches if item == "Q")
        print(f"\tQ = {q_count}")
        r_count = sum(1 for item in branches if item == "R")
        print(f"\tR = {r_count}")
        s_count = sum(1 for item in branches if item == "S")
        print(f"\tS = {s_count}")

    # Split V' and V'' into quantum numbers
    df_isotopologue[['v1\'', 'v2\'', 'l2\'', 'v3\'']
                    ] = df_isotopologue['V\''].str.split(expand=True)
    df_isotopologue[['v1\'\'', 'v2\'\'', 'l2\'\'', 'v3\'\'']
                    ] = df_isotopologue['V\'\''].str.split(expand=True)

    # Drop the original V' and V'' columns
    df_isotopologue.drop(columns=['V\'', 'V\'\''], inplace=True)

    # Add parity columns based on J and l2
    df_isotopologue['parity\''] = df_isotopologue.apply(
        lambda row: infer_parity(row['J\''], row['l2\'']), axis=1)

    df_isotopologue['parity\'\''] = df_isotopologue.apply(
        lambda row: infer_parity(row['J\'\''], row['l2\'\'']), axis=1)

    # Reorder columns for legibility
    df_isotopologue = df_isotopologue[[
        "v", "S", "A", "E\'\'",
        "v1\'", "v2\'", "l2\'", "v3\'",
        "v1\'\'", "v2\'\'", "l2\'\'", "v3\'\'",
        "J\'", "parity\'", "J\'\'", "parity\'\'", "g\'", "g\'\'"]]

    return df_isotopologue
