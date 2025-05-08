import pandas as pd


import pandas as pd


def hitran_to_dataframe(filepath):
    """
    Converts a 2004 HITRAN .par file to a pandas DataFrame.

    Parameters:
    filepath (str): The path to the HITRAN formatted file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the HITRAN file with columns:
        - 'M': Integer, 2 characters
        - 'I': Integer, 1 character
        - 'v': Float, 12 characters
        - 'S': Float (Exponential), 10 characters
        - 'A': Float (Exponential), 10 characters
        - 'gamma_air': Float, 5 characters
        - 'gamma_self': Float, 5 characters
        - 'E\'\'': Float, 10 characters
        - 'n_air': Float, 4 characters
        - 'sigma_air': Float, 8 characters
        - 'V'_blank6': String, 6 characters (unused)
        - 'v1\'': Integer, 2 characters
        - 'v2\'': Integer, 2 characters
        - 'l2\'': Integer, 2 characters
        - 'v3\'': Integer, 2 characters
        - 'r\'': Integer, 1 character
        - 'V\'\'_blank6': String, 6 characters (unused)
        - 'v1\'\'': Integer, 2 characters
        - 'v2\'\'': Integer, 2 characters
        - 'l2\'\'': Integer, 2 characters
        - 'v3\'\'': Integer, 2 characters
        - 'r\'\'': Integer, 1 character
        - 'Q\'_blank10': String, 10 characters (unused)
        - 'F\'': Integer, 5 characters
        - 'Q\'\'_blank5': String, 5 characters (unused)
        - 'Branch': String, 1 character
        - 'J\'\'': Integer, 3 characters
        - 'sym\'\'': String, 2 characters
        - 'F\'\'': Integer, 5 characters
        - 'I_error1' through 'I_error6': Integer, 1 character each
        - 'I_ref1' through 'I_ref6': Integer, 2 characters each
        - 'flag': String, 1 character
        - 'g\'': Float, 7 characters
        - 'g\'\'': Float, 7 characters
    """
    format_spec = [
        ('M', 2, int), ('I', 1, str), ('v', 12, float), ('S', 10, float),
        ('A', 10, float), ('gamma_air', 5, float), ('gamma_self', 5, float),
        ("E''", 10, float), ('n_air', 4, float), ('sigma_air', 8, float),
        ("V'_blank6", 6, str), ("v1'", 2, int), ("v2'", 2, int),
        ("l2'", 2, int), ("v3'", 2, int), ("r'", 1, int),
        ("V''_blank6", 6, str), ("v1''", 2, int), ("v2''", 2, int),
        ("l2''", 2, int), ("v3''", 2, int), ("r''", 1, int),
        ("Q'_blank10", 10, str), ("F'", 5, int), ("Q''_blank5", 5, str),
        ("Branch", 1, str), ("J''", 3, int), ("sym''", 2, str), ("F''", 5, int),
        ('I_error1', 1, int), ('I_error2', 1, int), ('I_error3', 1, int),
        ('I_error4', 1, int), ('I_error5', 1, int), ('I_error6', 1, int),
        ('I_ref1', 2, int), ('I_ref2', 2, int), ('I_ref3', 2, int),
        ('I_ref4', 2, int), ('I_ref5', 2, int), ('I_ref6', 2, int),
        ('flag', 1, str), ("g'", 7, float), ("g''", 7, float)
    ]

    colspecs = []
    start = 0
    for _, width, _ in format_spec:
        colspecs.append((start, start + width))
        start += width

    col_names = [name for name, _, _ in format_spec]
    col_types = {name: dtype for name, _, dtype in format_spec}

    df = pd.read_fwf(filepath, colspecs=colspecs, names=col_names)

    # Enforce column types
    for col, dtype in col_types.items():
        # avoids failure on bad values
        df[col] = df[col].astype(dtype, errors='ignore')

    return df


def infer_parity(df_iso):
    J_lower = int(df_iso["J''"])
    sym_lower = df_iso["sym''"]

    if J_lower % 2 == 0:
        return "f" if sym_lower == "e" else "e"
    else:
        return sym_lower


def calc_J_upper(branch, J_lower):
    """
    Calculate the upper state J value based on the branch and lower state J value.
    For CO2, the rules depend on the branch type.
    """
    # Convert J_lower to integer:
    try:
        J_lower = int(J_lower)
    except ValueError:
        raise ValueError(f"Invalid J'' value: {J_lower}")

    # Determine upper state J based on selection rules
    if branch == "O":
        return J_lower - 2
    elif branch == "P":
        return J_lower - 1
    elif branch == "Q":
        return J_lower
    elif branch == "R":
        return J_lower + 1
    elif branch == "S":
        return J_lower + 2
    else:
        raise ValueError(f"Invalid branch: {branch}")


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

    branch_list = []

    # Filter for specific isotopologue
    df_iso = df_all[df_all["I"] == isotopologue_number].copy()

    # Remove polyad rows from the dataframe
    df_iso = df_iso[~df_iso["v1'"] == -2]
    df_iso = df_iso[~df_iso["v1''"] == -2]

    df_iso["J'"] = df_iso.apply(lambda row: calc_J_upper(
        row["Branch"], row["J''"]), axis=1)

    # Apply the parity inference function
    df_iso["sym'"] = df_iso.apply(infer_parity, axis=1)

    # Optional: print branch counts
    if branches:
        print(f"Branches in iso {isotopologue_number}:")
        for b in ["O", "P", "Q", "R", "S"]:
            print(f"\t{b} = {branch_list.count(b)}")

    return df_iso
