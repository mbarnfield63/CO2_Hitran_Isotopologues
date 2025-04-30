import pandas as pd


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

    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # Store the full raw lines
    format_spec = [('M', 2), ('I', 1), ('v', 12), ('S', 10), ('A', 10),
                   ('gamma_air', 5), ('gamma_self', 5),
                   ("E''", 10), ('n_air', 4),
                   ('sigma_air', 8), ("V'", 15), ("V''", 15),
                   ("Q'", 15), ("Q''", 15),
                   ('I_error', 6), ('I_ref', 12),
                   ('flag', 1), ("g'", 7), ("g''", 7)]

    colspecs = [(sum([w for _, w in format_spec[:i]]),
                 sum([w for _, w in format_spec[:i+1]])) for i in range(len(format_spec))]

    hitran_all = pd.read_fwf(filepath, colspecs=colspecs, names=[
                             name for name, _ in format_spec])
    hitran_all.insert(0, 'raw', lines)

    return hitran_all


def infer_parity(df_iso):
    """
    Infer the lower state parity based on upper state parity and selection rules.
    For CO2, the rules depend on the l2 value and branch type.
    """
    # Convert l2 to integer:
    try:
        l2_upper = int(df_iso["l2'"])
    except ValueError:
        raise ValueError(f"Invalid l2' value: {df_iso["l2'"]}")

    # Get upper state parity
    upper_parity = df_iso["parity'"]
    if not upper_parity:
        raise ValueError(
            f"Upper state parity is not defined. In row:\n{df_iso}")

    # Determine lower state parity based on selection rules
    if l2_upper == 0:  # Σ-Σ transitions
        if df_iso["branch"] in ['P', 'R']:  # ΔJ = ±1
            # Parity must change
            lower_parity = 'f' if upper_parity == 'e' else 'e'
        else:  # Q branch (ΔJ = 0)
            # Parity must not change
            lower_parity = upper_parity
    else:  # l2 > 0 (transitions involving Π or higher states)
        if df_iso["branch"] in ['P', 'R']:  # ΔJ = ±1
            # Parity must not change
            lower_parity = upper_parity
        else:  # Q branch (ΔJ = 0)
            # Parity must change
            lower_parity = 'f' if upper_parity == 'e' else 'e'

    return lower_parity


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

    # Extract branch, J'', and parity from raw line (e.g. "P 51e")
    df_iso["branch"] = df_iso["raw"].str.extract(
        r'\b([OPQRS])\s*\d+[ef]?', expand=False)
    df_iso["J''"] = df_iso["raw"].str.extract(
        r'\b[OPQRS]\s*(\d+)[ef]?', expand=False).astype(int)
    df_iso["parity'"] = df_iso["raw"].str.extract(
        r'\b[OPQRS]\s*\d+([ef])', expand=False).fillna('')

    # Calculate J' based on branch
    def calc_Jprime(branch, Jpp):
        branch_list.append(branch)
        return {
            "O": Jpp - 2,
            "P": Jpp - 1,
            "Q": Jpp,
            "R": Jpp + 1,
            "S": Jpp + 2
        }.get(branch, Jpp)

    df_iso["J'"] = df_iso.apply(lambda row: calc_Jprime(
        row["branch"], row["J''"]), axis=1)

    # Drop now-unnecessary columns
    df_iso.drop(columns=["M", "I", "gamma_air", "gamma_self", "n_air",
                         "sigma_air", "I_error", "I_ref", "flag", "Q'", "Q''", "raw"], inplace=True)

    # Reorder
    df_iso = df_iso[["v", "S", "A", "E''", "V'",
                     "V''", "J'", "J''", "g'", "g''", "parity'", "branch"]]

    # Remove polyad rows from the dataframe
    df_iso = df_iso[~df_iso["V'"].str.contains('-2-2-2-20')]
    df_iso = df_iso[~df_iso["V''"].str.contains('-2-2-2-20')]

    # Split V' and V'' into quantum numbers
    df_iso[["v1'", "v2'", "l2'", "v3'"]
           ] = df_iso["V'"].str.split(expand=True)
    df_iso[["v1''", "v2''", "l2''", "v3''"]
           ] = df_iso["V''"].str.split(expand=True)

    # Convert quantum numbers to integers
    df_iso[["v1'", "v2'", "l2'", "v3'",
            "v1''", "v2''", "l2''", "v3''"]] = df_iso[["v1'", "v2'", "l2'", "v3'",
                                                       "v1''", "v2''", "l2''", "v3''"]].astype(int)
    # Convert J' and J'' to integers
    df_iso["J'"] = df_iso["J'"].astype(int)
    df_iso["J''"] = df_iso["J''"].astype(int)

    # Apply the parity inference function
    df_iso["parity''"] = df_iso.apply(infer_parity, axis=1)

    # Drop the branch column & V' and V'' columns
    df_iso.drop(columns=["V'", "V''", "branch"], inplace=True)

    # Reorder for clarity
    df_iso = df_iso[[
        "v", "S", "A", "E''",
        "v1'", "v2'", "l2'", "v3'",
        "v1''", "v2''", "l2''", "v3''",
        "J'", "parity'", "J''", "parity''", "g'", "g''"
    ]]

    # Optional: print branch counts
    if branches:
        print("Branches:")
        for b in ['O', 'P', 'Q', 'R', 'S']:
            print(f"\t{b} = {branch_list.count(b)}")

    return df_iso
