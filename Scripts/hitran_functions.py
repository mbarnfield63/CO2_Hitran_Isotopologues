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
        - 'I_error': Integer, 6 characters (6 individual integers)
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


def split_co2_quantum_label(label):
    label = str(label).ljust(15)
    return {
        "v1": label[6:8].strip(),
        "v2": label[8:10].strip(),
        "l2": label[10:12].strip(),
        "v3": label[12:14].strip(),
        "r":  label[14:15].strip()
    }


def split_co2_q_label(label):
    label = str(label).ljust(15)
    if label[0:1] == 'Q':
        return {
            "F'":      label[5:10].strip()
        }
    return {
        "branch": label[5:6].strip(),
        "J''":      label[6:9].strip(),
        "sym''":    label[9:10].strip(),
        "F''":      label[10:15].strip()
    }


def expand_hitran_fields(df):
    # Expand V' and V''
    for field in ["V'", "V''"]:
        label_split = df[field].apply(split_co2_quantum_label)
        suffix = "'" if field == "V'" else "''"
        label_df = pd.DataFrame(label_split.tolist())
        label_df.columns = [f"{col}{suffix}" for col in label_df.columns]
        df = pd.concat([df, label_df], axis=1)
    # Drop original V' and V''
    df.drop(columns=["V'", "V''"], inplace=True)
    # Convert quantum numbers to integers
    df[["v1'", "v2'", "l2'", "v3'", "r'"]] = df[[
        "v1'", "v2'", "l2'", "v3'", "r''"]].astype(int)

    # Expand Q' and Q'' without "_Q" in suffix
    for field in ["Q'", "Q''"]:
        label_split = df[field].apply(split_co2_q_label)
        suffix = "'" if field == "Q'" else "''"
        label_df = pd.DataFrame(label_split.tolist())
        label_df.columns = [f"{col}{suffix}" for col in label_df.columns]
        df = pd.concat([df, label_df], axis=1)
    # Drop original Q' and Q''
    df.drop(columns=["Q'", "Q''"], inplace=True)
    # Convert J to integers
    df["J''"] = df["J''"].astype(int)

    # Split I_error (6I1)
    df['I_error'] = df['I_error'].astype(str).str.zfill(6)
    ierror_df = df['I_error'].apply(lambda x: pd.Series([int(d) for d in x]))
    ierror_df.columns = [f'I_error_{i+1}' for i in range(6)]
    df = pd.concat([df, ierror_df], axis=1)

    # Split I_ref (6I2 = 12 digits)
    df['I_ref'] = df['I_ref'].astype(str).str.zfill(12)
    iref_df = df['I_ref'].apply(lambda x: pd.Series(
        [int(x[i:i+2]) for i in range(0, 12, 2)]))
    iref_df.columns = [f'I_ref_{i+1}' for i in range(6)]
    df = pd.concat([df, iref_df], axis=1)

    return df


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

    # Get lower state parity
    lower_parity = df_iso["sym''"]
    if not lower_parity:
        raise ValueError(
            f"Lower state parity is not defined. In row:\n{df_iso}")

    # Determine lower state parity based on selection rules
    if l2_upper == 0:  # Σ-Σ transitions
        if df_iso["branch"] in ["P", "R"]:  # ΔJ = ±1
            # Parity must change
            upper_parity = "f" if lower_parity == "e" else "e"
        else:  # Q branch (ΔJ = 0)
            # Parity must not change
            upper_parity = lower_parity
    else:  # l2 > 0 (transitions involving Π or higher states)
        if df_iso["branch"] in ["P", "R"]:  # ΔJ = ±1
            # Parity must not change
            upper_parity = lower_parity
        else:  # Q branch (ΔJ = 0)
            # Parity must change
            upper_parity = "f" if lower_parity == "e" else "e"

    return upper_parity


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
    df_iso = df_iso[~df_iso["V'"].str.contains("-2-2-2-20")]
    df_iso = df_iso[~df_iso["V''"].str.contains("-2-2-2-20")]

    # Expand V' and V''
    df_iso = expand_hitran_fields(df_iso)

    df_iso["J'"] = df_iso.apply(lambda row: calc_J_upper(
        row["branch"], row["J''"]), axis=1)

    # Apply the parity inference function
    df_iso["sym'"] = df_iso.apply(infer_parity, axis=1)

    # Optional: print branch counts
    if branches:
        print(f"Branches in iso {isotopologue_number}:")
        for b in ["O", "P", "Q", "R", "S"]:
            print(f"\t{b} = {branch_list.count(b)}")

    return df_iso
