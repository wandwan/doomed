import pandas as pd
import numpy as np

def convert_to_integer(df: pd.DataFrame) -> pd.DataFrame:
    # In the competition data, there are some columns that are integers, but are 
    # obfuscated as floats. This function will convert those columns to integers.

    int_denominators = {
     'AB': 0.004273,
     'AF': 0.00242,
     'AH': 0.008709,
     'AM': 0.003097,
     'AR': 0.005244,
     'AX': 0.008859,
     'AY': 0.000609,
     'AZ': 0.006302,
     'BC': 0.007028,
     'BD ': 0.00799,
     'BN': 0.3531,
     'BP': 0.004239,
     'BQ': 0.002605,
     'BR': 0.006049,
     'BZ': 0.004267,
     'CB': 0.009191,
     'CC': 6.12e-06,
     'CD ': 0.007928,
     'CF': 0.003041,
     'CH': 0.000398,
     'CL': 0.006365,
     'CR': 7.5e-05,
     'CS': 0.003487,
     'CU': 0.005517,
     'CW ': 9.2e-05,
     'DA': 0.00388,
     'DE': 0.004435,
     'DF': 0.000351,
     'DH': 0.002733,
     'DI': 0.003765,
     'DL': 0.00212,
     'DN': 0.003412,
     'DU': 0.0013794,
     'DV': 0.00259,
     'DY': 0.004492,
     'EB': 0.007068,
     'EE': 0.004031,
     'EG': 0.006025,
     'EH': 0.006084,
     'EL': 0.000429,
     'EP': 0.009269,
     'EU': 0.005064,
     'FC': 0.005712,
     'FD ': 0.005937,
     'FE': 0.007486,
     'FI': 0.005513,
     'FR': 0.00058,
     'FS': 0.006773,
     'GB': 0.009302,
     'GE': 0.004417,
     'GF': 0.004374,
     'GH': 0.003721,
     'GI': 0.002572
    }

    for k, v in int_denominators.items():
        df[k] = np.round(df[k]/v,1).astype(int)

    return df
