import pandas as pd


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column whose name starts with 'Unnamed'"""
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def clean_laptops_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run cleaning and basic conversions on a laptops dataframe.

    Operations:
    - drop unnamed cols
    - drop rows with no data (dropna)
    - normalize Ram -> Ram (GB)
    - normalize Weight -> Weight (KG)
    - parse Memory into Memory (GB) and add Memory_Value (SSD=1/HDD=0)
    - map CPU to Prosesor_Score if a CPU benchmark CSV is present (silently skip if not)

    Returns cleaned copy of df (does not mutate input).
    """
    df = df.copy()
    # Drop unnamed
    df = drop_unnamed_columns(df)

    # Remove fully empty rows
    df = df.dropna(how='all')

    # RAM -> numeric
    if 'Ram' in df.columns:
        # keep original if missing
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).str.strip()
        df.loc[df['Ram'] == '', 'Ram'] = '0'
        try:
            df['Ram'] = df['Ram'].astype(float)
            df.rename(columns={'Ram': 'Ram (GB)'}, inplace=True)
        except Exception:
            # fallback: leave as-is
            pass

    # Weight -> numeric (kg)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).str.strip()
        df.loc[df['Weight'] == '', 'Weight'] = '0'
        try:
            df['Weight'] = df['Weight'].astype(float)
            df.rename(columns={'Weight': 'Weight (KG)'}, inplace=True)
        except Exception:
            pass

    # Memory parsing
    if 'Memory' in df.columns:
        # create Memory_Value: SSD -> 1 else 0
        df['Memory_Value'] = df['Memory'].astype(str).apply(lambda x: 1 if 'SSD' in x.upper() else 0)

        # normalize memory sizes into GB numbers stored as strings then float
        def mem_to_gb(val: str):
            s = str(val)
            # handle composites like "256GB SSD + 1TB HDD"
            parts = [p.strip() for p in s.split('+')]
            total = 0
            for p in parts:
                up = p.upper()
                # extract numeric
                num = ''.join([c for c in p if c.isdigit() or c=='.'])
                if num == '':
                    continue
                try:
                    n = float(num)
                except Exception:
                    continue
                if 'TB' in up:
                    total += n * 1024
                else:
                    total += n
            return total

        df['Memory'] = df['Memory'].astype(str).apply(lambda x: str(int(mem_to_gb(x))) if mem_to_gb(x) else '0')
        try:
            df['Memory'] = df['Memory'].astype(float)
            df.rename(columns={'Memory': 'Memory (GB)'}, inplace=True)
        except Exception:
            pass

    # CPU mapping: optional external CSV (cpu_benchmark_scores.csv)
    try:
        cpu_scores = pd.read_csv('csv/cpu_benchmark_scores.csv')
        if 'Cpu_Name' in cpu_scores.columns and 'Cpu_Mark' in cpu_scores.columns and 'Cpu' in df.columns:
            cpu_scores = cpu_scores.set_index('Cpu_Name')
            df['Prosesor_Score'] = df['Cpu'].map(cpu_scores['Cpu_Mark'])
    except FileNotFoundError:
        # no cpu mapping available
        pass
    except Exception:
        pass

    return df
