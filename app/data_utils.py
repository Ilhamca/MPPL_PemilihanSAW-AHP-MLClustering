import pandas as pd
try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = 'rapidfuzz'
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_AVAILABLE = 'fuzzywuzzy'
    except ImportError:
        from difflib import SequenceMatcher
        FUZZY_AVAILABLE = 'difflib'


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column whose name starts with 'Unnamed'"""
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def fuzzy_match_cpu(cpu_name: str, cpu_list: list, threshold: int = 60):
    """Find the best fuzzy match for a CPU name from a list.
    
    Uses multiple matching strategies:
    1. Exact model number match (highest priority)
    2. Token sort ratio matching (handles word order)
    3. Partial ratio matching (handles substrings)
    
    Args:
        cpu_name: The CPU name to match
        cpu_list: List of available CPU names
        threshold: Minimum similarity score (0-100) to consider a match
        
    Returns:
        Matched CPU name or None if no good match found
    """
    if not cpu_name or pd.isna(cpu_name):
        return None
    
    cpu_name_str = str(cpu_name).strip()
    
    # Extract model number if present (e.g., "7200U", "8250U", "9420")
    import re
    model_pattern = r'\b(\d{4,5}[A-Z]{0,2})\b'
    model_match = re.search(model_pattern, cpu_name_str)
    
    # If we have a specific model number, prioritize exact model matches
    if model_match:
        model_num = model_match.group(1)
        # Find candidates with the same model number
        model_candidates = [cpu for cpu in cpu_list if model_num in cpu]
        if model_candidates:
            # Use fuzzy matching on the filtered list
            if FUZZY_AVAILABLE == 'rapidfuzz' or FUZZY_AVAILABLE == 'fuzzywuzzy':
                result = process.extractOne(cpu_name_str, model_candidates, scorer=fuzz.token_sort_ratio)
                if result and result[1] >= threshold:
                    return result[0]
            else:
                best_match = None
                best_ratio = 0
                for candidate in model_candidates:
                    ratio = SequenceMatcher(None, cpu_name_str.lower(), str(candidate).lower()).ratio() * 100
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = candidate
                if best_ratio >= threshold:
                    return best_match
    
    # Fallback to general fuzzy matching
    if FUZZY_AVAILABLE == 'rapidfuzz' or FUZZY_AVAILABLE == 'fuzzywuzzy':
        # Try token_sort_ratio first (best for different word orders)
        result = process.extractOne(cpu_name_str, cpu_list, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return result[0]
        
        # Try partial_ratio as fallback (good for substrings)
        result = process.extractOne(cpu_name_str, cpu_list, scorer=fuzz.partial_ratio)
        if result and result[1] >= threshold + 10:  # Higher threshold for partial
            return result[0]
    else:
        # Fallback to difflib
        best_match = None
        best_ratio = 0
        for candidate in cpu_list:
            ratio = SequenceMatcher(None, cpu_name_str.lower(), str(candidate).lower()).ratio() * 100
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        
        if best_ratio >= threshold:
            return best_match
    
    return None


def fuzzy_match_gpu(gpu_name: str, gpu_list: list, threshold: int = 60):
    """Find the best fuzzy match for a GPU name from a list.
    
    Uses multiple matching strategies optimized for GPU naming patterns:
    1. Exact model number match (e.g., GTX 1050, MX150, HD 620)
    2. Token sort ratio matching (handles word order)
    3. Partial ratio matching (handles substrings like "GeForce" prefix)
    
    Args:
        gpu_name: The GPU name to match
        gpu_list: List of available GPU names
        threshold: Minimum similarity score (0-100) to consider a match
        
    Returns:
        Matched GPU name or None if no good match found
    """
    if not gpu_name or pd.isna(gpu_name):
        return None
    
    gpu_name_str = str(gpu_name).strip()
    
    # Extract GPU model patterns (GTX/RTX/MX series, HD/UHD, Radeon series, etc.)
    import re
    # Pattern for Nvidia: GTX/RTX + 3-4 digits, MX + 3 digits, or just model numbers
    # Pattern for Intel: HD/UHD + 3-4 digits
    # Pattern for AMD: Radeon with various suffixes
    model_patterns = [
        r'\b(GTX\s*\d{3,4}[A-Z]*)\b',  # GTX 1050, GTX 1050 Ti
        r'\b(RTX\s*\d{3,4}[A-Z]*)\b',  # RTX 2060, RTX 3070
        r'\b(MX\s*\d{3})\b',            # MX150, MX250
        r'\b([UH]HD\s*Graphics\s*\d{3,4})\b',  # UHD Graphics 620, HD Graphics 620
        r'\b(Radeon\s*\w+\s*\d{3,4}[A-Z]*)\b',  # Radeon RX 580, Radeon R5
        r'\b(\d{3,4}[A-Z]{1,3})\b'     # Generic: 940MX, 1060, etc.
    ]
    
    model_match = None
    for pattern in model_patterns:
        match = re.search(pattern, gpu_name_str, re.IGNORECASE)
        if match:
            model_match = match.group(1)
            break
    
    # If we have a specific model pattern, prioritize matches containing it
    if model_match:
        # Normalize the model for searching (case-insensitive)
        model_lower = model_match.lower()
        model_candidates = [gpu for gpu in gpu_list if model_lower in gpu.lower()]
        
        if model_candidates:
            # Use fuzzy matching on the filtered list
            if FUZZY_AVAILABLE == 'rapidfuzz' or FUZZY_AVAILABLE == 'fuzzywuzzy':
                result = process.extractOne(gpu_name_str, model_candidates, scorer=fuzz.token_sort_ratio)
                if result and result[1] >= threshold:
                    return result[0]
            else:
                best_match = None
                best_ratio = 0
                for candidate in model_candidates:
                    ratio = SequenceMatcher(None, gpu_name_str.lower(), str(candidate).lower()).ratio() * 100
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = candidate
                if best_ratio >= threshold:
                    return best_match
    
    # Fallback to general fuzzy matching
    if FUZZY_AVAILABLE == 'rapidfuzz' or FUZZY_AVAILABLE == 'fuzzywuzzy':
        # Try token_sort_ratio first (best for different word orders)
        result = process.extractOne(gpu_name_str, gpu_list, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return result[0]
        
        # Try partial_ratio as fallback (good for substrings)
        result = process.extractOne(gpu_name_str, gpu_list, scorer=fuzz.partial_ratio)
        if result and result[1] >= threshold + 10:  # Higher threshold for partial
            return result[0]
    else:
        # Fallback to difflib
        best_match = None
        best_ratio = 0
        for candidate in gpu_list:
            ratio = SequenceMatcher(None, gpu_name_str.lower(), str(candidate).lower()).ratio() * 100
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        
        if best_ratio >= threshold:
            return best_match
    
    return None


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
            df['Ram'] = df['Ram'].astype(int)
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

    # CPU mapping with fuzzy search: use cpu_data.csv only
    if 'Cpu' in df.columns:
        try:
            cpu_scores = pd.read_csv('csv/cpu_data.csv')
            cpu_name_col = 'Nama CPU'
            cpu_mark_col = 'CPU Mark'
            
            if cpu_name_col in cpu_scores.columns and cpu_mark_col in cpu_scores.columns:
                # Clean CPU Mark column - remove commas and convert to numeric
                cpu_scores[cpu_mark_col] = cpu_scores[cpu_mark_col].astype(str).str.replace(',', '').str.strip()
                cpu_scores[cpu_mark_col] = pd.to_numeric(cpu_scores[cpu_mark_col], errors='coerce')
                
                # Normalize CPU names for better matching
                cpu_scores['Normalized_CPU'] = cpu_scores[cpu_name_col].astype(str).str.strip()
                
                # Get list of available CPU names
                cpu_list = cpu_scores['Normalized_CPU'].dropna().tolist()
                
                # Initialize Prosesor_Score column
                df['Prosesor_Score'] = pd.NA
                
                # Apply fuzzy matching for each CPU
                for idx, cpu in df['Cpu'].items():
                    if pd.notna(cpu):
                        cpu_clean = str(cpu).strip()
                        
                        # Try exact match first (case-insensitive)
                        exact_match = cpu_scores[cpu_scores['Normalized_CPU'].str.lower() == cpu_clean.lower()]
                        if not exact_match.empty:
                            score_val = exact_match.iloc[0][cpu_mark_col]
                            if pd.notna(score_val):
                                df.at[idx, 'Prosesor_Score'] = score_val
                        else:
                            # Use fuzzy matching with lower threshold for better matches
                            matched_cpu = fuzzy_match_cpu(cpu_clean, cpu_list, threshold=60)
                            if matched_cpu:
                                matched_row = cpu_scores[cpu_scores['Normalized_CPU'] == matched_cpu]
                                if not matched_row.empty:
                                    score_val = matched_row.iloc[0][cpu_mark_col]
                                    if pd.notna(score_val):
                                        df.at[idx, 'Prosesor_Score'] = score_val
                
                # Prosesor_Score should already be numeric from the cleaned cpu_scores
                df['Prosesor_Score'] = pd.to_numeric(df['Prosesor_Score'], errors='coerce')
                
        except FileNotFoundError:
            # no cpu mapping available
            pass
        except Exception as e:
            # silent fail but could log error if needed
            import traceback
            print(f"CPU mapping error: {e}")
            traceback.print_exc()

    # GPU mapping with fuzzy search: use gpu_data.csv
    if 'Gpu' in df.columns:
        try:
            gpu_scores = pd.read_csv('csv/gpu_data.csv')
            gpu_name_col = 'Nama GPU'
            gpu_mark_col = 'GPU Mark'
            
            if gpu_name_col in gpu_scores.columns and gpu_mark_col in gpu_scores.columns:
                # Clean GPU Mark column - remove commas and convert to numeric
                gpu_scores[gpu_mark_col] = gpu_scores[gpu_mark_col].astype(str).str.replace(',', '').str.strip()
                gpu_scores[gpu_mark_col] = pd.to_numeric(gpu_scores[gpu_mark_col], errors='coerce')
                
                # Normalize GPU names for better matching
                gpu_scores['Normalized_GPU'] = gpu_scores[gpu_name_col].astype(str).str.strip()
                
                # Get list of available GPU names
                gpu_list = gpu_scores['Normalized_GPU'].dropna().tolist()
                
                # Initialize GPU_Score column
                df['GPU_Score'] = pd.NA
                
                # Apply fuzzy matching for each GPU
                for idx, gpu in df['Gpu'].items():
                    if pd.notna(gpu):
                        gpu_clean = str(gpu).strip()
                        
                        # Try exact match first (case-insensitive)
                        exact_match = gpu_scores[gpu_scores['Normalized_GPU'].str.lower() == gpu_clean.lower()]
                        if not exact_match.empty:
                            score_val = exact_match.iloc[0][gpu_mark_col]
                            if pd.notna(score_val):
                                df.at[idx, 'GPU_Score'] = score_val
                        else:
                            # Use fuzzy matching with GPU-specific algorithm
                            # Try with threshold 60 first, then 50 for edge cases
                            matched_gpu = fuzzy_match_gpu(gpu_clean, gpu_list, threshold=60)
                            if not matched_gpu:
                                matched_gpu = fuzzy_match_gpu(gpu_clean, gpu_list, threshold=50)
                            if matched_gpu:
                                matched_row = gpu_scores[gpu_scores['Normalized_GPU'] == matched_gpu]
                                if not matched_row.empty:
                                    score_val = matched_row.iloc[0][gpu_mark_col]
                                    if pd.notna(score_val):
                                        df.at[idx, 'GPU_Score'] = score_val
                
                # GPU_Score should already be numeric from the cleaned gpu_scores
                df['GPU_Score'] = pd.to_numeric(df['GPU_Score'], errors='coerce')
                
        except FileNotFoundError:
            # no gpu mapping available
            pass
        except Exception as e:
            # silent fail but could log error if needed
            import traceback
            print(f"GPU mapping error: {e}")
            traceback.print_exc()

    return df
