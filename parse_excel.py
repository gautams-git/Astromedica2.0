import pandas as pd
import json
import re
import os

def parse_diseases():
    # Fix the file path
    excel_file = "Manoj-Lord-Organs-Rashi-Nakshatra-Disease.xlsx"
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file {excel_file} not found.")

    # Skip header-less first rows and let pandas read
    df = pd.read_excel(excel_file, sheet_name='CONSTELLATION', header=None)
    
    # Structure from the image:
    # Row 1 (0-indexed): Nakshatra names (headers)
    # Row 2 (0-indexed): Degrees and Pada info (padas 1,2 or 3,4 etc.)
    # Row 3+ (0-indexed): Padas (sometimes) and then diseases
    
    # Wait, the screenshot shows:
    # Row 2: ASHWINI, BHARANI, KRUTHIKA... (this is row index 1 in dataframe)
    # Row 3: Degrees (this is row index 2)
    # Row 4+: Diseases (row index 3+)
    
    nakshatra_row = df.iloc[1]
    pada_info_row = df.iloc[2]
    
    current_nk = None
    results = {}
    
    # Number of columns can vary
    for col_idx in range(df.shape[1]):
        nk_val = nakshatra_row.iloc[col_idx]
        if pd.notna(nk_val):
            current_nk = str(nk_val).strip().capitalize()
            
        if not current_nk: continue
            
        info = str(pada_info_row.iloc[col_idx])
        padas = []
        # Parse padas from strings like "( 1st pada)" or "( 2,3 4th pada)" or "( 1st,2nd pada)"
        if '1st' in info or '1,' in info or '(1' in info: padas.append(1)
        if '2nd' in info or '2,' in info or '2 ' in info: padas.append(2)
        if '3rd' in info or '3,' in info or '3 ' in info: padas.append(3)
        if '4th' in info or '4,' in info or '4 ' in info or '4)' in info: padas.append(4)
        
        # If no pada found in Row 3, maybe it covers all padas (like ASHWINI column)
        if not padas:
            padas = [1, 2, 3, 4]
            
        diseases = []
        for row_idx in range(3, df.shape[0]):
            val = df.iloc[row_idx, col_idx]
            if pd.notna(val) and str(val).strip():
                # Some rows might be junk like "Ready" or "RASHIS", filter them
                d_str = str(val).strip()
                if d_str.lower() not in ['ready', 'rashis', 'constellation', 'planet', 'tridosha']:
                    diseases.append(d_str)
        
        if current_nk not in results:
            results[current_nk] = {1:[], 2:[], 3:[], 4:[]}
            
        for p in padas:
            # Merging with existing if the nakshatra was already seen in a previous column
            for d in diseases:
                if d not in results[current_nk][p]:
                    results[current_nk][p].append(d)

    return results

if __name__ == "__main__":
    try:
        data = parse_diseases()
        with open('parsed_constellation.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("Successfully extracted Nakshatra/Pada data to parsed_constellation.json")
    except Exception as e:
        print(f"Error parsing excel: {e}")
