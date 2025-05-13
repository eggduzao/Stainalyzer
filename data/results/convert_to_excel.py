
# Import
import re, pandas as pd

# # Step 1: Normalize spaces -> tabs
# with open("extracted_scores.txt", "r") as f:
#     lines = [re.sub(r' +', '\t', line) for line in f]

# with open("extracted_scores.tsv", "w") as f:
#     f.writelines(lines)

# Step 2: TSV -> Excel
df = pd.read_csv("extracted_scores.tsv", sep='\t')
df.to_excel("extracted_scores.xlsx", index=False)

