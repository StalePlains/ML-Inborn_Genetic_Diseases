import pandas as pd
import numpy as np
import openpyxl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_excel("C:\\Users\\inspi\\Desktop\\Research\\vcf_data.xlsx")


# REF = Reference sequence, ORIGIN = Origin of phenotype (e.g. parental), AF_* = Allele Frequency, CHROM = Chromosome, CLINSIG = Benign, Likely_Benign, Pathogenic, CLNDN = CLinical Disease Name, POS = Position, ALT = Alternate Sequence, CLNVC = Variant Type
desired_columns = ['REF', "ORIGIN", "AF_ESP", "AF_TGP", "CHROM", "CLNSIG", "CLNDN", "AF_EXAC", "POS", "ALT", "CLNVC"]
df = df[desired_columns]

variant_types = ['single_nucleotide_variant', 'Microsatellite', 'Indel', 'Deletion', 'Duplication', 'Insertion', 'Variation', 'Inversion']

variant_type_mapping = {
    'single_nucleotide_variant': 0,
    'Microsatellite': 1,
    'Indel': 2,
    'Deletion': 3,
    'Duplication': 4,
    'Insertion': 5,
    'Variation': 6,
    'Inversion': 7
}

# Map the variant types to their numerical labels using the defined mapping
df['CLNVC'] = df['CLNVC'].map(variant_type_mapping)

# df = df[((df['CLNSIG'].isin(['Benign', 'Likely_Benign'])) & (df['CLNDN'].isin(['not_provided']))) | ((df['CLNSIG'].isin(['Likely_pathogenic', 'Pathogenic'])) & (df['CLNDN'].isin(['Inborn_genetic_diseases'])))]
df = df[(df['CLNDN']=='Inborn_genetic_diseases') & ((df['CLNSIG'] == 'Pathogenic')) | (((df['CLNDN']=='not_provided') & (df['CLNSIG'] == 'Benign')) | ((df['CLNSIG'] == 'Likely_Benign') & (df['CLNDN'] == 'not_provided')))]

df = df[(df['CLNDN'] == 'Inborn_genetic_diseases') | (df['CLNDN'] == 'not_provided')]

minority_class = df[df['CLNDN'] == 'Inborn_genetic_diseases']
majority_class = df[df['CLNDN'] == 'not_provided']

print(len(majority_class))
print(len(minority_class))

# # Perform a left join between the entire database and your new sample
# merged_df = pd.merge(df, undersampled_majority, how='left', indicator=True)

# # Filter for rows where the join indicator is 'left_only' (i.e., rows that are in the entire database but not in the new sample)
# unseen_rows = merged_df[merged_df['_merge'] == 'left_only']

# # Drop the indicator column
# unseen_rows = unseen_rows.drop(columns=['_merge'])

# df = pd.concat([undersampled_majority, undersampled_minority])

# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

mapping = {'Inborn_genetic_diseases': 1, 'not_provided': 0}

# Apply the mapping to the 'CLNDN' column
df['CLNDN_ENC'] = df['CLNDN'].map(mapping)
# unseen_rows['CLNDN_ENC'] = df['CLNDN'].map(mapping)

df = df.drop(columns=['CLNDN'])
#unseen_rows = unseen_rows.drop(columns=['CLNDN'])

df['LENGTH'] = abs(df['REF'].str.len() - df['ALT'].str.len())
#unseen_rows['LENGTH'] = abs(unseen_rows['REF'].str.len() - unseen_rows['ALT'].str.len())


df['REF_A'] = df['REF'].str.count('A')
df['REF_T'] = df['REF'].str.count('T')
df['REF_C'] = df['REF'].str.count('C')
df['REF_G'] = df['REF'].str.count('G')

#unseen_rows['REF_A'] = unseen_rows['REF'].str.count('A')
#unseen_rows['REF_T'] = unseen_rows['REF'].str.count('T')
##unseen_rows['REF_C'] = unseen_rows['REF'].str.count('C')
#unseen_rows['REF_G'] = unseen_rows['REF'].str.count('G')

# One-hot encode the counts for each nucleotide
#one_hot_df = pd.get_dummies(df[['A_count', 'T_count', 'C_count', 'G_count']], prefix='', prefix_sep='')

#df = pd.concat([df, one_hot_df], axis=1)

# Drop the original REF column
df.drop('REF', axis=1, inplace=True)

df['ALT_A'] = df['ALT'].str.count('A')
df['ALT_T'] = df['ALT'].str.count('T')
df['ALT_C'] = df['ALT'].str.count('C')
df['ALT_G'] = df['ALT'].str.count('G')

#unseen_rows.drop('REF', axis=1, inplace=True)

#unseen_rows['ALT_A'] = unseen_rows['ALT'].str.count('A')
#unseen_rows['ALT_T'] = unseen_rows['ALT'].str.count('T')
#unseen_rows['ALT_C'] = unseen_rows['ALT'].str.count('C')
#unseen_rows['ALT_G'] = unseen_rows['ALT'].str.count('G')

# One-hot encode the counts for each nucleotide
#one_hot_df = pd.get_dummies(df[['A_count', 'T_count', 'C_count', 'G_count']], prefix='', prefix_sep='')

#df = pd.concat([df, one_hot_df], axis=1)

# Drop the original REF column
df.drop('ALT', axis=1, inplace=True)
#unseen_rows.drop('ALT', axis=1, inplace=True)

label_encoder = LabelEncoder()

# Fit and transform the 'CLNSIG' column
df['CLNSIG_ENC'] = label_encoder.fit_transform(df['CLNSIG'])
df = df.drop(columns=['CLNSIG'])

#unseen_rows['CLNSIG_ENC'] = label_encoder.fit_transform(unseen_rows['CLNSIG'])
#unseen_rows = unseen_rows.drop(columns=['CLNSIG'])

# Assuming 'df' is your DataFrame and 'POS' is the column containing POS values
pos_values = df['POS'].values.reshape(-1, 1)  # Reshape to 2D array as required by MinMaxScaler
scaler = MinMaxScaler()
normalized_pos = scaler.fit_transform(pos_values)

# Replace the original POS column with the normalized values
df['POS'] = normalized_pos

#pos_values = unseen_rows['POS'].values.reshape(-1, 1)  # Reshape to 2D array as required by MinMaxScaler
#scaler = MinMaxScaler()
#normalized_pos = scaler.fit_transform(pos_values)

#unseen_rows['POS'] = normalized_pos

df.drop('CHROM', axis=1, inplace=True)
#unseen_rows.drop('CHROM', axis=1, inplace=True)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data and transform it
scaled_lengths = scaler.fit_transform(df[['LENGTH']])

# Replace the original LENGTH column with the scaled values
df['LENGTH'] = scaled_lengths

#scaled_lengths = scaler.fit_transform(unseen_rows[['LENGTH']])

# Replace the original LENGTH column with the scaled values
#unseen_rows['LENGTH'] = scaled_lengths

# unseen_rows.drop(['ORIGIN_16.0'])
# unseen_rows.drop(['ORIGIN_5.0'])

df.to_excel("final12unbalanced.xlsx", index=False)
#unseen_rows.to_excel('Testing3noorigin.xlsx', index=False)