import pandas as pd
import numpy as np

vcf_df = pd.read_excel("filtered_vcf.xlsx")
bcf_df = pd.read_excel("filtered_bcf.xlsx")

vcf_df['LENGTH'] = abs(vcf_df['ALT'].str.len() - vcf_df['REF'].str.len())
bcf_df['LENGTH'] = abs(bcf_df['CONSENSUS'].str.len())

bcf_df['CLNDN'] = 'not_provided'
bcf_df['CLNSIG'] = 'Benign'

bcf_df['LOW_POS'] = bcf_df['CIPOS'].apply(lambda x: int(x.split(',')[0][1:]))  # Extracts the first value
bcf_df['HIGH_POS'] = bcf_df['CIPOS'].apply(lambda x: int(x.split(',')[1][:-1]))  # Extracts the second value

bcf_df.rename(columns={'CHR2': 'CHROM'}, inplace=True)

final_df = pd.concat([bcf_df, vcf_df], ignore_index=True)
final_df = final_df.drop(columns=['CIPOS', 'CIEND'], axis=1)
final_df = final_df.fillna('NaN')
test = final_df.head(10).to_string()

final_df.to_excel('merged_data.xlsx')
