import pysam as sam
import os
import pandas as pd
import numpy as np


working_dir = "/mnt/c/Users/inspi/desktop/research/ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage_SV/working/20190725_EMBL_Delly/Delly"
files_list = os.listdir(working_dir)

test = files_list[5]

vcf_file = f'/mnt/c/Users/inspi/desktop/research/clinvar.vcf/clinvar.vcf.gz'

file = f'{working_dir}/{test}'

bcf_in = sam.VariantFile(file)

vcf_in = sam.TabixFile(vcf_file)

all_fields = set()
count = 0
try:
    for rec in vcf_in.fetch():
    
        fields = rec.split('\t')
    
        chromosome = fields[0]
        position = fields[1]
        id_ = fields[2]  # Variant ID
        ref = fields[3]  # Reference allele
        alt = fields[4]  # Alternative allele
        qual = fields[5]  # Quality score
        filter_ = fields[6]  # Filter status
        info_field = fields[7]  # INFO field is the 8th field (0-based indexing)
        
        # Split the INFO field by semicolons to get individual INFO keys
        info_fields = info_field.split(';')
        info_keys = [info.split('=')[0] for info in info_fields if '=' in info]    
        list_fields = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER'] + info_keys
        all_fields.update(list_fields)
except UnicodeDecodeError:
    pass


print(all_fields)
columns = list(all_fields)

initial_values = {key: [] for key in columns}

df = pd.DataFrame(initial_values, columns=columns)

vcf_list = []

try:
    for rec in vcf_in.fetch():
        vcf_list.append(rec)
except UnicodeDecodeError:
    pass

for i in range(len(vcf_list)):
    rec = vcf_list[i]
    data = {}

    
    fields = rec.split('\t')

    data['CHROM'] = fields[0]
    data['POS'] = fields[1]
    data['ID'] = fields[2]
    data['REF'] = fields[3]
    data['ALT'] = fields[4]
    data['QUAL'] = fields[5]
    data['FILTER'] = fields[6]

    info_field = fields[7]

    info_fields = info_field.split(';')
    info_dict = {}

    for field in info_fields:
        key_value = field.split('=')

        if len(key_value) == 2:
            info_dict[key_value[0]] = key_value[1]
    
    prev_keys = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']
    for field in all_fields:

        if field in info_dict.keys():
            data[field] = info_dict[field]
        elif field in prev_keys:
            pass
        else:
            data[field] = None

    for key, value in data.items():
        if value is None:
            data[key] = np.nan
        if isinstance(value, list) or isinstance(value, tuple):
            data[key] = str(value)
    
    print(f'At Number {i} out of {len(vcf_list)}')
    df.loc[len(df)] = data
    

df.to_excel('vcf_data.xlsx', index=False)

#df.to_excel('clinvar_vcf.xlsx', index=False)

#columns = ['PRECISE', 'IMPRECISE', 'SVTYPE', 'SVMETHOD', 'CHR2', 'PE', 'MAPQ', 'CT', 'CIPOS', 'CIEND', 'SRMAPQ', 'INSLEN', 'HOMLEN', 'SR', 'SRQ', 'CONSENSUS', 'CE', 'GT', 'GL', 'GQ', 'FT', 'RCL', 'RC', 'RCR', 'CN', 'DR', 'DV', 'RR', 'RV', "FILTER", "PHENOTYPE"]

#initial_values = {key: [] for key in columns}

#df = pd.DataFrame(initial_values, columns=columns)

#all_info_fields = set()
#for record in bcf_in.fetch():
#    all_info_fields.update(record.info.keys())

#all_format_fields = ['GT', 'GL', 'GQ', 'FT', 'RCL', 'RC', 'RCR', 'CN', 'DR', 'DV', 'RR', 'RV']

#all_filter_fields = ['PASS', 'LowQual']

#for record2 in bcf_in.fetch():
#    r_filter = record2.filter
#    filter_names = ",".join(r_filter)
#    print(filter_names)
#    break

#bcf_list = list(bcf_in.fetch())
#for i in range(len(bcf_list)):
#    rec = bcf_list[i]
#    data = {}
#    for field in all_info_fields:
#        if field in rec.info.keys():
#            data[field] = rec.info[field]
#        else:
#            data[field] = None
#
#    samples = rec.samples
#    for j in range(len(samples)):
#        sample = samples[j]
#        for field2 in all_format_fields:
#            data[field2] = sample[field2]

#    r_filter = rec.filter
#    filter_names = ",".join(r_filter)
#    data["FILTER"] = filter_names
#
#    data["PHENOTYPE"] = None
#    for key, value in data.items():
#        if value is None:
#            data[key] = np.nan
#        if isinstance(value, list) or isinstance(value, tuple):
#            data[key] = str(value)
# 
#    df.loc[len(df)] = data


#print(df.head())
#df.to_excel('bcf_data.xlsx', index=False)

