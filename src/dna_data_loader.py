from Bio import Entrez
from Bio import SeqIO
import pandas as pd
import numpy as np
import os

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class VirusData:
    def __init__ (self, email = "", seq_type = 'refseq'):
        Entrez.email = email
        self.seq_type = seq_type
        
        self.count = 1
        # chose the database : 
        # nuccore : for RefSeq which is smaller set with cleaner genome and include more annotation
        # nucleotide : whole genome 
        self.db = 'nuccore' #"nucleotide"
        data_file_path = os.path.join(HOME_DIR, 'data', 'VMR_MSL39_v1.xlsx')
        self.taxo_meta_df  = pd.read_excel(data_file_path, sheet_name='VMR MSL39 v1')
        
        # drop not used columns
        self.taxo_meta_df.drop(['Isolate Sort', 'Exemplar or additional isolate', 'Virus isolate designation'], 
                axis=1, inplace=True)
        
        self.df = None
        self.create_df_header()
        self.df.attrs['seq_type'] = 'seq_type'
  
    def fetch_genome(self, accession):
        handle = Entrez.efetch(db=self.db , id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "gb")
        handle.close()
        return record

    def create_df_header(self):
        header = ['sort', 'gb_accession', 'refseq_accession', 'accession', 'sequence_version', 
                  'date', 'molecule_type', 'topology', 
                  'realm', 'subrealm', 'kingdom', 'subkingdom', 
                  'phylum','subphylum', 'class', 'subclass', 'order', 'suborder', 
                  'family', 'subfamily', 'genus', 'subgenus', 'species', 
                  'virus_name_abbreviation', 'virus_name', 'genome_coverage', 
                  'genome_composition', 'host_source', 'organism', 'host', 'seq_length', 'seq']
        
        self.df = pd.DataFrame(columns = header)
    
    def get_accession(self, n):
        if not isinstance(n, str): return None 
        n_split = n.split(';')
        if  len(n_split) == 1 : return n_split[0].strip()
        else : 
            n_split = n_split[0].split(':')
            if len(n_split) == 1 : return n_split[0].strip()
            else: return n_split[1].strip()
    
            
    
    def append_records(self, taxa_name = 'all'):
        
        '''        
        query = df[
                    (df['Genome composition'] == 'dsDNA'') &
                    (df[''Host source''] == 'archaea') |
                    (df == {taxa_name}).any(axis=1)
                    ]
        ''' 
        seq_type = self.seq_type
        
        df = self.taxo_meta_df
        if isinstance(taxa_name, list):
            query = df[df.isin(taxa_name).any(axis=1)]  
        else:
            query = df if taxa_name == 'all' else df[(df == taxa_name).any(axis=1)]
        
        for index, row in query.iterrows():
            refseq_accession = row['Virus REFSEQ accession']
            refseq_accession = self.get_accession(refseq_accession)
            
            gb_accession = row['Virus GENBANK accession']
            gb_accession = self.get_accession(gb_accession)
                
            if seq_type == 'refseq':
                gr_accession = refseq_accession
            elif seq_type == 'gb':
                gr_accession = gb_accession
            else:
                print ("Unknown sequence number ")
                exit()
                
            if gr_accession is None: continue
            if isinstance(gr_accession, float) and np.isnan(gr_accession): continue
                
            #genome_record = self.fetch_genome(refseq_accession)
            genome_record = self.fetch_genome(gr_accession)
            
            record = {
                'sort' : row['Sort'],
                'gb_accession' : gb_accession,
                'refseq_accession' : refseq_accession,
                'accession': genome_record.annotations['accessions'][0],
                'sequence_version': genome_record.annotations.get('sequence_version', None),
                'date' : genome_record.annotations['date'],
                'molecule_type': genome_record.annotations['molecule_type'],
                'topology': genome_record.annotations['topology'] 
                }

        
            taxo_keys = ['Realm', 'Subrealm', 'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum', 
                 'Class', 'Subclass', 'Order', 'Suborder', 'Family', 'Subfamily', 'Genus', 
                 'Subgenus', 'Species']
        
            taxo_keys_lower = [v.lower() for v in taxo_keys]
            taxonomy_values = row[taxo_keys].tolist()
            
            taxonomy_dict = dict(zip(taxo_keys_lower, taxonomy_values))
            record.update(taxonomy_dict)
            
            record['virus_name_abbreviation'] = row['Virus name abbreviation(s)']
            record['virus_name']      =     row['Virus name(s)']
            record['genome_coverage'] = row['Genome coverage']
            record['genome_composition'] =  row['Genome composition']
            record['host_source'] = row ['Host source']
    
        
            record['organism'] = genome_record.name
            record['host'] = None
            if 'host' in genome_record.annotations:
                record['host'] = genome_record.annotations['host'] 
            else:
                for feature in genome_record.features:
                    if feature.type == "source" and 'host' in feature.qualifiers:
                        record['host'] = feature.qualifiers['host'][0]
                        break
                
            record['seq_length'] = len(genome_record.seq)
            record['seq'] = str(genome_record.seq)    
            
            print(self.count, len(genome_record.annotations['taxonomy']), genome_record.annotations['taxonomy'])
            self.df.loc[len(self.df.index)] = record    
            self.count = self.count + 1
        
        return self.df
    
    def save (self, file_name):
        output_path = os.path.join(HOME_DIR, 'data', file_name)
        self.df.to_csv(output_path, index = False)
        
if __name__ == "__main__":
    
    import os
    # chose which taxas to be downloaded 
    #viru_taxas = ['Alsuviricetes']
    viru_taxas = ['Martellivirales']
    data = VirusData(email = 'aalshaikh@najah.edu', seq_type = 'refseq')
    df = data.append_records(viru_taxas)
    #df = df[df['family'] != 'Duinviridae']
    #df = df[df['molecule_type'] == 'DNA']
    #data_path = os.path.join(HOME_DIR, "data", "Alsuviricetes_refseq.csv")
    data_path = os.path.join(HOME_DIR, "data", "Martellivirales_refseq_org.csv")
    
    data.save(data_path)

