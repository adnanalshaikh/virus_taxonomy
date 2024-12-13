from Bio import Entrez
from Bio import SeqIO
import pandas as pd
import numpy as np
import os

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VirusDataLoader:
    def __init__ (self, email = "", seq_type = 'refseq'):
        
        """
        Initialize VirusDataLoader with Entrez email and sequence type.

        Args:
            email (str): Email address for NCBI API usage.
            seq_type (str): Sequence type, either 'refseq' or 'gb'.
        """
        
        Entrez.email = email
        self.seq_type = seq_type
        self.db = 'nuccore' # NCBI database: 'nuccore' for RefSeq or 'nucleotide' for all genomes
        self.count = 1
        
        # Load taxonomy metadata
        data_file_path = os.path.join(HOME_DIR, 'data', 'VMR_MSL39_v1.xlsx')
        self.taxo_meta_df  = pd.read_excel(data_file_path, sheet_name='VMR MSL39 v1')
        
        # drop not used columns
        self.taxo_meta_df.drop(['Isolate Sort', 'Exemplar or additional isolate', 'Virus isolate designation'], 
                axis=1, inplace=True)
        
        # Initialize DataFrame for storing results
        self.data_frame = self.create_data_frame_header()
        self.data_frame.attrs['seq_type'] = 'seq_type'
  
    def create_data_frame_header(self):
        """Define and initialize the structure of the output DataFrame."""
        header = [
            'sort', 'gb_accession', 'refseq_accession', 'accession', 'sequence_version',
            'date', 'molecule_type', 'topology', 'realm', 'subrealm', 'kingdom', 'subkingdom',
            'phylum', 'subphylum', 'class', 'subclass', 'order', 'suborder', 'family',
            'subfamily', 'genus', 'subgenus', 'species', 'virus_name_abbreviation',
            'virus_name', 'genome_coverage', 'genome_composition', 'host_source',
            'organism', 'host', 'seq_length', 'seq'
        ]
        return  pd.DataFrame(columns=header)
   
    def fetch_genome(self, accession):
        """
        Fetch genome record from NCBI using accession number.

        Args:
            accession (str): Accession number of the genome.

        Returns:
            record: Genome record as a Biopython SeqRecord object.
        """
        
        print ("fetching accession # ", accession)
        
        try:
            handle = Entrez.efetch(db=self.db , id=accession, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "gb")
            handle.close()
            return record
        
        except Exception as e:
            print(f"Error fetching genome for accession {accession}: {e}")
            return None
    
    @staticmethod
    def parse_accession(accession):
        
        """
        Extract the first valid accession from a semicolon-separated list.

        Args:
            accession (str): Accession string.

        Returns:
            str: The first valid accession.
        """

        if not isinstance(accession, str): 
            return None
        
        n_split = accession.split(';')
        if  len(n_split) == 1 : return n_split[0].strip()
        else : 
            n_split = n_split[0].split(':')
            if len(n_split) == 1 : return n_split[0].strip()
            else: return n_split[1].strip()
    
            
    def extract_record_data(self, row, genome_record, gb_accession, refseq_accession):
        
        """
        Extract relevant data from a genome record and taxonomy row.

        Args:
            row (Series): Taxonomy metadata row.
            genome_record (SeqRecord): Genome record.

        Returns:
            dict: A dictionary with extracted data.
        """
        
       
        record = {
            'sort': row['Sort'],
            'gb_accession': gb_accession,
            'refseq_accession': refseq_accession,
            'accession': genome_record.annotations.get('accessions', [None])[0],
            'sequence_version': genome_record.annotations.get('sequence_version', None),
            'date': genome_record.annotations.get('date', None),
            'molecule_type': genome_record.annotations.get('molecule_type', None),
            'topology': genome_record.annotations.get('topology', None),
            'organism': genome_record.name,
            'host': self.extract_host_from_record(genome_record),
            'seq_length': len(genome_record.seq),
            'seq': str(genome_record.seq)
        }

        # Add taxonomy data
        taxonomy_keys = [
            'Realm', 'Subrealm', 'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum',
            'Class', 'Subclass', 'Order', 'Suborder', 'Family', 'Subfamily', 
            'Genus', 'Subgenus', 'Species'
        ]
        
        taxonomy_values = row[taxonomy_keys].fillna('').tolist()        
        taxonomy_dict = dict(zip([key.lower() for key in taxonomy_keys], taxonomy_values))
        record.update(taxonomy_dict)

        # Add additional metadata
        record.update({
            'virus_name_abbreviation': row['Virus name abbreviation(s)'],
            'virus_name': row['Virus name(s)'],
            'genome_coverage': row['Genome coverage'],
            'genome_composition': row['Genome composition'],
            'host_source': row['Host source']
        })
        
        return record
    
    @staticmethod
    def extract_host_from_record(genome_record):
        """
        Extract host information from a genome record.
    
        Args:
            genome_record (SeqRecord): Genome record.
    
        Returns:
            str: Host information or None.
        """
        if 'host' in genome_record.annotations:
            return genome_record.annotations['host']
        for feature in genome_record.features:
            if feature.type == "source" and 'host' in feature.qualifiers:
                return feature.qualifiers['host'][0]
        return None
    
    def append_records(self, taxa_name = 'all'):
        
        """
        Append genome records for a specific taxa or all taxa to the DataFrame.

        Args:
            taxa_name (str or list): Specific taxa name(s) to fetch records for.

        Returns:
            DataFrame: Updated DataFrame with genome records.
        """
        
        df = self.taxo_meta_df
        
        # Filter taxa
        if isinstance(taxa_name, list):
            query = df[df.isin(taxa_name).any(axis=1)]  
        else:
            query = df if taxa_name == 'all' else df[(df == taxa_name).any(axis=1)]
        
        # Process each genome record
        for index, row in query.iterrows():
            refseq_accession = self.parse_accession(row['Virus REFSEQ accession'])
            gb_accession = self.parse_accession(row['Virus GENBANK accession'])
                
            if self.seq_type == 'refseq':
                accession = refseq_accession
            elif self.seq_type == 'gb':
                accession = gb_accession
            else:
                print ("Unknown sequence number ")
                exit()
                
            if accession is None: continue
            if isinstance(accession, float) and np.isnan(accession): continue
                
            genome_record = self.fetch_genome(accession)
            if not genome_record:
                continue
            
            # Create a record dictionary           
            record = self.extract_record_data(row, genome_record, gb_accession, refseq_accession)
            self.data_frame.loc[len(self.data_frame.index)] = record    
            self.count +=  1
        
        return self.data_frame
        
    def save (self, file_name):
        """
        Save the DataFrame to a CSV file.

        Args:
            file_name (str): Name of the output file.
        """
        
        output_path = os.path.join(HOME_DIR, 'data', file_name)
        self.data_frame.to_csv(output_path, index = False)
        print(f"Data saved to {output_path}")
        
if __name__ == "__main__":
    
   #taxa_lists = ['Alsuviricetes']
   taxa_list = ['Tymovirales', 'Martellivirales', 'Alsuviricetes']
   seq_type='refseq'
   
   for taxa in taxa_list:
       loader = VirusDataLoader(email='aalshaikh@najah.edu', seq_type=seq_type)
       df = loader.append_records(taxa)
       loader.save(f"{taxa}_{seq_type}_org.csv")   
    
