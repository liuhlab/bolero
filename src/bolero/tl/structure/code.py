import numpy as np
import pandas as pd


class AACodeConverter:
    """
    A class for converting between amino acid representations (full name,
    three-letter code, and single-letter code) as well as DNA and RNA bases.
    """

    def __init__(self):
        """Initialize the amino acid and nucleotide code table as a Pandas DataFrame."""
        data = [
            ("Alanine", "Ala", "A"),
            ("Arginine", "Arg", "R"),
            ("Asparagine", "Asn", "N"),
            ("Aspartic Acid", "Asp", "D"),
            ("Cysteine", "Cys", "C"),
            ("Glutamic Acid", "Glu", "E"),
            ("Glutamine", "Gln", "Q"),
            ("Glycine", "Gly", "G"),
            ("Histidine", "His", "H"),
            ("Isoleucine", "Ile", "I"),
            ("Leucine", "Leu", "L"),
            ("Lysine", "Lys", "K"),
            ("Methionine", "Met", "M"),
            ("Phenylalanine", "Phe", "F"),
            ("Proline", "Pro", "P"),
            ("Serine", "Ser", "S"),
            ("Threonine", "Thr", "T"),
            ("Tryptophan", "Trp", "W"),
            ("Tyrosine", "Tyr", "Y"),
            ("Valine", "Val", "V"),
            # DNA & RNA Bases
            ("Adenine", "A", "A"),
            ("Cytosine", "C", "C"),
            ("Guanine", "G", "G"),
            ("Thymine", "T", "T"),
            ("Uracil", "U", "U"),
        ]

        self.aa_df = pd.DataFrame(data, columns=["name", "triple", "single"])
        self.mapping = {
            "name_to_triple": self.aa_df.set_index("name")["triple"].to_dict(),
            "name_to_single": self.aa_df.set_index("name")["single"].to_dict(),
            "triple_to_name": self.aa_df.set_index("triple")["name"].to_dict(),
            "triple_to_single": self.aa_df.set_index("triple")["single"].to_dict(),
            "single_to_name": self.aa_df.set_index("single")["name"].to_dict(),
            "single_to_triple": self.aa_df.set_index("single")["triple"].to_dict(),
        }

    def _convert(self, query, mapping_dict):
        """
        Internal function to perform efficient vectorized mapping of queries.

        Args:
            query (str, list, np.array, pd.Index, pd.Series): The input query.
            mapping_dict (dict): The dictionary used for mapping values.

        Returns
        -------
            str, list, np.array, pd.Index, pd.Series: Converted values in the same type as input.
        """
        if isinstance(query, str):
            return mapping_dict.get(query.capitalize(), None)
        elif isinstance(query, (list, np.ndarray, pd.Index, pd.Series)):
            query_series = pd.Series(query).str.capitalize()
            mapped_values = query_series.map(mapping_dict)
            return (
                mapped_values
                if isinstance(query, pd.Series)
                else (
                    mapped_values.to_numpy()
                    if isinstance(query, np.ndarray)
                    else mapped_values.tolist()
                )
            )
        else:
            raise TypeError(
                "Unsupported input type. Must be str, list, np.ndarray, pd.Index, or pd.Series."
            )

    def name_to_triple(self, query):
        """Convert amino acid full name to three-letter code."""
        return self._convert(query, self.mapping["name_to_triple"])

    def name_to_single(self, query):
        """Convert amino acid full name to single-letter code."""
        return self._convert(query, self.mapping["name_to_single"])

    def triple_to_name(self, query):
        """Convert three-letter amino acid code to full name."""
        return self._convert(query, self.mapping["triple_to_name"])

    def triple_to_single(self, query):
        """Convert three-letter amino acid code to single-letter code."""
        return self._convert(query, self.mapping["triple_to_single"])

    def single_to_name(self, query):
        """Convert single-letter amino acid code to full name."""
        return self._convert(query, self.mapping["single_to_name"])

    def single_to_triple(self, query):
        """Convert single-letter amino acid code to three-letter code."""
        return self._convert(query, self.mapping["single_to_triple"])


converter = AACodeConverter()
