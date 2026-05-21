import requests
import pandas as pd
from typing import List, Dict
import time
from tqdm import tqdm


def get_lemma_frequencies(
    lemmas: List[str],
    api_url: str = "https://korap.ids-mannheim.de/api/v1.0",
    query_language: str = "poliqarp",
    virtual_corpus: str = "",
    delay: float = 0.1,
    print_results: bool = False,
) -> Dict[str, int]:
    """
    Extract lemma frequencies from the KorAP API.

    Parameters:
    -----------
    lemmas : List[str]
        List of lemmas to query (e.g., ["laufen", "gehen", "kommen"])
    api_url : str
        Base URL of the KorAP API (default: DeReKo instance)
    query_language : str
        Query language to use (default: "poliqarp")
    virtual_corpus : str
        Virtual corpus definition to restrict the search (default: "" for whole corpus)
    delay : float
        Delay in seconds between requests to avoid rate limiting (default: 0.25)
    print_results: bool
        Whether to print the successfully retrieved results (default: False)

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping lemmas to their frequencies

    Example:
    --------
    >>> lemmas = ["Haus", "Baum", "Auto"]
    >>> frequencies = get_lemma_frequencies(lemmas)
    >>> print(frequencies)
    {'Haus': 1234567, 'Baum': 234567, 'Auto': 456789}
    """

    frequencies = {}

    for lemma in tqdm(lemmas, desc="Querying lemma frequencies"):
        if "|" in lemma:  # special case: compound with separator
            # call recursively for both parts and average
            mapping_parts = {
                "aus|eintragen": ["austragen", "eintragen"],
                "ein|ausgehen": ["eingehen", "ausgehen"],
                "auf|abgehen": ["aufgehen", "abgehen"],
                "aus|eingehen": ["ausgehen", "eingehen"],
            }
            parts = mapping_parts[lemma]
            freq_sum = 0
            for part in parts:
                part_freqs = get_lemma_frequencies(
                    [part],
                    api_url=api_url,
                    query_language=query_language,
                    virtual_corpus=virtual_corpus,
                    delay=delay,
                )
                freq_sum += part_freqs.get(part, 0)
            frequencies[lemma] = freq_sum / len(parts)
            print(
                f"✓ {lemma}: {frequencies[lemma]:,} occurrences (averaged from parts)"
            )
            continue
        # Construct the lemma query using KoralQuery/Poliqarp syntax
        # [tt/l=LEMMA] queries for the lemma in TreeTagger annotations
        query = f"[tt/l={lemma}]"

        # Prepare the API request parameters
        params = {
            "q": query,
            "ql": query_language,
            "count": 0,  # We only need totalResults, not actual matches
        }

        # Add virtual corpus if specified
        if virtual_corpus:
            params["cq"] = virtual_corpus

        try:
            # Make the API request
            response = requests.get(f"{api_url}/search", params=params, timeout=30)

            # Check if request was successful
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            # Extract the total number of results (frequency)
            total_results = data.get("meta", {}).get("totalResults", 0)

            # Store the frequency
            frequencies[lemma] = total_results
            if print_results == True:
                print(f"✓ {lemma}: {total_results:,} occurrences")

        except requests.exceptions.RequestException as e:
            print(f"✗ Error querying lemma '{lemma}': {e}")
            frequencies[lemma] = None

        except KeyError as e:
            print(f"✗ Error parsing response for lemma '{lemma}': {e}")
            frequencies[lemma] = None

        # Add delay to avoid overwhelming the API
        if delay > 0 and lemma != lemmas[-1]:
            time.sleep(delay)

    return frequencies


def main(data_file: str):
    """
    Main function to process lemma frequencies and assign averaged frequencies to lemma IDs.

    Parameters:
    -----------
    data_file : str
        Path to the input CSV file containing the data.
    use_geometric_mean : bool
        Whether to use the geometric mean for averaging frequencies (default: False).
    """
    # Load the data
    data = pd.read_csv(data_file)

    # Filter for ENHG corpus
    enhg = data[data["corpus"] == "ENHG"].copy()

    # Get unique lemmas
    lemmas_lst = enhg["lemma"].unique().tolist()

    # Fetch lemma frequencies
    lemma_freqs = get_lemma_frequencies(lemmas_lst)

    # Map frequencies back to the DataFrame
    enhg["modern_lemma_count"] = enhg["lemma"].map(lemma_freqs)

    # Group by lemma_id and calculate the average frequency
    mean_freqs = enhg.groupby("lemma_id")["modern_lemma_count"].mean()

    # Assign averaged frequencies back to the original data
    data["modern_lemma_count"] = data["lemma_id"].map(mean_freqs)
    data = data[(data["modern_lemma_count"].notna()) & (data["modern_lemma_count"] > 0)]
    data["modern_lemma_count"] = data["modern_lemma_count"].astype(int)

    # Save the updated data to a new file
    output_file = data_file.replace(".csv", "_freqs.csv")
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    # return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and assign modern lemma frequencies to lemma IDs."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the data.",
    )
    args = parser.parse_args()

    main(args.input)
