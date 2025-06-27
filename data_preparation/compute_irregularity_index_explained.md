# Step-by-Step Explanation of the `compute_irregularity_index` Function

This function calculates an irregularity index for verbs based on their principal parts. Here's how it works:

## 1. Setup and Initialization
- **Inputs**: A DataFrame of linguistic data, lists of lemmas and time points, and a weighting parameter `alpha`
- **Output matrices**: Creates two empty matrices:
  - `irregularity`: Stores irregularity scores for each lemma at each time point
  - `m_values`: Stores the number of principal parts observed for each calculation

## 2. Data Processing Loop
For each time point and lemma combination:

- **Filter data**: Only consider observations up to the current time point
- **Calculate observed parts**: Count how many different principal parts exist for this lemma
- **Record m-value**: Store this count in the m-values matrix
- **Check sufficiency**: If fewer than 2 principal parts exist, set irregularity to 0 (cannot calculate)

## 3. Irregularity Calculation
When enough principal parts exist:

- **Collect observations**: For each principal part, take the most recent observation and add its vowel and consonant forms to sets
- **Calculate components**:
  ```
  V_irreg = (|distinct vowels| - 1) / (m - 1)
  C_irreg = (|distinct consonants| - 1) / (m - 1)
  ```
- **Combine components**: `IrregIndex = alpha * V_irreg + (1 - alpha) * C_irreg`

## 4. Data Imputation
After processing all time-lemma combinations:

- **Forward fill**: For each lemma, if a time point has no data (both irregularity and m-values are 0), copy the values from the most recent time point with data

## 5. Return Results
- Convert both matrices to Python lists
- Return them as a tuple: `(irregularity_list, m_values_list)`

## Key Insight

The irregularity score measures how much variation exists across a verb's principal parts. A regular verb should have consistent vowel and consonant patterns, resulting in a low score. An irregular verb will show more variation, resulting in a higher score.