from typing import Dict, List,Any
from datetime import datetime, timedelta
import pandas as pd
import polyline
import numpy as np
import re


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    :param lst: A list of integers
    :param n: Size of the group to reverse
    :return: List of integers with elements reversed in groups of n
    """
    #Iterating through the list and reversing chunks of size n.
    for i in range(0, len(lst), n):
        lst[i:i + n] = lst[i:i + n][::-1]  # Reverse the chunk directly within lst

    return lst




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    :param lst: A list of strings
    :return: A dictionary where keys are string lengths and values are lists of strings
    """
    length_dict = {}
    # Group strings by their lengths
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []  # Create a new list for this length if it doesn't exist
        length_dict[length].append(string)  # Append the string to the corresponding length

    # Sort the dictionary by key (length)
    sorted_dict = dict(sorted(length_dict.items()))

    return sorted_dict



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(current_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key))  # Recursively flatten the dictionary
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{i}]"))  # Flatten the dict inside the list
                    else:
                        items[f"{new_key}[{i}]"] = item  # Directly assign the item
            else:
                items[new_key] = v  # Directly assign the value
        return items
    return flatten(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        # If we've reached the end of the array, add a copy of the current permutation
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()  # To track which numbers we've already used at this position
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack

    result = []
    nums.sort()  # Sort to ensure duplicates are adjacent
    backtrack(0)
    return result
    #pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expressions for different date formats
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'   # yyyy.mm.dd
    ]
    
    # Find all dates using regex patterns
    found_dates = []
    for pattern in patterns:
        found_dates.extend(re.findall(pattern, text))
    
    return found_dates
#pass



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Check if the polyline string is valid and not empty
    if not polyline_str:
        raise ValueError("The polyline string is empty or invalid.")

    # Decode the polyline string
    try:
        coordinates = polyline.decode(polyline_str)
    except Exception as e:
        raise ValueError(f"Error decoding polyline: {e}")

    df=pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
     # Calculate distances between consecutive points
    distances = [0.0]  # First distance is 0
    for i in range(1, len(df)):
        lat1, lon1 = df.at[i-1, 'latitude'], df.at[i-1, 'longitude']
        lat2, lon2 = df.at[i, 'latitude'], df.at[i, 'longitude']

        # Calculate Euclidean distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        distance = np.sqrt(dlat**2 + dlon**2) * 111139  # Convert to meters
        distances.append(distance)
    
    # Assign the distances to the DataFrame
    df['distance'] = distances

    return df
    #return pd.Dataframe()

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
     """
    # Your code here 
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Create a new matrix for the transformed values
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            #Sum all elements in the same row and column, excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix
    #return []


 


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
     # Convert timestamp columns to datetime objects
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Determine the start and end of the 7-day period
    min_start = df.groupby(['id', 'id_2'])['start_datetime'].min()
    max_end = df.groupby(['id', 'id_2'])['end_datetime'].max()

    # Check if the timestamps cover a full 24-hour period (from 00:00:00 to 23:59:59)
    full_day_condition = (max_end - min_start).dt.days == 6  # 6 days difference means they span 7 days
    full_day_time_condition = (min_start.dt.time == pd.to_datetime("00:00:00").time()) & \
                              (max_end.dt.time == pd.to_datetime("23:59:59").time())
    
    # Combine the checks
    is_correct = full_day_condition & full_day_time_condition

    # Create a boolean series with multi-index (id, id_2)
    result = is_correct.groupby(['id', 'id_2']).all()

    return ~result  # Return True if incorrect

