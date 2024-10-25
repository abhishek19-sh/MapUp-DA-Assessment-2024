from typing import Dict, List
from typing import Dict, List
import pandas as pd
import itertools
import re
import math
import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code here
    result = []
    for i in range(0, len(lst), n):
        new_list = []
        # Collecting elements for the current group and reverse manually
        for j in range(min(n, len(lst) - i)):
            new_list.insert(0, lst[i + j])  # Insert elements at the beginning to reverse

        result.extend(new_list)  # Appending the reversed group to the final list

    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    len_dict = {}

    for string in lst:
        length = len(string)
        if length not in len_dict:
            len_dict[length] = []
        len_dict[length].append(string)

    len_dict = dict(sorted(len_dict.items()))
    return len_dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flat_dict = {}
    stack = [(nested_dict, '')]  # Initialize the stack with the original dictionary and an empty parent key

    # Process each item in the stack until it's empty
    while stack:
        current_dict, parent_key = stack.pop()  # Get the current dictionary and its parent key from the stack
        for key, value in current_dict.items():  # Iterate through all key-value pairs in the current dictionary
            # Create a new key by combining the parent key and the current key
            if parent_key:
                new_key = f"{parent_key}{sep}{key}"
            else:
                new_key = key

            # If the value is a dictionary, add it to the stack for further processing
            if isinstance(value, dict):
                stack.append((value, new_key))
            # If the value is a list, iterate through each item in the list
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    # If the item is a dictionary, add it to the stack with an updated key including the index
                    if isinstance(item, dict):
                        stack.append((item, f"{new_key}[{index}]") )
                    else:
                        # If the item is not a dictionary, add it to the flat dictionary with an indexed key
                        flat_dict[f"{new_key}[{index}]"] = item
            # If the value is neither a dictionary nor a list, add it to the flat dictionary
            else:
                flat_dict[new_key] = value

    return flat_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
   """

    unique_perms = set(itertools.permutations(nums))
    return [list(perm) for perm in unique_perms]




def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.

    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',   # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',   # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]

    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))

    return dates



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode polyline string into coordinates using the polyline module
    coords = polyline.decode(polyline_str)  # Decode the polyline into coordinates
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # Calculate distances between consecutive points using the Haversine formula
    distances = [0]  # First point has no previous point, so distance is 0
    for i in range(1, len(coords)):
        # Using Haversine formula to calculate the distance between two points on the Earth's surface
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        R = 6371000  # Radius of Earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distances.append(distance)
    
    # Add distance column to DataFrame
    df['distance'] = distances
    return df
    
    


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
    # Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Create the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix


def time_check(df) -> pd.Series:
    """
    Check whether each unique (`id`, `id_2`) pair covers a full 24-hour and 7-day period.

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: A boolean series with a multi-index (id, id_2) that indicates
                   if the timestamps are incorrect (True if incorrect, False otherwise).
    """
    # Initialize a dictionary to track completeness for each (id, id_2) pair
    completeness_dict = {}

    # Map day names to integers
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    # Group by 'id' and 'id_2'
    grouped = df.groupby(['id', 'id_2'])

    for (id_val, id_2_val), group in grouped:
        # Track days and hours covered
        days_covered = set()
        hours_covered = set()

        for _, row in group.iterrows():
            start_hour = int(row['startTime'].split(':')[0])
            end_hour = int(row['endTime'].split(':')[0])
            start_day = day_mapping[row['startDay']]
            end_day = day_mapping[row['endDay']]

            # Add days covered
            current_day = start_day
            while True:
                days_covered.add(current_day)
                if current_day == end_day:
                    break
                current_day = (current_day + 1) % 7  # Assume days are represented as 0-6 (Monday-Sunday)

            # Add hours covered
            for hour in range(start_hour, end_hour + 1):
                hours_covered.add(hour % 24)

        # Check if all days (0-6) and all hours (0-23) are covered
        is_complete = len(days_covered) == 7 and len(hours_covered) == 24
        completeness_dict[(id_val, id_2_val)] = is_complete if id_2_val != -1 else False

    # Convert the dictionary to a pandas Series with a MultiIndex
    completeness_series = pd.Series(completeness_dict)
    return completeness_series
