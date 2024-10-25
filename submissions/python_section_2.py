import pandas as pd





def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = df['id_start'].unique()
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids, data=float('inf'))
    
    # Set diagonal to 0
    for id_val in unique_ids:
        distance_matrix.at[id_val, id_val] = 0
    
    # Fill in the known distances
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']
    
    # cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix



def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # List to store the records
    records = []


    # Iterate through rows and columns of the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            # Avoid self-loops (i.e., rows where id_start == id_end)
            if id_start != id_end:
                distance = df.loc[id_start, id_end]

                # Appending the record only if the distance is finite and not missing
                if pd.notna(distance) and distance != float('inf'):
                    records.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance
                    })

    # Create a DataFrame from the list of records
    unrolled_df = pd.DataFrame(records)

    return unrolled_df




def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1

    # Finding IDs whose average distance is within 10%
    ids_within_threshold = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = ids_within_threshold[(ids_within_threshold >= lower_bound) & (ids_within_threshold <= upper_bound)].index.tolist()

    return sorted(ids_within_threshold)




def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Adding columns for time-based toll rates
    # Generate all possible combinations of days and times for a full week and a 24-hour period
    records = []
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_ranges = [
        ('00:00:00', '10:00:00', 0.8),
        ('10:00:00', '18:00:00', 1.2),
        ('18:00:00', '23:59:59', 0.8)
    ]
    weekend_discount = 0.7

    for _, row in df.iterrows():
        for day in days_of_week:
            for start_time, end_time, discount_factor in time_ranges:
                record = row.copy()
                record['start_day'] = day
                record['start_time'] = pd.to_datetime(start_time).time()
                record['end_day'] = day
                record['end_time'] = pd.to_datetime(end_time).time()

                if day in ['Saturday', 'Sunday']:
                    discount_factor = weekend_discount

                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    record[vehicle] *= discount_factor

                records.append(record)

    # Create a DataFrame from the list of records
    time_based_df = pd.DataFrame(records)

    return time_based_df


