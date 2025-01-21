from collections import Counter
import pandas as pd

def calculate_event_weights_from_df(df, column_name):
    """
    Calculate event combination counts and edge weights from a DataFrame column.

    Args:
        df (pd.DataFrame): A DataFrame containing the log sequence.
        column_name (str): The column name containing the event IDs.

    Returns:
        tuple: A dictionary of event counts and a dictionary of edge weights.
    """
    # Step 1: Extract the event sequence from the specified column
    event_list = df[column_name].astype(str).tolist()

    # Step 2: Extract all adjacent event pairs
    event_pairs = [event_list[i] + '@' + event_list[i+1] for i in range(len(event_list) - 1)]

    # Step 3: Count occurrences of each event pair
    event_counts = Counter(event_pairs)

    # Step 4: Calculate edge weights
    total_pairs = sum(event_counts.values())
    edge_weights = {pair: count / total_pairs for pair, count in event_counts.items()}

    return event_counts, edge_weights

def calculate_event_weights_from_df_with_interval(df, column_name, interval=1):
    # Step 1: Extract the event sequence from the specified column
    event_list = df[column_name].astype(str).tolist()

    # Step 2: Extract all adjacent event pairs
    event_pairs = []
    for x in range(interval):
        event_pairs.append([event_list[i] + '@' + event_list[i+x+1] for i in range(len(event_list) - interval)])

    # Step 3: Count occurrences of each event pair
    event_counts = []
    for x in range(interval):
        event_counts.append(Counter(event_pairs[x]))

    # Step 4: Calculate edge weights
    total_pairs = []
    for x in range(interval):
        total_pairs.append(sum(event_counts[x].values()))
    edge_weights = []
    for x in range(interval):
        edge_weight = {}
        for pair, count in event_counts[x].items():
            edge_weight[pair] = count / total_pairs[x]
        edge_weights.append(edge_weight)

    return event_counts, edge_weights

if __name__ == '__main__':
    # Example usage
    data = {
        'EventId': ['A21312', '31231B', 'A', 'B', 'B', 'B', 'C', 'A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    event_counts, edge_weights = calculate_event_weights_from_df_with_interval(df, 'EventId', interval=2)

    print("Event Counts:", event_counts)
    print("Edge Weights:", edge_weights)
