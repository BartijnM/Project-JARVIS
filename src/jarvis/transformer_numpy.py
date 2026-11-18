import numpy as np

def token_embeddings(
        embedding_table: np.ndarray, input_ids: np.ndarray
) -> np.ndarray:
    return embedding_table[input_ids]

def add_position_embeddings(
    embeddings: np.ndarray, position_table: np.ndarray
) -> np.ndarray:
    sequence_length = embeddings.shape[1]
    positions = position_table[:sequence_length]
    return embeddings + positions

def attention_projections(
    inputs: np.ndarray,
    query_weights: np.ndarray,
    key_weights: np.ndarray,
    value_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = inputs @ query_weights
    keys = inputs @ key_weights
    values = inputs @ value_weights
    return queries, keys, values

def attention_scores(
    queries: np.ndarray, keys: np.ndarray
) -> np.ndarray:
    head_size = queries.shape[-1]
    transposed_keys = keys.swapaxes(-1, -2)
    return (queries @ transposed_keys) / np.sqrt(head_size)