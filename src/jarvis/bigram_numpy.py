import numpy as np

def bigram_logits(weights: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
    return weights[input_ids]

def cross_entropy(logits: np.ndarray, target_ids: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizers = np.log(np.exp(shifted).sum(axis=1))
    target_scores = shifted[np.arange(logits.shape[0]), target_ids]
    return float((log_normalizers - target_scores).mean())

def bigram_gradients(
        weights: np.ndarray, input_ids: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    logits = bigram_logits(weights, input_ids)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    logit_gradients = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    batch_size = input_ids.size
    logit_gradients[np.arange(batch_size), target_ids] -= 1
    logit_gradients /= batch_size

    weight_gradients = np.zeros_like(weights)
    np.add.at(weight_gradients, input_ids, logit_gradients)
    return weight_gradients

def train_bigram(
    weights: np.ndarray,
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    learning_rate: float,
    steps: int,
) -> np.ndarray:
    trained_weights = weights.copy()
    for _ in range(steps):
        gradients = bigram_gradients(trained_weights, input_ids, target_ids)
        trained_weights -= learning_rate * gradients
    return trained_weights

def generate_bigram(
    weights: np.ndarray, start_id: int, steps: int, seed: int
) -> list[int]:
    rng = np.random.default_rng(seed)
    token_ids = [start_id]

    for _ in range(steps):
        logits = weights[token_ids[-1]]
        exp_scores = np.exp(logits - logits.max())
        probabilities = exp_scores / exp_scores.sum()
        next_id = rng.choice(weights.shape[1], p=probabilities)
        token_ids.append(int(next_id))

    return token_ids