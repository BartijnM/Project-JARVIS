import torch
import torch.nn.functional as F

def bigram_logits(
    weights: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    return weights[input_ids]

def cross_entropy(
        logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(logits, target_ids, reduction="mean")

def train_bigram(
    weights: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    learning_rate: float,
    steps: int,
) -> torch.Tensor:
    trained_weights = weights.detach().clone().requires_grad_(True)

    for _ in range(steps):
        trained_weights.grad = None

        logits = bigram_logits(trained_weights, input_ids)
        loss = cross_entropy(logits, target_ids)
        loss.backward()

        gradients = trained_weights.grad
        assert gradients is not None

        with torch.no_grad():
            trained_weights -= learning_rate * gradients

    return trained_weights.detach()

@torch.no_grad()
def generate_bigram(
    weights: torch.Tensor, start_id: int, steps: int, seed: int
) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_ids = [start_id]

    for _ in range(steps):
        logits = weights[token_ids[-1]]
        probabilities = torch.softmax(logits, dim=0)
        next_id = torch.multinomial(
            probabilities, num_samples=1, generator=generator
        )
        token_ids.append(int(next_id.item()))

    return token_ids
