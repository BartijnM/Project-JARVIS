def split_text(text:str, split_index: int) -> tuple[str, str]:
    return text[:split_index], text[split_index:]

def make_next_token_pairs(token_ids: list[int]) -> tuple[list[int], list[int]]:
    inputs = token_ids[:-1]
    targets = token_ids[1:]
    return inputs, targets

