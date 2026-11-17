class CharTokenizer:
    def __init__(self, training_text: str) -> None:
        self.char_to_id ={
            char: token_id
            for token_id, char in enumerate(sorted(set(training_text)), start=1)
        }
        
    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(char, 0) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {
            token_id: char for char, token_id in self.char_to_id.items()
        }
        return "".join(id_to_char.get(token_id, "\uFFFD") for token_id in token_ids)