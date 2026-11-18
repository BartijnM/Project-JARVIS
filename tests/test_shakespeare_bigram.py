"""J02.5: vergelijk NumPy en PyTorch op hetzelfde Shakespeare-proefdeel."""

import math
from pathlib import Path
import unittest

import numpy as np
import torch

from jarvis import bigram_numpy as bn
from jarvis import bigram_torch as bt
from jarvis.dataset import load_text, make_next_token_pairs, split_text
from jarvis.tokenizer import CharTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SIZE = 10_000
SPLIT_INDEX = 9_000
LEARNING_RATE = 5.0
TRAINING_STEPS = 100
GENERATION_STEPS = 200
SEED = 42


class ShakespeareBigramTest(unittest.TestCase):
    def test_training_and_generation(self):
        # Beide varianten gebruiken dezelfde tekst, splits en tokenizer.
        text = load_text(str(PROJECT_ROOT / "data" / "shakespeare.txt"))
        sample = text[:SAMPLE_SIZE]
        self.assertEqual(len(sample), SAMPLE_SIZE)
        training_text, validation_text = split_text(sample, SPLIT_INDEX)
        tokenizer = CharTokenizer(training_text)
        vocab_size = len(tokenizer.char_to_id) + 1

        train_x, train_y = make_next_token_pairs(tokenizer.encode(training_text))
        val_x, val_y = make_next_token_pairs(tokenizer.encode(validation_text))
        self.assertEqual(len(train_x), 8_999)
        self.assertEqual(len(val_x), 999)

        train_x_np = np.array(train_x, dtype=np.int64)
        train_y_np = np.array(train_y, dtype=np.int64)
        val_x_np = np.array(val_x, dtype=np.int64)
        val_y_np = np.array(val_y, dtype=np.int64)

        train_x_t = torch.tensor(train_x, dtype=torch.long)
        train_y_t = torch.tensor(train_y, dtype=torch.long)
        val_x_t = torch.tensor(val_x, dtype=torch.long)
        val_y_t = torch.tensor(val_y, dtype=torch.long)

        # Nulgewichten geven ieder token dezelfde beginkans.
        weights_np = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        weights_t = torch.tensor(weights_np, dtype=torch.float64)
        initial_loss = bn.cross_entropy(
            bn.bigram_logits(weights_np, train_x_np), train_y_np
        )
        self.assertAlmostEqual(initial_loss, math.log(vocab_size), delta=1e-12)

        trained_np = bn.train_bigram(
            weights_np, train_x_np, train_y_np, LEARNING_RATE, TRAINING_STEPS
        )
        trained_t = bt.train_bigram(
            weights_t, train_x_t, train_y_t, LEARNING_RATE, TRAINING_STEPS
        )

        train_loss_np = bn.cross_entropy(
            bn.bigram_logits(trained_np, train_x_np), train_y_np
        )
        val_loss_np = bn.cross_entropy(
            bn.bigram_logits(trained_np, val_x_np), val_y_np
        )
        train_loss_t = bt.cross_entropy(
            bt.bigram_logits(trained_t, train_x_t), train_y_t
        ).item()
        val_loss_t = bt.cross_entropy(
            bt.bigram_logits(trained_t, val_x_t), val_y_t
        ).item()

        np.testing.assert_allclose(
            trained_t.numpy(), trained_np, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            [train_loss_t, val_loss_t], [train_loss_np, val_loss_np],
            rtol=0, atol=1e-12,
        )
        self.assertLess(train_loss_np, initial_loss)
        np.testing.assert_array_equal(weights_np, np.zeros_like(weights_np))
        self.assertTrue(torch.equal(weights_t, torch.zeros_like(weights_t)))

        print(f"\nBeginloss:         {initial_loss:.6f}")
        print(f"NumPy training:    {train_loss_np:.6f}")
        print(f"PyTorch training:  {train_loss_t:.6f}")
        print(f"NumPy validatie:   {val_loss_np:.6f}")
        print(f"PyTorch validatie: {val_loss_t:.6f}")

        # Dezelfde seed moet per variant dezelfde generatie opleveren.
        # NumPy en PyTorch hoeven onderling niet dezelfde tekst te trekken.
        start_id = tokenizer.encode("F")[0]
        self.assertNotEqual(start_id, 0)
        for name, model, weights in (
            ("NumPy", bn, trained_np),
            ("PyTorch", bt, trained_t),
        ):
            with self.subTest(variant=name):
                before = weights.copy() if name == "NumPy" else weights.clone()
                first = model.generate_bigram(weights, start_id, GENERATION_STEPS, SEED)
                second = model.generate_bigram(weights, start_id, GENERATION_STEPS, SEED)
                self.assertEqual(first, second)
                self.assertEqual(len(first), GENERATION_STEPS + 1)
                self.assertEqual(first[0], start_id)
                self.assertTrue(all(0 <= token_id < vocab_size for token_id in first))
                self.assertEqual(
                    model.generate_bigram(weights, start_id, 0, SEED), [start_id]
                )
                if name == "NumPy":
                    np.testing.assert_array_equal(weights, before)
                else:
                    self.assertTrue(torch.equal(weights, before))
                print(f"\n{name}-tekst:\n{tokenizer.decode(first)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
