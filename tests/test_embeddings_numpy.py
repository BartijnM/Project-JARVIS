import unittest

import numpy as np

from jarvis.transformer_numpy import token_embeddings
from jarvis.transformer_numpy import add_position_embeddings


class TokenEmbeddingsTest(unittest.TestCase):
    def test_lookup_preserves_batch_and_positions(self):
        table = np.array(
            [[0.0, 0.0],
             [1.0, 0.5],
             [0.2, 1.0]],
            dtype=np.float64,
        )
        input_ids = np.array(
            [[1, 2, 1],
             [2, 1, 0]],
            dtype=np.int64,
        )

        actual = token_embeddings(table, input_ids)

        expected = np.array(
            [[[1.0, 0.5], [0.2, 1.0], [1.0, 0.5]],
             [[0.2, 1.0], [1.0, 0.5], [0.0, 0.0]]],
            dtype=np.float64,
        )

        self.assertEqual(actual.shape, (2, 3, 2))
        np.testing.assert_array_equal(actual, expected)

    def test_positions_are_added_to_each_sequence(self):
        table = np.array(
            [[0.0, 0.0],
             [1.0, 0.5],
             [0.2, 1.0]],
            dtype=np.float64,
        )
        input_ids = np.array(
            [[1, 2, 1],
             [2, 1, 0]],
            dtype=np.int64,
        )
        position_table = np.array(
            [[0.0, 0.0],
             [0.1, 0.2],
             [0.2, 0.4],
             [9.0, 9.0]],
            dtype=np.float64,
        )
        embeddings = token_embeddings(table, input_ids)
        original = embeddings.copy()

        actual = add_position_embeddings(embeddings, position_table)

        expected = np.array(
            [[[1.0, 0.5], [0.3, 1.2], [1.2, 0.9]],
             [[0.2, 1.0], [1.1, 0.7], [0.2, 0.4]]],
            dtype=np.float64,
        )

        self.assertEqual(actual.shape, (2, 3, 2))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(embeddings, original)