import unittest

import numpy as np

from jarvis.transformer_numpy import attention_projections
from jarvis.transformer_numpy import attention_scores


class AttentionProjectionsTest(unittest.TestCase):
    def test_query_key_value_projections(self):
        inputs = np.array(
            [[[1.0, 2.0],
              [3.0, 4.0]]],
            dtype=np.float64,
        )
        query_weights = np.eye(2, dtype=np.float64)
        key_weights = np.array(
            [[0.0, 1.0], [1.0, 0.0]]
        )
        value_weights = np.array(
            [[2.0, 0.0], [0.0, 3.0]]
        )

        queries, keys, values = attention_projections(
            inputs, query_weights, key_weights, value_weights
        )

        for result in (queries, keys, values):
            self.assertEqual(result.shape, (1, 2, 2))

        np.testing.assert_array_equal(
            queries, [[[1.0, 2.0], [3.0, 4.0]]]
        )
        np.testing.assert_array_equal(
            keys, [[[2.0, 1.0], [4.0, 3.0]]]
        )
        np.testing.assert_array_equal(
            values, [[[2.0, 6.0], [6.0, 12.0]]]
        )
        
    def test_scaled_query_key_scores(self):
        queries = np.array(
            [[[1.0, 2.0],
              [3.0, 4.0]]],
            dtype=np.float64,
        )
        keys = np.array(
            [[[2.0, 1.0],
              [4.0, 3.0]]],
            dtype=np.float64,
        )

        actual = attention_scores(queries, keys)

        expected = np.array(
            [[[4.0, 10.0],
              [10.0, 24.0]]],
            dtype=np.float64,
        ) / np.sqrt(2.0)

        self.assertEqual(actual.shape, (1, 2, 2))
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-12
        )