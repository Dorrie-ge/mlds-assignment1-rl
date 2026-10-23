"""Check motion sign, packed trajectory storage and rollout input consistency."""
import unittest
import numpy as np
import torch
from pong_batch import collect, encode_state, unpack


class RecordingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        return torch.zeros((len(x), 2))


class TwoStepEnvironment:
    def image(self, column):
        frame = np.full((210, 160, 3), 144, dtype=np.uint8)
        frame[35, 2 * column, 0] = 255
        return frame

    def reset(self, seed):
        self.step_number = 0
        return self.image(0), {}

    def step(self, action):
        assert action in (2, 3)
        self.step_number += 1
        return self.image(self.step_number), float(self.step_number == 2), self.step_number == 2, False, {}


class RepresentationTests(unittest.TestCase):
    def test_default_preserves_original_stack(self):
        previous = np.zeros((80, 80), dtype=np.float32)
        current = previous.copy()
        current[3, 5] = 1
        np.testing.assert_array_equal(encode_state(previous, current), np.stack((previous, current)))
        self.assertEqual(previous.sum(), 0)

    def test_motion_sign_and_static_pixels(self):
        previous = np.zeros((80, 80), dtype=np.float32)
        current = previous.copy()
        previous[4, 5] = 1
        current[4, 6] = 1
        previous[20, 30] = current[20, 30] = 1
        state = encode_state(previous, current, 'split_frame_difference')
        self.assertEqual(state[0, 4, 6], 1)
        self.assertEqual(state[1, 4, 5], 1)
        self.assertEqual(state[:, 20, 30].sum(), 0)
        self.assertEqual(state.sum(), 2)
        np.testing.assert_array_equal(state[0] - state[1], current - previous)
        self.assertEqual(encode_state(current, current, 'split_frame_difference').sum(), 0)

    def test_packing_preserves_both_motion_channels(self):
        rng = np.random.default_rng(73)
        frames = rng.integers(2, size=(4, 80, 80)).astype(np.float32)
        states = np.stack([encode_state(a, b, 'split_frame_difference')
                           for a, b in zip(frames[:-1], frames[1:])])
        packed = np.packbits(states.astype(np.uint8).reshape(len(states), -1), axis=1)
        np.testing.assert_array_equal(unpack(packed), states)

    def test_rollout_and_training_storage_use_same_encoding(self):
        policy = RecordingPolicy()
        records = collect([TwoStepEnvironment()], policy, torch.device('cpu'), [8],
                          representation='split_frame_difference')
        actual = unpack(records[0]['packed'])
        observed = torch.cat(policy.inputs).numpy()
        np.testing.assert_array_equal(actual, observed)
        self.assertEqual(actual[0].sum(), 0)
        self.assertEqual(actual[1, 0, 0, 1], 1)
        self.assertEqual(actual[1, 1, 0, 0], 1)
        self.assertEqual(records[0]['rewards'], [0., 1.])


if __name__ == '__main__':
    unittest.main()
