"""Check both time-weighting gradients against an independent full-batch loss."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from pong_batch import update
from policy_gradient import returns_to_go


class PixelPolicy(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.head = nn.Linear(2, outputs)

    def forward(self, x):
        return self.head(x[:, :, 0, 0])


class TimeWeightingTests(unittest.TestCase):
    def test_chunked_actor_and_critic_gradients(self):
        torch.set_num_threads(1)
        for mode in ('episode_discount', 'uniform'):
            for baseline in (False, True):
                with self.subTest(mode=mode, baseline=baseline):
                    torch.manual_seed(17)
                    rng = np.random.default_rng(17)
                    policy = PixelPolicy(2)
                    value = PixelPolicy(1) if baseline else None
                    reference = copy.deepcopy(policy)
                    critic_reference = copy.deepcopy(value)
                    episodes, states = [], []
                    for length in (301, 37):  # Crosses the 256-step update chunk.
                        x = np.zeros((length, 2, 80, 80), dtype=np.float32)
                        x[:, :, 0, 0] = rng.integers(2, size=(length, 2))
                        rewards = rng.choice([-1., 0., 1.], size=length)
                        episodes.append(dict(
                            packed=np.packbits(x.astype(np.uint8).reshape(length, -1), axis=1),
                            actions=rng.integers(2, size=length).tolist(),
                            rewards=rewards.tolist(), returns=returns_to_go(rewards, .99)))
                        states.append(torch.tensor(x))
                    actor_loss = torch.tensor(0.)
                    critic_loss = torch.tensor(0.)
                    for ep, x in zip(episodes, states):
                        target = torch.tensor(ep['returns'])
                        v = critic_reference(x).squeeze(-1) if baseline else torch.zeros_like(target)
                        weights = .99 ** torch.arange(len(target)) if mode == 'episode_discount' else torch.ones_like(target)
                        actor_loss -= (weights * Categorical(logits=reference(x)).log_prob(
                            torch.tensor(ep['actions'])) * (target - v.detach())).sum() / len(episodes)
                        if baseline:
                            critic_loss += ((v - target) ** 2).mean() / len(episodes)
                    actor_loss.backward()
                    if baseline:
                        self.assertTrue(all(p.grad is None for p in critic_reference.parameters()))
                        critic_loss.backward()
                    opt = torch.optim.SGD(policy.parameters(), lr=0.)
                    vopt = torch.optim.SGD(value.parameters(), lr=0.) if baseline else None
                    update(policy, value, opt, vopt, episodes, torch.device('cpu'), 1e9, mode)
                    for actual, expected in zip(policy.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-4, rtol=1e-5)
                    if baseline:
                        for actual, expected in zip(value.parameters(), critic_reference.parameters()):
                            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
