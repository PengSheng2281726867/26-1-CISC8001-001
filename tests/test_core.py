"""Scientific integration gates using real Push-T frames and verified ImageNet weights."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from alt.encoder import FusionEncoder, make_augmentation
from alt.loss import nt_xent
from alt.memory import MemoryBank, build_bank
from pusht.data import PushTData, ObservationFrames
from pusht.repro import load_config, load_split

PROJECT = Path(__file__).resolve().parents[1]
CFG = load_config(os.environ.get('ALT_CONFIG', PROJECT/'configs/alt_pusht_base.yaml'), PROJECT)


class CoreGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.split = load_split(CFG['split']['manifest'], CFG['paths']['dataset'], 30)
        cls.data = PushTData(CFG['paths']['dataset'])
        cls.device = torch.device(CFG['training']['device'])

    def test_split_integrity_and_leakage(self):
        s = self.split
        self.assertFalse(set(s['heldout_validation_episode_ids']) & set(s['ordered_train_pool_episode_ids']))
        self.assertEqual(s['train_subsets']['30'], s['train_subsets']['100'][:30])
        self.assertIsNone(s['train_subsets']['300'])
        self.assertIsNone(s['train_subsets']['500'])
        self.assertEqual(s['test_env_seeds'], list(range(100000, 100050)))
        folder = PROJECT/'artifacts/tests'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path = Path(temp)/'split.json'
            path.write_bytes(Path(CFG['split']['manifest']).read_bytes())
            path.with_suffix('.sha256').write_bytes(Path(CFG['split']['manifest']).with_suffix('.sha256').read_bytes())
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_split(path)

    def test_actual_alignment_and_all_episode_boundaries(self):
        data = self.data
        for ep in range(len(data.ends)):
            for frame in [0, int(data.ends[ep]-data.starts[ep])-1]:
                chunk, indices = data.action_chunk(ep, frame)
                start = data.index(ep, frame)
                np.testing.assert_array_equal(chunk[0], data.root['data/action'][start])
                self.assertEqual(chunk.dtype, np.float32)
                self.assertEqual(chunk.shape, (8, 2))
                self.assertTrue(np.all(indices >= data.starts[ep]))
                self.assertTrue(np.all(indices < data.ends[ep]))
                if frame:
                    np.testing.assert_array_equal(chunk, np.repeat(data.actions[start][None], 8, axis=0))
        source = (Path(CFG['paths']['dp_source'])/'demo_pusht.py').read_text()
        self.assertLess(source.index('episode.append(data)'), source.index('env.step(act)'))
        # Distinct real consecutive targets make an accidental +1 offset observable.
        ep = self.split['train_subsets']['30'][0]
        i = int(data.starts[ep])
        while np.array_equal(data.actions[i], data.actions[i+1]):
            i += 1
        frame = i-int(data.starts[ep])
        chunk, _ = data.action_chunk(ep, frame)
        np.testing.assert_array_equal(chunk[0], data.actions[i])
        self.assertFalse(np.array_equal(chunk[0], data.actions[i+1]))

    def test_two_views_have_identical_current_position(self):
        ds = ObservationFrames(self.data, self.split['train_subsets']['30'][:1],
                               make_augmentation(CFG['augmentation']))
        a, b, pos = ds[20]
        self.assertEqual(a.shape, (3, 96, 96))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(0 <= a.min() <= a.max() <= 1)
        self.assertFalse(torch.equal(a, b))
        np.testing.assert_array_equal(pos.numpy(), self.data.positions[ds.indices[20]])
        self.assertEqual(len(ds[20]), 3)  # views + current position; no action labels

    def test_nt_xent_matches_hand_computation_and_backward(self):
        x = torch.eye(4, requires_grad=True)
        actual = nt_xent(x, x, 0.4)
        # each anchor: exp(1/tau) positive, six orthogonal negatives
        expected = np.log(np.exp(2.5)+6)-2.5
        self.assertAlmostEqual(actual.item(), expected, places=6)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(actual.item(), nt_xent(x, x.roll(1, 0), 0.4).item())

    def test_pretrained_finetuning_and_bank_roundtrip(self):
        torch.manual_seed(42)
        model = FusionEncoder.from_pretrained(CFG['paths']['pretrained']).to(self.device)
        original = torch.load(CFG['paths']['pretrained'], weights_only=True, map_location='cpu')
        torch.testing.assert_close(model.image_encoder.conv1.weight.cpu(), original['conv1.weight'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        ds = ObservationFrames(self.data, self.split['train_subsets']['30'][:1],
                               make_augmentation(CFG['augmentation']))
        items = [ds[i] for i in [0, 10, 20, 30]]
        a, b, p = [torch.stack([x[j] for x in items]).to(self.device) for j in range(3)]
        model.train()
        before = model.image_encoder.conv1.weight.detach().clone()
        z1, z2 = model(a, p), model(b, p)
        self.assertEqual(z1.shape, (4, 128))
        torch.testing.assert_close(z1.norm(dim=-1), torch.ones(4, device=self.device))
        loss = nt_xent(z1, z2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.image_encoder.conv1.weight.grad).all())
        self.assertGreater(model.image_encoder.conv1.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.position_encoder[0].weight.grad.abs().sum().item(), 0)
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
        self.assertFalse(torch.equal(before, model.image_encoder.conv1.weight))
        bank = build_bank(model, self.data, self.split['train_subsets']['30'][:1], {'test': True})
        idx, sim = bank.query(bank.embeddings)
        # Duplicate observations can tie; returned match must still be a cosine maximum.
        self.assertTrue(torch.allclose(sim, torch.ones_like(sim), atol=1e-5))
        for n in [0, 20, len(bank.embeddings)-1]:
            self.assertEqual(self.data.index(bank.episode_ids[n], bank.local_frames[n]), bank.source_indices[n])
        folder = PROJECT/'artifacts/tests'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path = Path(temp)/'bank.pt'
            bank.save(path)
            loaded = MemoryBank.load(path)
            torch.testing.assert_close(loaded.embeddings, bank.embeddings)
            np.testing.assert_array_equal(loaded.source_indices, bank.source_indices)
        frames = ObservationFrames(self.data, self.split['train_subsets']['30'][:1])
        image, pos = frames[20]
        model.eval()
        with torch.inference_mode():
            q = model(image[None].to(self.device), pos[None].to(self.device))
            match, score = bank.query(q)
        self.assertGreater(score.item(), 0.99999)
        self.assertTrue(np.allclose(bank.embeddings[match.item()], bank.embeddings[20], atol=1e-5))
