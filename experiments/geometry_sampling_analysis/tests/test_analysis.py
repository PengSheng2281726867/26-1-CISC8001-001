import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import read_selected
from geometry import (batch_metrics,candidate_summary,distances,nearest,pools_and_time,
                      relation,resolve_calibration,resolve_profile,sample_structured,wrapped_angle)


class GeometryTests(unittest.TestCase):
    def test_absolute_angle_conversion_and_profile(self):
        settings = {'absolute_values':{
            'near':{'d_p':{'x':20},'d_T':{'x':20},'d_theta_degrees':{'x':5}},
            'far':{'d_p':{'y':80},'d_T':{'y':60},'d_theta_degrees':{'y':20}}}}
        cal = resolve_calibration(None,None,settings)
        t = resolve_profile(cal,{'near_level':'x','far_level':'y'})
        self.assertEqual(t['p_close'],20)
        self.assertAlmostEqual(t['theta_close'],np.deg2rad(5))
        self.assertAlmostEqual(t['theta_far'],np.deg2rad(20))

    def test_periodic_distance_and_components(self):
        state = np.array([[0,0,10,10,np.deg2rad(359)], [3,4,16,18,np.deg2rad(1)]])
        d = distances(state)
        self.assertAlmostEqual(float(d['d_p'][0,1]),5)
        self.assertAlmostEqual(float(d['d_T'][0,1]),10)
        self.assertAlmostEqual(float(d['d_theta'][0,1]),np.deg2rad(2),places=7)
        self.assertAlmostEqual(float(d['D'][0,1]),(5/512)**2+(10/512)**2+(2/180)**2,places=9)
        self.assertAlmostEqual(float(wrapped_angle(0,2*np.pi)),0)

    def test_candidate_pools_exact_boundaries(self):
        ep,local = np.array([0,0,0,0,1]),np.array([0,1,8,9,0])
        pools,time = pools_and_time(ep,local)
        self.assertFalse(pools['all'][0,0])
        self.assertTrue(pools['all'][0,1])
        self.assertFalse(pools['exclude_local8'][0,2])
        self.assertTrue(pools['exclude_local8'][0,3])
        self.assertTrue(pools['cross'][0,4])
        self.assertFalse(pools['cross'][0,3])
        self.assertEqual(sum(int(m.sum()) for m in time.values()),20)

    def test_hard_a_and_hard_b_or_rule(self):
        state = np.array([[0,0,0,0,0], [100,0,1,0,.01], [1,0,100,0,0],
                          [1,0,1,0,2], [1,0,1,0,.01]])
        d = distances(state)
        threshold = {'p_close':5,'T_close':5,'theta_close':.1,'p_far':80,'T_far':80,'theta_far':1}
        self.assertEqual(np.flatnonzero(relation(d,'hard_a',threshold)[0]).tolist(),[1])
        self.assertEqual(np.flatnonzero(relation(d,'hard_b',threshold)[0]).tolist(),[2,3])
        self.assertEqual(np.flatnonzero(relation(d,'similar',threshold)[0]).tolist(),[4])

    def test_pair_counts_and_failed_anchors(self):
        mask = np.array([[0,1,1,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]],dtype=bool)
        result = candidate_summary(mask,np.array([0,0,1,1]),[0,1])
        self.assertEqual(result['ordered_pairs'],4)
        self.assertEqual(result['unordered_pairs'],2)
        self.assertEqual(result['anchors_without_candidate'],1)
        self.assertEqual(result['episode_endpoint_counts'],[3,1])

    def test_nearest_excludes_self_and_stable_ties(self):
        d = {'D':np.zeros((4,4),dtype=np.float32)}
        allowed = ~np.eye(4,dtype=bool)
        result = nearest(d,allowed,2)
        np.testing.assert_array_equal(result[0],[1,2])
        np.testing.assert_array_equal(result[1],[0,2])
        with self.assertRaises(ValueError):
            nearest(d,allowed,4)

    def test_restricted_state_slices_only(self):
        class Restricted:
            def __init__(self):
                self.calls = []
            def __getitem__(self,index):
                self.calls.append((index[0].start,index[0].stop))
                return np.zeros((index[0].stop-index[0].start,5))
        array = Restricted()
        state,ep,local,source,reads = read_selected(array,np.array([3,7,9]),[2,0])
        self.assertEqual(array.calls,[(7,9),(0,3)])
        self.assertEqual(ep.tolist(),[2,2,0,0,0])
        self.assertEqual(source.tolist(),[7,8,0,1,2])

    def test_failed_slots_never_backfilled(self):
        n = 64
        empty = [np.array([],dtype=int) for _ in range(n)]
        candidates = {role:empty for role in ['similar','hard_a','hard_b']}
        batch = sample_structured(np.arange(8),[True]*4+[False]*4,candidates,n,
                                  np.random.default_rng(1),True)
        self.assertEqual(len(batch['indices']),16)
        self.assertEqual(len(batch['failures']),16)
        self.assertEqual(sum(f['role']=='hard_a' for f in batch['failures']),4)
        self.assertEqual(len(set(batch['indices'])),16)

    def test_uniqueness_exhaustion_is_reported_not_relaxed(self):
        n = 64
        single = [np.array([0]) for _ in range(n)]
        empty = [np.array([],dtype=int) for _ in range(n)]
        candidates = {'similar':single,'hard_a':empty,'hard_b':empty}
        anchors = np.arange(1,9)
        batch = sample_structured(anchors,[True]*4+[False]*4,candidates,n,np.random.default_rng(1),True)
        self.assertEqual(sum(f['reason']=='unique_constraint_exhausted' for f in batch['failures']),7)
        self.assertEqual(len(set(batch['indices'])),len(batch['indices']))
        repeated = sample_structured(anchors,[True]*4+[False]*4,candidates,n,np.random.default_rng(1),False)
        self.assertGreaterEqual(len(repeated['indices'])-len(set(repeated['indices'])),7)

    def test_complete_batch_roles_and_reproducibility(self):
        n = 64
        candidates = {role:[np.delete(np.arange(n),i) for i in range(n)]
                      for role in ['similar','hard_a','hard_b']}
        args = (np.arange(8),[True]*4+[False]*4,candidates,n)
        a = sample_structured(*args,np.random.default_rng(1),True)
        b = sample_structured(*args,np.random.default_rng(1),True)
        self.assertEqual(a,b)
        self.assertEqual(len(a['indices']),32)
        self.assertEqual(len(set(a['indices'])),32)
        self.assertEqual(a['roles'].count('hard_a'),4)
        self.assertEqual(a['roles'].count('hard_b'),4)
        masks = {role:~np.eye(n,dtype=bool) for role in candidates}
        result = batch_metrics(a,masks,np.arange(n)%4)
        self.assertTrue(result['complete'])
        self.assertEqual(result['similar_unique_pairs'],496)


if __name__ == '__main__':
    unittest.main()
