"""
Track D: Submodular Optimization
Implements:
1. Submodular Set Function (Facility Location / Coverage Problem)
2. Rigorous empirical verification of Diminishing Returns: f(A + e) - f(A) >= f(B + e) - f(B) for A subset B
3. Greedy Submodular Maximization under cardinality constraint |S| <= K
4. Verification of the (1 - 1/e) approx 0.6321 approximation bound vs exact brute-force optimum
"""

import numpy as np

class FacilityLocationCoverage:
    """
    Submodular facility location / sensor coverage:
    Given universe of client points C and candidate facility locations F.
    f(S) = sum_{j in C} max_{i in S} coverage_score(i, j)
    Provably monotone and submodular.
    """
    def __init__(self, n_candidates=10, n_clients=30, seed=42):
        self.rng = np.random.default_rng(seed)
        self.n = n_candidates
        self.m = n_clients
        # Coordinates in [0, 10]^2
        self.facility_coords = self.rng.uniform(0, 10, size=(self.n, 2))
        self.client_coords = self.rng.uniform(0, 10, size=(self.m, 2))
        # Coverage matrix: score = 1 / (1 + dist(i, j))
        dists = np.linalg.norm(self.facility_coords[:, None, :] - self.client_coords[None, :, :], axis=2)
        self.scores = 10.0 / (1.0 + dists)

    def evaluate(self, S):
        """Computes f(S) = sum_{j in C} max_{i in S} score(i, j). If S is empty, returns 0."""
        if len(S) == 0:
            return 0.0
        sub_scores = self.scores[list(S), :]
        max_per_client = np.max(sub_scores, axis=0)
        return float(np.sum(max_per_client))

    def marginal_gain(self, S, element):
        """Computes Delta(e | S) = f(S union {e}) - f(S)."""
        if element in S:
            return 0.0
        return self.evaluate(set(S) | {element}) - self.evaluate(S)

    def test_diminishing_returns(self, n_trials=50):
        """
        Tests whether f satisfies diminishing returns:
        For random A subset B and e not in B:
        f(A union {e}) - f(A) >= f(B union {e}) - f(B)
        Returns: (all_passed, min_slack)
        """
        all_passed = True
        min_slack = float("inf")

        elements = list(range(self.n))
        for _ in range(n_trials):
            # Sample size k_b > k_a
            k_b = self.rng.integers(2, self.n)
            k_a = self.rng.integers(1, k_b)

            b_indices = self.rng.choice(elements, size=k_b, replace=False)
            B = set(b_indices)
            a_indices = self.rng.choice(list(B), size=k_a, replace=False)
            A = set(a_indices)

            remaining = [e for e in elements if e not in B]
            if not remaining:
                continue
            e = self.rng.choice(remaining)

            gain_A = self.marginal_gain(A, e)
            gain_B = self.marginal_gain(B, e)
            slack = gain_A - gain_B
            if slack < -1e-9:
                all_passed = False
            if slack < min_slack:
                min_slack = slack

        return all_passed, min_slack

    def greedy_select(self, K):
        """
        Standard Nemhauser greedy selection:
        Starts with S = empty.
        At each step, picks e* = argmax_{e not in S} Delta(e | S).
        Terminates when |S| = K.
        Guarantees: f(S_greedy) >= (1 - 1/e) f(S*)
        """
        S = set()
        history = [0.0]

        for step in range(K):
            best_e = -1
            best_gain = -1.0
            for e in range(self.n):
                if e not in S:
                    gain = self.marginal_gain(S, e)
                    if gain > best_gain:
                        best_gain = gain
                        best_e = e

            S.add(best_e)
            history.append(self.evaluate(S))

        return S, history[-1], history

    def exact_optimal(self, K):
        """Finds true optimum S* via exhaustive search over all binom(N, K) subsets."""
        import itertools
        best_S = None
        best_val = -1.0

        for subset in itertools.combinations(range(self.n), K):
            val = self.evaluate(subset)
            if val > best_val:
                best_val = val
                best_S = set(subset)

        return best_S, best_val

