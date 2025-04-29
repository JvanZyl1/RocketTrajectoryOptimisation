import numpy as np

def compute_n_step_numpy(buf, gamma, state_dim, action_dim, n):
    rew_i = state_dim + action_dim
    ns_i = rew_i + 1
    ned_i = ns_i + state_dim
    done_i = ned_i

    seq = buf[:n][::-1]

    G = 0.0
    next_s = np.zeros(state_dim, dtype=buf.dtype)
    done_seen = False

    for tr in seq:
        r = tr[rew_i]
        s_next = tr[ns_i:ned_i]
        d = tr[done_i] > 0.5

        if not done_seen:
            if d:
                G = r
                next_s = s_next
                done_seen = True
                break
            else:
                G = r + gamma * G
                next_s = s_next

    return G, next_s, float(done_seen)

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "name": "No terminal within n steps",
            "buffer": np.array([
                [10, 20, 1, 3.0, 11, 21, 0.0],
                [30, 40, 2, 5.0, 31, 41, 0.0]
            ], dtype=np.float32),
            "gamma": 0.9,
            "state_dim": 2,
            "action_dim": 1,
            "n": 2,
            "expected": {
                "reward": 3.0 + 0.9 * 5.0,
                "next_state": np.array([31, 41], dtype=np.float32),
                "done": 0.0
            }
        },
        {
            "name": "Terminal at first step",
            "buffer": np.array([
                [10, 20, 1, 3.0, 11, 21, 1.0],
                [30, 40, 2, 5.0, 31, 41, 0.0]
            ], dtype=np.float32),
            "gamma": 0.9,
            "state_dim": 2,
            "action_dim": 1,
            "n": 2,
            "expected": {
                "reward": 3.0,
                "next_state": np.array([11, 21], dtype=np.float32),
                "done": 1.0
            }
        }
    ]

    for tc in test_cases:
        G, ns, d = compute_n_step_numpy(
            tc["buffer"], tc["gamma"], tc["state_dim"], tc["action_dim"], tc["n"])
        print(f"Test: {tc['name']}")
        print(f"Computed reward: {G:.6f}, Expected reward: {tc['expected']['reward']:.6f}")
        print(f"Computed next_state: {ns}, Expected next_state: {tc['expected']['next_state']}")
        print(f"Computed done: {d}, Expected done: {tc['expected']['done']}\n")
