"""Sequence shuffling models used to build null distributions for motif counts."""
import random
from collections import deque, defaultdict

__all__ = ["random_shuffle", "shuffle_preserve", "kandel_uniform_k_shuffle"]


def random_shuffle(seq):
    seq_list = list(seq)
    random.shuffle(seq_list)
    return seq_list

def shuffle_preserve(seq, k_node):
    n = len(seq)
    if k_node == 0:
        return random_shuffle(seq)
    if n <= k_node:
        return seq.copy()

    graph = defaultdict(list)
    for i in range(n - k_node):
        prefix = tuple(seq[i:i+k_node])
        suffix = tuple(seq[i+1:i+k_node+1])
        graph[prefix].append(suffix)

    path = []
    stack = [tuple(seq[:k_node])]
    while stack:
        v = stack[-1]
        if graph[v]:
            u = random.choice(graph[v])
            graph[v].remove(u)
            stack.append(u)
        else:
            path.append(stack.pop())

    path = path[::-1]
    out = list(path[0])
    for node in path[1:]:
        out.append(node[-1])
    return out

def kandel_uniform_k_shuffle(sequence, k):

    if k <= 1:
        res = list(sequence)
        random.shuffle(res)
        return "".join(res) if isinstance(sequence, str) else res

    # 1. Build De Bruijn graph (multiset of edges)
    adj = defaultdict(list)
    # nodes kept in order of first appearance (a dict, not a set): a set of string symbols
    # is iterated in a different order in every Python process, which would make the
    # shuffle irreproducible even with a fixed seed
    nodes = {}
    for i in range(len(sequence) - k + 1):
        u = tuple(sequence[i     : i + k - 1])
        v = tuple(sequence[i + 1 : i + k    ])
        adj[u].append(v)
        nodes[u] = None
        nodes[v] = None

    source = tuple(sequence[:k - 1])
    sink   = tuple(sequence[-(k - 1):])

    # 2. Build a random arborescence rooted at sink
    #    using Wilson's loop-erased random walk.
    #    tree[node] = one chosen "last" edge (next hop toward sink)
    in_tree = {sink}
    tree_next = {}  # node -> next node toward sink

    for start in nodes:
        if start in in_tree:
            continue
        # Random walk from start until we hit a node in the tree
        path = [start]
        visited_idx = {start: 0}
        curr = start
        while curr not in in_tree:
            neighbors = adj[curr]
            if not neighbors:
                break
            nxt = random.choice(neighbors)
            if nxt in visited_idx:
                # Loop-erase: remove the cycle
                cycle_start = visited_idx[nxt]
                for node in path[cycle_start + 1:]:
                    del visited_idx[node]
                path = path[:cycle_start + 1]
            else:
                visited_idx[nxt] = len(path)
                path.append(nxt)
            curr = nxt

        # Commit path to tree
        for i in range(len(path) - 1):
            tree_next[path[i]] = path[i + 1]
            in_tree.add(path[i])
        if path[-1] not in in_tree:
            in_tree.add(path[-1])

    # 3. For each node (except sink), shuffle edges but keep the
    #    arborescence edge last (Kandel's "last-edge" reservation).
    shuffled_adj = {}
    for node in nodes:
        edges = list(adj[node])  # copy — don't mutate original
        if not edges:
            continue
        if node != sink and node in tree_next:
            reserved = tree_next[node]
            edges.remove(reserved)        # pull out the reserved edge
            random.shuffle(edges)
            edges.append(reserved)        # push it to the end
        else:
            random.shuffle(edges)
        shuffled_adj[node] = deque(edges)

    # 4. Hierholzer's algorithm for Eulerian path (source -> sink)
    stack = [source]
    path  = []
    while stack:
        v = stack[-1]
        if v in shuffled_adj and shuffled_adj[v]:
            stack.append(shuffled_adj[v].popleft())
        else:
            path.append(stack.pop())

    path.reverse()

    # 5. Reconstruct sequence from node path
    if len(path) < 1:
        return sequence  # fallback — shouldn't happen on valid input
    out = list(path[0])
    for node in path[1:]:
        out.append(node[-1])

    return "".join(out) if isinstance(sequence, str) else out
