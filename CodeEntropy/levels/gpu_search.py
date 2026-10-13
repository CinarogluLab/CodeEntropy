"""Optional FP64 GPU candidate generation with CPU reference-distance checks.

Only candidate generation runs on CUDA. RAD blocking, tie ordering and final
distances use the unchanged CPU arithmetic. No fastmath or reduced precision.
"""
import numpy as np


def prepare_candidates(coms, dimensions, distance_function, batch_size=128):
    import cupy as cp

    n = len(coms)
    # Invalid/non-finite geometry retains reference CPU behaviour.
    if n <= 65 or not np.isfinite(coms).all() or not np.isfinite(dimensions).all():
        return None
    if np.any(dimensions <= 0):
        return None
    # Conservative bound on FP64 subtraction/wrapping/norm roundoff; comparisons
    # close to the shortlist boundary are resolved with a full CPU search.
    margin = 128*np.finfo(np.float64).eps*max(1., float(np.abs(coms).max()),
                                            float(np.abs(dimensions).max()))
    coordinates = cp.asarray(coms, dtype=cp.float64)
    box = cp.asarray(dimensions, dtype=cp.float64)
    rows = []
    for start in range(0, n, batch_size):
        stop = min(start+batch_size, n)
        delta = coordinates[None, :, :] - coordinates[start:stop, None, :]
        for d in range(3):
            delta[:, :, d] = cp.where(delta[:, :, d] > 0.5*box[d],
                                      delta[:, :, d]-box[d], delta[:, :, d])
            delta[:, :, d] = cp.where(delta[:, :, d] < -0.5*box[d],
                                      delta[:, :, d]+box[d], delta[:, :, d])
        distances = cp.sqrt((delta*delta).sum(axis=2))
        distances[cp.arange(stop-start), cp.arange(start,stop)] = cp.inf
        selected = cp.argsort(distances, axis=1)[:, :64]
        boundary = cp.asnumpy(cp.take_along_axis(distances, selected[:, -1:], axis=1))[:,0]
        selected = cp.asnumpy(selected)
        for row, mol_id in enumerate(range(start, stop)):
            # Original stable sorting breaks ties by ascending fragment index.
            ids = np.sort(selected[row])
            exact = distance_function(coms[ids], coms[mol_id], dimensions)
            order = np.argsort(exact, kind='mergesort')[:30]
            if boundary[row] - exact[order[-1]] <= margin:
                rows.append(None)  # full CPU selection for ambiguous boundaries
            else:
                rows.append((ids[order], exact[order]))
    return rows
