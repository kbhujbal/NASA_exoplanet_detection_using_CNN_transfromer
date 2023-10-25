import numpy as np


def augment_sample(
    global_view: np.ndarray,
    local_view: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global_view = global_view.copy()
    local_view = local_view.copy()

    # 1. Time shift
    if np.random.random() < 0.5:
        global_shift = np.random.randint(-200, 201)
        # Scale shift proportionally to local view length
        local_shift = int(round(global_shift * (local_view.shape[0] / global_view.shape[0])))
        global_view = np.roll(global_view, global_shift, axis=0)
        local_view = np.roll(local_view, local_shift, axis=0)

    # 2. Gaussian noise
    if np.random.random() < 0.5:
        global_view = global_view + np.random.normal(0, 0.001, size=global_view.shape)
        local_view = local_view + np.random.normal(0, 0.001, size=local_view.shape)

    # 3. Flux scaling
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.98, 1.02)
        global_view = global_view * scale
        local_view = local_view * scale

    return global_view, local_view
