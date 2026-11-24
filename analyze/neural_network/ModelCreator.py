import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def read_snapshots(path, num_spins):
    # Read snapshots from a binary file, yielding one snapshot at a time
    expected = (num_spins + 63) // 64

    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if not hdr:
                return
            if len(hdr) < 8:
                raise ValueError("Truncated header")
            n = int.from_bytes(hdr, "little")
            if n != expected:
                f.seek(n * 8, 1)
                continue
            state = np.fromfile(f, dtype="<u8", count=n)
            if state.size != n:
                raise ValueError("Truncated snapshot data")
            yield state

def unpack_spins(states, num_spins):
    # (n, words) uint64 -> (n, num_spins) float32 ±1; only called on small batches
    bits = np.unpackbits(states.view(np.uint8), axis=1, bitorder="little")[:, :num_spins]
    return bits.astype(np.float32) * 2 - 1

def load_packed(path, num_spins):
    chunks = list(read_snapshots(path, num_spins))
    if not chunks:
        return np.empty((0, (num_spins + 63) // 64), dtype="<u8")
    return np.stack(chunks)

def read_temperatures(temp_file):
    temps = []
    with open(temp_file, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temps.append(float(parts[1]))
    return np.array(temps)

def load_data(DATA_PATH, N_NODES, N_RUNS, TEMP_FILE):
    temps = read_temperatures(TEMP_FILE)[1:]
    packed_by_temp = []
    for i in range(len(temps)):
        runs = [load_packed(os.path.join(DATA_PATH, f"temp_{i}/Snapshots_{r}.dat"), N_NODES)
                for r in range(N_RUNS)]
        packed_by_temp.append(np.concatenate(runs))
    return temps, packed_by_temp

# ---------- Batch generator ----------

class SpinSequence(keras.utils.Sequence):
    def __init__(self, packed, labels, idx, num_spins, batch_size=32, shuffle=False):
        super().__init__()
        self.packed, self.labels = packed, labels
        self.idx, self.n, self.bs, self.shuffle = idx.copy(), num_spins, batch_size, shuffle

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def __getitem__(self, i):
        b = self.idx[i * self.bs:(i + 1) * self.bs]
        return unpack_spins(self.packed[b], self.n), self.labels[b]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

# ---------- Training ----------

def create_and_train_model(temps, packed_by_temp, N_NODES, T_c):
    packed = np.concatenate(packed_by_temp)           # small: packed bits only
    labels = np.concatenate([np.full(len(p), t < T_c, dtype=np.float32)
                             for t, p in zip(temps, packed_by_temp)])

    idx = np.arange(len(packed))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=50, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=50,
                                          stratify=labels[train_idx])

    train_seq = SpinSequence(packed, labels, train_idx, N_NODES, shuffle=True)
    val_seq   = SpinSequence(packed, labels, val_idx, N_NODES)
    test_seq  = SpinSequence(packed, labels, test_idx, N_NODES)

    if N_NODES == 128 * 128:
        model = keras.Sequential([
            keras.Input(shape=(N_NODES,)),
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(1, activation='sigmoid'),
        ])
    else:
        model = keras.Sequential([
            keras.Input(shape=(N_NODES,)),
            layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(1, activation='sigmoid'),
        ])

    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_seq, validation_data=val_seq, epochs=2,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)],
        verbose=1,
    )
    test_loss, test_acc = model.evaluate(test_seq, verbose=0)
    print(f'Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}')
    return model, history

# ---------- Plot ----------

def get_metric(name, N, T):
    metrics = {
        "direct":         lambda M: M[1],
        "susceptibility": lambda M: (M[2] - M[1]**2) * N / T,
        "specific_heat":  lambda M: (M[2] - M[1]**2) / (N * T**2),
        "binder":         lambda M: 1 - M[4] / (3 * M[2]**2),
    }
    return metrics[name]

def block_jackknife(x, stat, n_blocks=20, return_replicates=False):
    """
    Returns the jackknife estimate of the statistic and its error.

    x: data of size (n_samples, n_T) 
    stat: function of moment dict -> (n_T,) array 
    (Ex of stat: stat = lambda M: 1.0 - M[4] / (3.0 * M[2]**2), 
    the Binder cumulant)
    
    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the block jackknife function 
    for computing statistics and errors from data.
    """
    n = (x.shape[0] // n_blocks) * n_blocks
    b = n // n_blocks
    x = x[:n]
    S = {k: (x**k).reshape(n_blocks, b, -1).sum(axis=1) for k in (1, 2, 4)}
    full = {k: S[k].sum(axis=0) / n for k in S}
    loo  = {k: (S[k].sum(axis=0) - S[k]) / (n - b) for k in S}
    theta, theta_i = stat(full), stat(loo)
    err = np.sqrt((n_blocks - 1) * np.var(theta_i, axis=0, ddof=0))
    if return_replicates:
        return theta, err, theta_i
    return theta, err

def plot_result(model, packed_by_temp, temps, N_NODES, T_c, bs=256):
    preds = []
    for p in packed_by_temp:
        preds.append(np.concatenate([model.predict_on_batch(unpack_spins(p[j:j + bs], N_NODES))
                                for j in range(0, len(p), bs)]).squeeze(-1))
    preds = np.array(preds).T
    print(preds.shape)

    mean, err = block_jackknife(preds, get_metric("direct", N_NODES, temps))
    print(len(mean))
    print(len(temps))

    plt.figure(figsize=(10, 6))
    plt.errorbar(temps, mean, yerr=err, marker='o', label='Mean prediction')
    plt.axvline(T_c, ls='--', color='k', label='$T_c$')
    plt.xlabel('Temperature')
    plt.ylabel('Mean P(ordered)')
    plt.legend()
    plt.show()

# ---------- Main ----------

if __name__ == "__main__":
    MODEL_NUM, N_RUNS, TRAIN = 2, 1, True

    for L in [64]:
        N_NODES = L * L
        BASE_DIR = '/home/sindrekampennesheim/Documents/PhD/IsingModel/NeuralNetwork/'
        MODEL_DIR = os.path.join(BASE_DIR, 'Models')
        TEMP_PATH = os.path.join(BASE_DIR, f'TraningData/square_nn_run{MODEL_NUM}/L_{L}/temp_index.txt')
        DATA_PATH = os.path.join(BASE_DIR, f'TraningData/square_nn_run{MODEL_NUM}/L_{L}/')
        T_c = 2 / np.log(1 + np.sqrt(2))
        model_path = os.path.join(MODEL_DIR, f'test_Model{MODEL_NUM}_L{L}.keras')

        temps, packed_by_temp = load_data(DATA_PATH, N_NODES, N_RUNS, TEMP_PATH)

        if TRAIN:
            model, _ = create_and_train_model(temps, packed_by_temp, N_NODES, T_c)
            model.save(model_path)
        else:
            model = keras.models.load_model(model_path)

        plot_result(model, packed_by_temp, temps, N_NODES, T_c)