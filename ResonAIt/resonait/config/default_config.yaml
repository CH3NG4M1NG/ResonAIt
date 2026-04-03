# resonait/config/default_config.yaml
# =====================================
# Konfigurasi default ResonAIt Brain
# Salin dan modifikasi sesuai kebutuhan

# === ARSITEKTUR BRAIN ===
freq_dim: 512        # Dimensi ruang frekuensi — semakin besar = lebih detail tapi lebih berat
hidden_dim: 256      # Dimensi representasi internal
n_modes: 64          # Mode Fourier yang dipertahankan (disarankan: freq_dim // 8)
n_fno_layers: 4      # Kedalaman encoder/decoder FNO

# === KONFIGURASI TRAINING ===
learning_rate: 1.0e-4
batch_size: 8
max_steps: 10000
warmup_steps: 200
weight_decay: 0.01

# === PAIN SYSTEM ===
pain_recovery_rate: 0.01   # Seberapa cepat pain pulih per step
max_pain: 1.0              # Batas maksimum pain (1.0 = fatal)
pain_importance_threshold: 0.5  # Minimum pain level yang mempengaruhi kognitif

# === MEMORY ===
stm_capacity: 64           # Kapasitas Short-Term Memory (jumlah FrequencyTensor)
ltm_capacity: 100000       # Kapasitas Long-Term Memory
memory_dim: 64             # Dimensi compressed memory vector
consolidate_every: 100     # Konsolidasi STM→LTM setiap N steps

# === CONVERTER ===
normalize_amplitude: true  # Normalisasi amplitude ke [0, 1]
image_target_size: [64, 64] # Ukuran target resize gambar
audio_n_fft: 1024          # FFT window size untuk audio
audio_hop_length: 256      # Hop length untuk STFT

# === ENVIRONMENT HOOK ===
n_actions: 8               # Jumlah aksi diskrit yang tersedia
network_host: "localhost"
network_port: 8765

# === LOGGING ===
log_dir: "./resonait_logs"
save_every: 1000
eval_every: 100
