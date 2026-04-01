"""
resonait/core/brain.py
=======================
OTAK UTAMA RESONAIT — AGI Core menggunakan Fourier Neural Operator (FNO)

Modul ini mengimplementasikan ResonAItBrain: sistem saraf tiruan yang:
1. Memproses SEMUA persepsi di domain frekuensi (bukan token/pixel/waveform)
2. Menjalankan modul Imajinasi, Logika, dan Memori secara PARALEL
3. Bisa "merasakan sakit" melalui interferensi frekuensi destruktif
4. Bisa di-upgrade dari LLM privat manapun via AlignmentTool

Arsitektur Fourier Neural Operator (FNO):
    Input → FFT → Filter di domain frekuensi → IFFT → Output
    
    Keunggulan FNO dibanding Transformer biasa:
    - Kompleksitas O(N log N) vs O(N²) pada Attention
    - Secara alami memproses sinyal periodik
    - Resolution-invariant (tidak bergantung pada ukuran input)

Referensi: Li et al., "Fourier Neural Operator for Parametric PDEs" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import yaml

from resonait.core.frequency_space import (
    FrequencyTensor, UniversalFrequencySpace, Modality
)


# ============================================================
# KELAS INTI: SpectralConvolution
# ============================================================

class SpectralConvolution(nn.Module):
    """
    Lapisan konvolusi di domain FREKUENSI — inti dari Fourier Neural Operator.
    
    Alih-alih konvolusi biasa di domain spasial (menggeser kernel di atas pixel),
    SpectralConvolution melakukan PERKALIAN di domain frekuensi.
    
    Teorema Konvolusi: konvolusi di domain spasial = perkalian di domain frekuensi
    Ini jauh lebih efisien: O(N log N) vs O(N²)
    
    Operasi:
        1. Input x → FFT → X(f) [domain frekuensi]
        2. X(f) * W(f) [filter belajar di domain frekuensi]
        3. Hasil → IFFT → output di domain spasial
    
    Args:
        in_channels  (int): Jumlah channel input
        out_channels (int): Jumlah channel output
        n_modes      (int): Jumlah mode Fourier yang dipertahankan
                           (mode high-freq biasanya noise, kita potong)
    """
    
    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes  # Hanya n_modes frekuensi pertama yang diproses
        
        # === WEIGHT KOMPLEKS ===
        # Bobot filter di domain frekuensi adalah bilangan kompleks
        # (karena FFT menghasilkan bilangan kompleks)
        # Shape: (in_channels, out_channels, n_modes)
        scale = 1.0 / (in_channels * out_channels)  # Inisialisasi Xavier-like
        
        # Komponen real dari bobot filter frekuensi
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )
        # Komponen imajiner dari bobot filter frekuensi
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )
    
    def complex_multiply(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perkalian bilangan kompleks: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        
        Dilakukan secara manual karena lebih efisien di PyTorch
        daripada menggunakan torch.complex langsung.
        
        Args:
            x_real, x_imag: Bagian real dan imajiner dari input
            w_real, w_imag: Bagian real dan imajiner dari bobot
            
        Returns:
            Tuple (hasil_real, hasil_imag) dari perkalian kompleks
        """
        # Gunakan einsum untuk efisiensi: 'bix,iox->box'
        # b=batch, i=in_channel, o=out_channel, x=freq_mode
        out_real = (
            torch.einsum('bix,iox->box', x_real, w_real) -
            torch.einsum('bix,iox->box', x_imag, w_imag)
        )
        out_imag = (
            torch.einsum('bix,iox->box', x_real, w_imag) +
            torch.einsum('bix,iox->box', x_imag, w_real)
        )
        return out_real, out_imag
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass SpectralConvolution.
        
        Alur:
        x [domain spasial] → rfft → filter di frekuensi → irfft → output
        
        Args:
            x: Tensor input. Shape: (batch, in_channels, length)
            
        Returns:
            Tensor output setelah konvolusi spektral. Shape: (batch, out_channels, length)
        """
        batch_size, in_ch, length = x.shape
        
        # Step 1: FFT ke domain frekuensi
        # rfft hanya menghasilkan frekuensi positif (lebih efisien untuk sinyal real)
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        # Shape x_ft: (batch, in_channels, length//2 + 1) — kompleks
        
        # Step 2: Potong hanya n_modes pertama
        # Mode frekuensi tinggi biasanya noise — kita filter out
        x_ft_modes = x_ft[:, :, :self.n_modes]
        
        # Step 3: Perkalian dengan bobot filter (di domain frekuensi)
        # Pisahkan real dan imajiner untuk perkalian kompleks efisien
        out_real, out_imag = self.complex_multiply(
            x_ft_modes.real, x_ft_modes.imag,
            self.weight_real, self.weight_imag
        )
        
        # Step 4: Rekonstruksi tensor kompleks dan kembalikan ke panjang semula
        out_ft = torch.zeros(
            batch_size, self.out_channels, length // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.n_modes] = torch.complex(out_real, out_imag)
        
        # Step 5: IFFT kembali ke domain spasial
        x_out = torch.fft.irfft(out_ft, n=length, dim=-1, norm="ortho")
        # Shape: (batch, out_channels, length)
        
        return x_out


# ============================================================
# KELAS: FourierNeuralOperatorBlock
# ============================================================

class FourierNeuralOperatorBlock(nn.Module):
    """
    Satu blok Fourier Neural Operator (FNO Block).
    
    Setiap blok menggabungkan:
    - SpectralConv: Memproses komponen frekuensi (informasi global)
    - Conv1D biasa: Memproses komponen lokal (informasi lokal)
    - Residual connection: Menjaga aliran gradient
    - Aktivasi GELU: Non-linearitas yang mulus
    
    Ini adalah building block utama yang bisa ditumpuk untuk
    membuat jaringan lebih dalam (lebih ekspresif).
    
    Args:
        channels (int): Jumlah channels di seluruh blok
        n_modes  (int): Jumlah mode Fourier yang dipertahankan
    """
    
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        
        # Konvolusi spektral (domain frekuensi)
        self.spectral_conv = SpectralConvolution(channels, channels, n_modes)
        
        # Konvolusi biasa (domain spasial) — menangkap hubungan lokal
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        
        # Normalisasi setelah penjumlahan
        self.norm = nn.LayerNorm(channels)
        
        # Aktivasi non-linear
        self.activation = nn.GELU()
        
        # Proyektor feed-forward (seperti FFN di Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass satu FNO Block.
        
        x → [SpectralConv + LocalConv] → Residual → Norm → FFN → Residual
        
        Args:
            x: Tensor. Shape: (batch, channels, length)
            
        Returns:
            Tensor. Shape: (batch, channels, length)
        """
        # Cabang 1: Konvolusi spektral (global, domain frekuensi)
        x_spectral = self.spectral_conv(x)
        
        # Cabang 2: Konvolusi lokal (lokal, domain spasial)
        x_local = self.local_conv(x)
        
        # Gabungkan keduanya + residual connection
        x = self.activation(x_spectral + x_local) + x
        
        # FFN dengan residual (seperti di Transformer)
        # Perlu transpose karena LayerNorm dan Linear bekerja di dim terakhir
        x_t = x.transpose(-1, -2)           # (batch, length, channels)
        x_t = self.norm(x_t)
        x_t = x_t + self.ffn(x_t)           # Residual FFN
        x = x_t.transpose(-1, -2)           # (batch, channels, length) — balik lagi
        
        return x


# ============================================================
# KELAS: CognitiveModule (Imajinasi, Logika, Memori)
# ============================================================

class CognitiveModule(nn.Module):
    """
    Modul kognitif yang bekerja di domain frekuensi.
    
    Tiga modul kognitif utama ResonAIt:
    - LogicModule     : Analisis kausal dan deduktif
    - ImaginationModule: Generasi dan skenario hipotetis  
    - MemoryModule    : Konsolidasi dan retrieval memori jangka panjang
    
    Ketiga modul ini berjalan PARALEL di dalam satu spektrum frekuensi.
    Setiap modul "mengklaim" range frekuensi yang berbeda:
    - Logika     → frekuensi rendah  (pola stabil, berulang)
    - Imajinasi  → frekuensi tengah  (variasi kreatif)
    - Memori     → frekuensi tinggi  (detail spesifik)
    
    Args:
        channels (int): Jumlah channels representasi
        n_modes  (int): Mode Fourier untuk modul ini
        role     (str): "logic", "imagination", atau "memory"
    """
    
    def __init__(self, channels: int, n_modes: int, role: str):
        super().__init__()
        
        self.role = role
        
        # Setiap modul punya tumpukan FNO block sendiri
        self.fno_blocks = nn.ModuleList([
            FourierNeuralOperatorBlock(channels, n_modes)
            for _ in range(2)  # 2 layer per modul kognitif
        ])
        
        # Gate: seberapa besar kontribusi modul ini pada output akhir
        # Ini memungkinkan otak belajar untuk "fokus" ke modul tertentu
        self.gate = nn.Sequential(
            nn.Linear(channels, 1),
            nn.Sigmoid()  # Output 0.0 sampai 1.0
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass modul kognitif.
        
        Args:
            x: Tensor representasi frekuensi. Shape: (batch, channels, length)
            
        Returns:
            Tuple:
            - output tensor setelah diproses modul ini
            - gate_value: seberapa besar modul ini "aktif" (skalar per batch)
        """
        # Proses melalui tumpukan FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Hitung gate value — seberapa "percaya diri" modul ini dengan outputnya
        # Rata-rata over length, lalu gate
        gate_value = self.gate(x.mean(dim=-1))  # (batch, 1)
        
        return x, gate_value


# ============================================================
# KELAS UTAMA: ResonAItBrain
# ============================================================

class ResonAItBrain(nn.Module):
    """
    OTAK UTAMA RESONAIT — Titik masuk API untuk semua operasi AGI.
    
    ResonAItBrain mengorkestrasikan:
    1. Persepsi multimodal via UniversalFrequencySpace
    2. Pemrosesan paralel oleh 3 modul kognitif
    3. Integrasi "active binding" — menggabungkan semua modul
    4. Output generasi di domain frekuensi (bisa dikonversi ke teks/gambar/audio)
    5. Pain system — penerimaan sinyal dissonance dari environment
    
    Args:
        freq_dim   (int): Dimensi ruang frekuensi (default: 512)
        hidden_dim (int): Dimensi representasi internal (default: 256)
        n_modes    (int): Mode Fourier yang dipertahankan (default: 64)
        n_fno_layers (int): Jumlah lapisan FNO utama (default: 4)
    """
    
    def __init__(
        self,
        freq_dim:     int = 512,
        hidden_dim:   int = 256,
        n_modes:      int = 64,
        n_fno_layers: int = 4,
    ):
        super().__init__()
        
        self.freq_dim     = freq_dim
        self.hidden_dim   = hidden_dim
        self.n_modes      = n_modes
        self.n_fno_layers = n_fno_layers
        
        # === LAPISAN 1: UNIVERSAL FREQUENCY SPACE ===
        # Menerima FrequencyTensor dari converter dan memproyeksikannya
        # ke dalam ruang frekuensi universal yang sama
        self.frequency_space = UniversalFrequencySpace(
            freq_dim=freq_dim,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
        )
        
        # === LAPISAN 2: ENCODER UTAMA (FNO Stack) ===
        # Memproses representasi yang sudah diproyeksikan
        # Ini adalah "pemahaman awal" sebelum modul kognitif mengambil alih
        self.encoder = nn.ModuleList([
            FourierNeuralOperatorBlock(hidden_dim, n_modes)
            for _ in range(n_fno_layers)
        ])
        
        # === LAPISAN 3: MODUL KOGNITIF PARALEL ===
        # Tiga modul yang berjalan secara paralel (Active Binding)
        # Masing-masing mengolah representasi dari sudut pandang berbeda
        self.logic_module      = CognitiveModule(hidden_dim, n_modes // 4, "logic")
        self.imagination_module = CognitiveModule(hidden_dim, n_modes // 2, "imagination")
        self.memory_module     = CognitiveModule(hidden_dim, n_modes,       "memory")
        
        # === LAPISAN 4: ACTIVE BINDING ===
        # Menggabungkan output ketiga modul kognitif menjadi satu representasi
        # "Active Binding" = proses otak menyatukan berbagai jenis pemrosesan
        self.binding_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # Gabung 3 modul
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),       # Reduksi ke ukuran standar
            nn.LayerNorm(hidden_dim),
        )
        
        # === LAPISAN 5: DECODER UTAMA ===
        # Mengubah representasi internal kembali ke domain frekuensi output
        self.decoder = nn.ModuleList([
            FourierNeuralOperatorBlock(hidden_dim, n_modes)
            for _ in range(2)
        ])
        
        # === LAPISAN 6: OUTPUT PROJECTION ===
        # Proyeksikan ke ruang output yang diinginkan
        self.output_projection = nn.Linear(hidden_dim, freq_dim)
        
        # === PAIN RECEIVER ===
        # Layer khusus yang menerima sinyal dissonance dari environment
        # Sinyal pain mengubah state internal otak secara langsung
        self.pain_receiver = nn.Linear(freq_dim, hidden_dim)
        
        # Pain state: seberapa parah "sakit" yang sedang dirasakan (0.0 = normal)
        self.register_buffer('pain_state', torch.zeros(1))
        
        # === HEALTH MONITOR ===
        # Melacak "kesehatan" spektrum frekuensi otak
        self.register_buffer('spectral_health', torch.ones(freq_dim))
        
    @classmethod
    def from_config(cls, config_path: str) -> "ResonAItBrain":
        """
        Buat ResonAItBrain dari file konfigurasi YAML.
        
        Contoh config.yaml:
            freq_dim: 512
            hidden_dim: 256
            n_modes: 64
            n_fno_layers: 4
        
        Args:
            config_path: Path ke file YAML
            
        Returns:
            Instance ResonAItBrain yang sudah dikonfigurasi
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def perceive(self, freq_tensor: FrequencyTensor) -> torch.Tensor:
        """
        Proses persepsi multimodal: FrequencyTensor → representasi internal.
        
        Ini adalah "pintu masuk" semua input ke otak ResonAIt.
        
        Args:
            freq_tensor: FrequencyTensor dari salah satu converter
            
        Returns:
            Representasi internal di Universal Frequency Space.
            Shape: (batch, hidden_dim, freq_dim)
        """
        # Proyeksikan ke ruang frekuensi universal
        # Output: (batch, length, hidden_dim)
        projected = self.frequency_space.project(freq_tensor)
        
        # Reshape untuk FNO: perlu (batch, channels, length)
        # Anggap setiap posisi frekuensi sebagai "token" dengan hidden_dim channels
        x = projected.transpose(-1, -2)  # (batch, hidden_dim, freq_dim)
        
        return x
    
    def think(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Proses kognitif utama: representasi → output + metadata kognitif.
        
        Alur:
        1. Encode via FNO Stack
        2. Jalankan 3 modul kognitif secara PARALEL
        3. Active Binding — gabungkan semua modul
        4. Decode dan proyeksikan ke output
        
        Args:
            x: Representasi dari perceive(). Shape: (batch, hidden_dim, length)
            
        Returns:
            Dict berisi:
            - "output"     : Tensor output. Shape: (batch, hidden_dim, length)
            - "logic_gate" : Seberapa aktif modul logika
            - "imagination_gate": Seberapa aktif modul imajinasi
            - "memory_gate": Seberapa aktif modul memori
            - "pain_penalty": Reduksi akibat pain state saat ini
        """
        # === ENCODING ===
        # Proses input melalui tumpukan encoder FNO
        for encoder_block in self.encoder:
            x = encoder_block(x)
        
        # Simpan representasi sebelum modul kognitif (untuk residual connection)
        x_before_cognitive = x
        
        # === ACTIVE BINDING: PARALEL KOGNITIF ===
        # Ketiga modul memproses input yang SAMA secara paralel
        logic_out,      logic_gate      = self.logic_module(x)
        imagination_out, imagination_gate = self.imagination_module(x)
        memory_out,     memory_gate     = self.memory_module(x)
        
        # Gabungkan output ketiga modul dengan weighting dari gate
        # Gate memastikan modul yang "lebih relevan" berkontribusi lebih besar
        logic_weighted      = logic_out      * logic_gate.unsqueeze(-1)
        imagination_weighted = imagination_out * imagination_gate.unsqueeze(-1)
        memory_weighted     = memory_out     * memory_gate.unsqueeze(-1)
        
        # Concatenate di dimensi channel untuk binding layer
        # Shape: (batch, hidden_dim * 3, length)
        combined = torch.cat([
            logic_weighted,
            imagination_weighted,
            memory_weighted
        ], dim=1)
        
        # === ACTIVE BINDING LAYER ===
        # Transpose untuk Linear layer, proses, lalu transpose balik
        combined_t = combined.transpose(-1, -2)     # (batch, length, hidden_dim*3)
        bound      = self.binding_layer(combined_t) # (batch, length, hidden_dim)
        bound      = bound.transpose(-1, -2)         # (batch, hidden_dim, length)
        
        # Residual connection dari sebelum modul kognitif
        x = bound + x_before_cognitive
        
        # === PAIN PENALTY ===
        # Jika ada pain state, aplikasikan gangguan pada representasi
        if self.pain_state.item() > 0.0:
            pain_noise = self._apply_pain_perturbation(x)
            x = x + pain_noise
        
        # === DECODING ===
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        return {
            "output":            x,
            "logic_gate":        logic_gate.mean().item(),
            "imagination_gate":  imagination_gate.mean().item(),
            "memory_gate":       memory_gate.mean().item(),
            "pain_level":        self.pain_state.item(),
        }
    
    def _apply_pain_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplikasikan gangguan frekuensi (noise) akibat pain state.
        
        Semakin tinggi pain_state, semakin besar gangguan pada
        representasi internal. Ini mensimulasikan bagaimana "rasa sakit"
        mengganggu kemampuan kognitif.
        
        Pain diimplementasikan sebagai noise yang proporsional dengan pain_state
        dan berbanding terbalik dengan spectral_health.
        
        Args:
            x: Tensor representasi. Shape: (batch, hidden_dim, length)
            
        Returns:
            Noise tensor untuk ditambahkan ke representasi
        """
        pain_intensity = self.pain_state.item()
        
        # Noise proporsional dengan pain intensity
        # Gunakan Gaussian noise yang scaled oleh pain level
        noise = torch.randn_like(x) * pain_intensity * 0.1
        
        return noise
    
    def receive_pain(self, dissonance_signal: torch.Tensor, intensity: float):
        """
        Terima sinyal dissonance dari environment (Pain System API).
        
        Dipanggil oleh DissonanceEngine atau SimulatedEnvironmentHook
        ketika AI menerima "damage" dari environment.
        
        Args:
            dissonance_signal: FrequencyTensor yang merepresentasikan sinyal sakit
            intensity: Kekuatan rasa sakit (0.0 sampai 1.0)
        """
        # Update pain state (exponential moving average untuk decay alami)
        # Pain tidak hilang seketika — dia memudar perlahan
        decay_rate = 0.9
        new_pain = self.pain_state * decay_rate + torch.tensor(intensity) * (1 - decay_rate)
        self.pain_state.copy_(new_pain.clamp(0.0, 1.0))
        
        # Perbarui spectral health — frekuensi yang terganggu menjadi kurang sehat
        # Ini akan mempengaruhi pemrosesan selanjutnya
        if dissonance_signal is not None:
            health_damage = dissonance_signal.to(self.spectral_health.device)
            if health_damage.shape == self.spectral_health.shape:
                self.spectral_health = (
                    self.spectral_health - health_damage * intensity * 0.1
                ).clamp(0.1, 1.0)  # Minimum 10% kesehatan untuk mencegah collapse total
        
    def recover(self, amount: float = 0.05):
        """
        Pulihkan kesehatan otak dari pain secara bertahap.
        
        Dipanggil saat AI tidak sedang menerima damage — mensimulasikan
        proses recovery/healing secara alami.
        
        Args:
            amount: Seberapa banyak recovery per step (default: 5%)
        """
        # Kurangi pain state
        self.pain_state = (self.pain_state - amount).clamp(0.0, 1.0)
        # Pulihkan spectral health
        self.spectral_health = (self.spectral_health + amount).clamp(0.0, 1.0)
    
    def get_health_report(self) -> Dict[str, float]:
        """
        Laporan kesehatan sistem otak saat ini.
        
        Returns:
            Dict berisi metrik kesehatan:
            - pain_level     : Level rasa sakit saat ini
            - spectral_health: Rata-rata kesehatan spektrum frekuensi
            - cognitive_stability: Estimasi kestabilan kognitif (0.0-1.0)
        """
        spectral_health_avg = self.spectral_health.mean().item()
        pain              = self.pain_state.item()
        cognitive_stability = spectral_health_avg * (1.0 - pain)
        
        return {
            "pain_level":          pain,
            "spectral_health":     spectral_health_avg,
            "cognitive_stability": cognitive_stability,
        }
    
    def forward(self, freq_tensor: FrequencyTensor) -> Dict[str, Any]:
        """
        Forward pass lengkap: FrequencyTensor → output dict.
        
        Menggabungkan perceive() + think() dalam satu panggilan.
        
        Args:
            freq_tensor: Input dari converter manapun
            
        Returns:
            Dict output dari think() ditambah representasi internal
        """
        # Persepsi: ubah frekuensi tensor ke representasi internal
        perception = self.perceive(freq_tensor)
        
        # Pemikiran: jalankan modul kognitif paralel
        result = self.think(perception)
        
        # Tambahkan info modalitas ke output
        result["modality"] = freq_tensor.modality.value
        
        return result
    
    def save(self, path: str):
        """Simpan checkpoint otak."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'freq_dim': self.freq_dim,
                'hidden_dim': self.hidden_dim,
                'n_modes': self.n_modes,
                'n_fno_layers': self.n_fno_layers,
            }
        }, path)
        print(f"[ResonAItBrain] ✓ Checkpoint disimpan ke: {path}")
    
    @classmethod
    def load(cls, path: str) -> "ResonAItBrain":
        """Muat checkpoint otak dari file."""
        checkpoint = torch.load(path, map_location='cpu')
        brain = cls(**checkpoint['config'])
        brain.load_state_dict(checkpoint['model_state_dict'])
        print(f"[ResonAItBrain] ✓ Checkpoint dimuat dari: {path}")
        return brain
