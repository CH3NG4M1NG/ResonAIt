"""
resonait/core/frequency_space.py
=================================
Modul ini mendefinisikan struktur data fundamental ResonAIt:
- FrequencyTensor: Representasi universal semua modalitas di domain frekuensi
- UniversalFrequencySpace: Ruang matematis tempat semua persepsi hidup berdampingan

Filosofi Desain:
    Alih-alih merepresentasikan teks sebagai token integer, gambar sebagai pixel,
    dan audio sebagai waveform — ResonAIt mengubah SEMUANYA menjadi satu bahasa:
    Amplitudo dan Fase di domain frekuensi.
    
    Ini memungkinkan operasi lintas-modalitas yang tidak mungkin dilakukan
    dengan pendekatan tokenisasi tradisional.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from enum import Enum


# ============================================================
# ENUM: Modalitas yang didukung ResonAIt
# ============================================================

class Modality(Enum):
    """
    Enum untuk semua modalitas yang dapat diproses ResonAIt.
    Penambahan modalitas baru cukup dengan menambah entry di sini
    dan mendaftarkan converter-nya via register_sensor().
    """
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    # Modalitas ekstensi — komunitas bisa menambahkan di sini
    LIDAR    = "lidar"    # Untuk robot / self-driving
    EEG      = "eeg"      # Sinyal otak (Brain-Computer Interface)
    TACTILE  = "tactile"  # Sensor sentuhan
    CUSTOM   = "custom"   # Plugin komunitas


# ============================================================
# DATACLASS: FrequencyTensor
# ============================================================

@dataclass
class FrequencyTensor:
    """
    Representasi universal untuk SEMUA jenis data di domain frekuensi.
    
    Setiap input (teks, gambar, audio) diubah menjadi dua komponen:
    - amplitude : Seberapa kuat setiap frekuensi hadir (magnitude)
    - phase     : Di mana posisi gelombang pada siklus-nya (0 sampai 2π)
    
    Bersama-sama, amplitude + phase = representasi frekuensi yang lengkap
    yang dapat merekonstruksi kembali sinyal asli via Inverse FFT.
    
    Shape standar: (batch, channels, freq_bins)
    - batch     : Jumlah sampel dalam satu batch
    - channels  : Jumlah "saluran" (e.g., RGB=3 untuk gambar, Mono=1 untuk audio)
    - freq_bins : Jumlah komponen frekuensi yang dianalisis
    
    Attributes:
        amplitude  (Tensor): Magnitude tiap komponen frekuensi. Shape: (B, C, F)
        phase      (Tensor): Fase tiap komponen frekuensi dalam radian. Shape: (B, C, F)
        modality   (Modality): Asal modalitas data ini
        metadata   (dict): Informasi tambahan (resolusi asli, sample rate, dll)
        complex_repr (Tensor, optional): Representasi kompleks amplitude*e^(i*phase)
    """
    amplitude:    torch.Tensor
    phase:        torch.Tensor
    modality:     Modality
    metadata:     Dict[str, Any] = field(default_factory=dict)
    complex_repr: Optional[torch.Tensor] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validasi dan buat representasi kompleks saat inisialisasi."""
        # Validasi shape harus sama
        assert self.amplitude.shape == self.phase.shape, (
            f"Shape amplitude {self.amplitude.shape} harus sama dengan "
            f"phase {self.phase.shape}"
        )
        # Buat representasi kompleks: A * e^(i*φ) = A*cos(φ) + i*A*sin(φ)
        # Ini adalah bentuk polar ke Kartesian
        self._build_complex_repr()
    
    def _build_complex_repr(self):
        """
        Bangun tensor kompleks dari amplitude dan phase.
        
        Menggunakan identitas Euler: A*e^(iφ) = A*(cos φ + i·sin φ)
        
        Ini memungkinkan operasi aritmatika gelombang langsung,
        seperti interference (superposisi gelombang).
        """
        real_part = self.amplitude * torch.cos(self.phase)
        imag_part = self.amplitude * torch.sin(self.phase)
        # torch.complex menggabungkan real + imaginary menjadi satu tensor
        self.complex_repr = torch.complex(real_part, imag_part)
    
    def interfere_with(self, other: "FrequencyTensor") -> "FrequencyTensor":
        """
        Lakukan interferensi gelombang antara dua FrequencyTensor.
        
        Interferensi Konstruktif: gelombang sama fase → amplitudo bertambah
        Interferensi Destruktif : gelombang berlawanan fase → amplitudo berkurang
        
        Ini adalah operasi KUNCI dalam Pain System — rasa sakit diimplementasikan
        sebagai interferensi destruktif yang mengganggu representasi frekuensi.
        
        Args:
            other: FrequencyTensor lain yang akan diinterferensikan
            
        Returns:
            FrequencyTensor hasil interferensi (superposisi gelombang)
        """
        # Tambahkan representasi kompleks — ini adalah superposisi gelombang
        combined_complex = self.complex_repr + other.complex_repr
        
        # Ekstrak amplitude dan phase dari hasil superposisi
        new_amplitude = combined_complex.abs()  # |z| = magnitude
        new_phase     = combined_complex.angle() # arg(z) = phase
        
        return FrequencyTensor(
            amplitude=new_amplitude,
            phase=new_phase,
            modality=self.modality,
            metadata={**self.metadata, "interfered_with": other.modality.value}
        )
    
    def to_power_spectrum(self) -> torch.Tensor:
        """
        Hitung power spectrum: P(f) = |A(f)|²
        
        Power spectrum menunjukkan distribusi energi di tiap frekuensi.
        Berguna untuk analisis "kesehatan" frekuensi AI — frekuensi yang
        terdistorsi oleh pain akan memiliki power spectrum yang abnormal.
        
        Returns:
            Tensor power spectrum. Shape: (B, C, F)
        """
        return self.amplitude ** 2
    
    def dominant_frequencies(self, top_k: int = 10) -> torch.Tensor:
        """
        Temukan top-K frekuensi dengan amplitudo terkuat.
        
        Ini adalah ringkasan representasi — seperti "kata kunci" dalam teks,
        tapi dalam domain frekuensi.
        
        Args:
            top_k: Jumlah frekuensi dominan yang ingin diambil
            
        Returns:
            Indeks frekuensi dominan. Shape: (B, C, top_k)
        """
        return torch.topk(self.amplitude, k=min(top_k, self.amplitude.shape[-1]), dim=-1).indices
    
    def coherence_with(self, other: "FrequencyTensor") -> float:
        """
        Ukur koherensi (keselarasan frekuensi) antara dua tensor.
        
        Nilai 1.0 = perfectly coherent (sangat selaras)
        Nilai 0.0 = completely incoherent (kacau total)
        
        Ini digunakan untuk mengukur "apakah AI memahami input ini?"
        — input yang tidak dimengerti akan menghasilkan koherensi rendah.
        
        Args:
            other: FrequencyTensor pembanding
            
        Returns:
            Skor koherensi antara 0.0 dan 1.0
        """
        # Hitung cross-spectrum
        cross_spectrum = self.complex_repr * other.complex_repr.conj()
        
        # Koherensi = |cross_spectrum|² / (power_self * power_other)
        numerator   = cross_spectrum.abs().mean()
        denominator = (self.amplitude.mean() * other.amplitude.mean()) + 1e-8
        
        return (numerator / denominator).item()
    
    @property
    def shape(self) -> tuple:
        """Shape dari tensor frekuensi."""
        return self.amplitude.shape
    
    @property
    def device(self) -> torch.device:
        """Device (CPU/GPU) tempat tensor berada."""
        return self.amplitude.device
    
    def to(self, device: torch.device) -> "FrequencyTensor":
        """Pindahkan tensor ke device tertentu (CPU/GPU)."""
        return FrequencyTensor(
            amplitude=self.amplitude.to(device),
            phase=self.phase.to(device),
            modality=self.modality,
            metadata=self.metadata
        )
    
    def __repr__(self) -> str:
        return (
            f"FrequencyTensor("
            f"modality={self.modality.value}, "
            f"shape={self.shape}, "
            f"device={self.device}, "
            f"dominant_amp={self.amplitude.max().item():.4f}"
            f")"
        )


# ============================================================
# KELAS: UniversalFrequencySpace
# ============================================================

class UniversalFrequencySpace(nn.Module):
    """
    Ruang matematis universal tempat semua modalitas hidup berdampingan.
    
    Ini adalah "lapangan bermain" utama ResonAIt — semua operasi
    (persepsi, pemikiran, memori, rasa sakit) terjadi di dalam ruang ini.
    
    Prinsip Utama:
        1. Semua modalitas diproyeksikan ke dimensi frekuensi yang sama
        2. Operasi lintas-modalitas dilakukan via manipulasi spektrum
        3. Alignment LLM berarti memetakan embedding-nya ke koordinat di sini
    
    Args:
        freq_dim   (int): Dimensi ruang frekuensi (jumlah frekuensi bins)
        hidden_dim (int): Dimensi representasi tersembunyi di dalam ruang
        n_modes    (int): Jumlah mode Fourier yang dipertahankan di FNO
    """
    
    def __init__(
        self,
        freq_dim:   int = 512,
        hidden_dim: int = 256,
        n_modes:    int = 64,
    ):
        super().__init__()
        
        self.freq_dim   = freq_dim    # Jumlah bins frekuensi
        self.hidden_dim = hidden_dim  # Dimensi representasi internal
        self.n_modes    = n_modes     # Jumlah mode Fourier yang aktif
        
        # === PROYEKTOR MODALITAS ===
        # Setiap modalitas punya proyektor berbeda untuk memetakan
        # frekuensi raw-nya ke dalam ruang universal yang sama.
        # Ini adalah "translator" antar bahasa modalitas.
        self.modality_projectors = nn.ModuleDict({
            "text" : nn.Linear(freq_dim, hidden_dim),
            "image": nn.Linear(freq_dim, hidden_dim),
            "audio": nn.Linear(freq_dim, hidden_dim),
            "video": nn.Linear(freq_dim, hidden_dim),
        })
        
        # === PROYEKTOR BALIK ===
        # Untuk merekonstruksi output ke ruang frekuensi penuh
        self.output_projector = nn.Linear(hidden_dim, freq_dim)
        
        # === LAYER NORMALISASI FREKUENSI ===
        # Menjaga skala amplitudo tetap terkontrol agar tidak explode/vanish
        self.freq_norm = nn.LayerNorm(hidden_dim)
        
        # === PHASE ENCODER ===
        # Phase mengandung informasi temporal/spasial yang sangat penting.
        # Layer ini memproses phase secara terpisah agar tidak hilang.
        self.phase_encoder = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.Tanh(),  # Tanh karena phase adalah nilai sudut (-π sampai π)
        )
        
    def project(self, freq_tensor: FrequencyTensor) -> torch.Tensor:
        """
        Proyeksikan FrequencyTensor ke dalam Universal Frequency Space.
        
        Langkah-langkah:
        1. Ambil proyektor sesuai modalitas
        2. Proyeksikan amplitude ke hidden_dim
        3. Encode phase secara terpisah
        4. Gabungkan (fusi) amplitude + phase representation
        5. Normalisasi
        
        Args:
            freq_tensor: FrequencyTensor dari converter
            
        Returns:
            Tensor di dalam Universal Frequency Space. Shape: (B, C, hidden_dim)
        """
        modality_key = freq_tensor.modality.value
        
        # Ambil proyektor yang sesuai modalitas
        # Jika modalitas belum ada, gunakan proyektor generic (bisa di-extend)
        if modality_key not in self.modality_projectors:
            # Fallback: gunakan proyektor text sebagai default
            projector = self.modality_projectors["text"]
        else:
            projector = self.modality_projectors[modality_key]
        
        # Proyeksikan amplitude (komponen energi)
        amp_projection   = projector(freq_tensor.amplitude)
        
        # Encode phase (komponen temporal/spasial)
        phase_projection = self.phase_encoder(freq_tensor.phase)
        
        # Fusi: gabungkan amplitudo + phase via penjumlahan
        # (keduanya sudah dalam hidden_dim yang sama)
        fused = amp_projection + phase_projection
        
        # Normalisasi untuk stabilitas training
        return self.freq_norm(fused)
    
    def register_modality(self, modality_name: str):
        """
        Tambahkan proyektor baru untuk modalitas custom (extension point komunitas).
        
        Args:
            modality_name: Nama modalitas baru (e.g., "lidar", "eeg")
        """
        if modality_name not in self.modality_projectors:
            new_projector = nn.Linear(self.freq_dim, self.hidden_dim)
            self.modality_projectors[modality_name] = new_projector
            print(f"[UniversalFrequencySpace] ✓ Proyektor '{modality_name}' ditambahkan.")
    
    def forward(self, freq_tensor: FrequencyTensor) -> torch.Tensor:
        """Forward pass — alias dari project() untuk kompatibilitas nn.Module."""
        return self.project(freq_tensor)
