"""
resonait/converters/base.py
============================
Kelas dasar (Abstract Base Class) untuk semua konverter modalitas.

EXTENSION POINT UTAMA UNTUK KOMUNITAS GITHUB:
    Untuk menambahkan sensor baru ke ResonAIt, buat kelas baru
    yang mewarisi BaseConverter dan implementasikan metode:
    - to_frequency_tensor(): Konversi input ke FrequencyTensor
    - from_frequency_tensor(): Rekonstruksi output dari FrequencyTensor (opsional)
    
    Kemudian daftarkan via:
    >>> from resonait import register_sensor
    >>> register_sensor("nama_sensor", KelasSensorku)
    
Contoh sensor komunitas:
    - LidarConverter (untuk robotika)
    - EEGConverter (untuk BCI)
    - ThermalConverter (untuk kamera termal)
    - HapticConverter (untuk feedback sentuhan)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import torch
import numpy as np

from resonait.core.frequency_space import FrequencyTensor, Modality


class BaseConverter(ABC):
    """
    Abstract Base Class untuk semua konverter modalitas di ResonAIt.
    
    Setiap konverter bertanggung jawab untuk:
    1. Menerima input dalam format aslinya (teks string, array gambar, dll)
    2. Mengekstrak representasi frekuensi (DFT/FFT)
    3. Membungkusnya dalam FrequencyTensor standar
    
    Komunitas dapat menambahkan konverter baru dengan mewarisi kelas ini.
    
    Args:
        freq_dim       (int): Target dimensi frekuensi output (default: 512)
        normalize      (bool): Apakah normalize amplitudo ke [0, 1]
        device         (str): Device untuk tensor ("cpu" atau "cuda")
    """
    
    def __init__(
        self,
        freq_dim:  int  = 512,
        normalize: bool = True,
        device:    str  = "cpu",
    ):
        self.freq_dim  = freq_dim
        self.normalize = normalize
        self.device    = torch.device(device)
    
    @abstractmethod
    def to_frequency_tensor(self, data: Any) -> FrequencyTensor:
        """
        [WAJIB DIIMPLEMENTASI] Konversi input ke FrequencyTensor.
        
        Ini adalah metode utama yang harus diimplementasikan oleh
        setiap konverter baru. Terima input dalam format apapun
        dan kembalikan FrequencyTensor yang terstandarisasi.
        
        Args:
            data: Data input dalam format asli modalitas ini
            
        Returns:
            FrequencyTensor dengan shape (1, channels, freq_dim)
        """
        raise NotImplementedError
    
    def from_frequency_tensor(self, freq_tensor: FrequencyTensor) -> Any:
        """
        [OPSIONAL] Rekonstruksi output dari FrequencyTensor.
        
        Implementasikan ini jika konverter kamu mendukung operasi inversi
        (merekonstruksi sinyal asli dari representasi frekuensi).
        
        Penting untuk:
        - Text generation (frekuensi → teks)
        - Image synthesis (frekuensi → gambar)  
        - Audio generation (frekuensi → audio)
        
        Args:
            freq_tensor: FrequencyTensor yang akan direkonstruksi
            
        Returns:
            Data dalam format asli modalitas ini
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} belum mengimplementasikan from_frequency_tensor(). "
            "Ini opsional tapi diperlukan untuk generasi output."
        )
    
    def _apply_dft(
        self, 
        signal: torch.Tensor, 
        target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [UTILITAS] Terapkan Discrete Fourier Transform dan ekstrak amplitude + phase.
        
        DFT adalah inti dari semua konversi di ResonAIt. Fungsi ini
        menyederhanakan proses untuk semua konverter turunan.
        
        Langkah-langkah:
        1. Padding/cropping sinyal ke target_length
        2. Terapkan rfft (FFT untuk sinyal real, lebih efisien)
        3. Ekstrak amplitude = |FFT output|
        4. Ekstrak phase = arg(FFT output) = arctan(imag/real)
        5. Normalisasi jika diminta
        
        Args:
            signal      : Tensor sinyal 1D atau 2D. Shape: (..., length)
            target_length: Panjang target setelah padding (default: 2 * freq_dim)
            
        Returns:
            Tuple (amplitude, phase), masing-masing shape (..., freq_dim)
        """
        if target_length is None:
            # Kita ingin freq_dim bins output
            # rfft menghasilkan length//2 + 1 bins, jadi kita perlu 2*freq_dim input
            target_length = self.freq_dim * 2
        
        # === Padding atau Cropping ke target_length ===
        current_length = signal.shape[-1]
        
        if current_length < target_length:
            # Padding dengan nol di akhir (zero-padding)
            # Ini adalah praktik standar dalam analisis frekuensi
            padding_size = target_length - current_length
            signal = torch.nn.functional.pad(signal, (0, padding_size))
        elif current_length > target_length:
            # Crop ke panjang yang diinginkan
            signal = signal[..., :target_length]
        
        # === Terapkan Real FFT ===
        # rfft lebih efisien dari fft penuh untuk sinyal real
        # Output shape: (..., target_length // 2 + 1) — bilangan kompleks
        fft_output = torch.fft.rfft(signal.float(), norm="ortho")
        
        # Potong ke freq_dim bins yang kita inginkan
        fft_output = fft_output[..., :self.freq_dim]
        
        # === Ekstrak Amplitude dan Phase ===
        amplitude = fft_output.abs()   # Magnitude: |z| = sqrt(real² + imag²)
        phase     = fft_output.angle() # Phase: arctan(imag/real), dalam radian
        
        # === Normalisasi Amplitude ===
        if self.normalize:
            # Normalisasi ke [0, 1] agar semua modalitas dalam skala yang sama
            max_amp = amplitude.max() + 1e-8  # Epsilon untuk menghindari div by zero
            amplitude = amplitude / max_amp
        
        return amplitude, phase
    
    def _validate_output(self, freq_tensor: FrequencyTensor) -> bool:
        """
        Validasi FrequencyTensor yang dihasilkan konverter.
        
        Memastikan output memenuhi standar ResonAIt sebelum
        dimasukkan ke dalam otak.
        
        Args:
            freq_tensor: FrequencyTensor yang akan divalidasi
            
        Returns:
            True jika valid, raise Exception jika tidak
        """
        # Cek tidak ada NaN
        if torch.isnan(freq_tensor.amplitude).any():
            raise ValueError(f"[{self.__class__.__name__}] Amplitude mengandung NaN!")
        if torch.isnan(freq_tensor.phase).any():
            raise ValueError(f"[{self.__class__.__name__}] Phase mengandung NaN!")
        
        # Cek dimensi terakhir sesuai freq_dim
        assert freq_tensor.amplitude.shape[-1] == self.freq_dim, (
            f"[{self.__class__.__name__}] Dimensi frekuensi tidak sesuai: "
            f"expected {self.freq_dim}, got {freq_tensor.amplitude.shape[-1]}"
        )
        
        return True
    
    def __call__(self, data: Any) -> FrequencyTensor:
        """
        Shortcut: langsung panggil konverter seperti fungsi.
        
        Contoh:
            >>> converter = TextConverter()
            >>> freq = converter("Halo dunia!")  # Sama dengan to_frequency_tensor()
        """
        return self.to_frequency_tensor(data)
    
    @property
    def modality(self) -> Modality:
        """Modalitas yang ditangani konverter ini (harus di-override)."""
        raise NotImplementedError("Subkelas harus mendefinisikan property modality")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"freq_dim={self.freq_dim}, "
            f"normalize={self.normalize}, "
            f"device={self.device}"
            f")"
        )
