"""
resonait/converters/image_converter.py
========================================
Konverter Gambar → FrequencyTensor menggunakan 2D DFT
"""
import torch
import numpy as np
from typing import Any, Union
from resonait.converters.base import BaseConverter
from resonait.core.frequency_space import FrequencyTensor, Modality


class ImageConverter(BaseConverter):
    """
    Konverter Gambar ke FrequencyTensor menggunakan 2D Discrete Fourier Transform.
    
    Gambar di-analisis menggunakan 2D FFT (sama seperti JPEG compression, tapi
    kita tidak membuang frekuensi tinggi — kita pertahankan semua informasi).
    
    Interpretasi frekuensi pada gambar:
    - Frekuensi rendah  → Kontur dan warna dominan (struktur global)
    - Frekuensi tengah  → Tepi dan tekstur utama
    - Frekuensi tinggi  → Detail halus, noise
    
    Args:
        freq_dim    (int): Target dimensi frekuensi output
        target_size (tuple): Resize gambar ke ukuran ini sebelum FFT (H, W)
    """
    
    def __init__(
        self,
        freq_dim:    int   = 512,
        target_size: tuple = (64, 64),
        normalize:   bool  = True,
        device:      str   = "cpu",
    ):
        super().__init__(freq_dim=freq_dim, normalize=normalize, device=device)
        self.target_size = target_size
    
    @property
    def modality(self) -> Modality:
        return Modality.IMAGE
    
    def to_frequency_tensor(
        self, 
        data: Union[np.ndarray, torch.Tensor, str],
        **kwargs
    ) -> FrequencyTensor:
        """
        Konversi gambar ke FrequencyTensor.
        
        Input yang diterima:
        - numpy array: shape (H, W) atau (H, W, C)
        - torch Tensor: shape (C, H, W) atau (H, W, C)
        - str/Path: path ke file gambar
        
        Alur:
        1. Load dan resize gambar ke target_size
        2. Konversi ke grayscale atau pertahankan channel RGB
        3. Terapkan 2D FFT per channel
        4. Flatten dan sample freq_dim komponen
        5. Bungkus dalam FrequencyTensor
        
        Returns:
            FrequencyTensor dengan shape (1, channels, freq_dim)
        """
        # === Step 1: Load gambar ===
        if isinstance(data, str):
            try:
                from PIL import Image as PILImage
                img = PILImage.open(data).convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0
            except ImportError:
                raise ImportError("Pillow diperlukan untuk membaca file gambar: pip install Pillow")
        elif isinstance(data, np.ndarray):
            img_array = data.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0  # Normalisasi ke [0, 1]
        elif isinstance(data, torch.Tensor):
            img_array = data.cpu().numpy().astype(np.float32)
            if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                img_array = img_array.transpose(1, 2, 0)  # CHW → HWC
        else:
            raise TypeError(f"Tipe data tidak didukung: {type(data)}")
        
        # === Step 2: Pastikan shape konsisten (H, W, C) ===
        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]  # Tambah channel dim
        
        H, W, C = img_array.shape
        
        # === Step 3: Resize ke target_size ===
        if (H, W) != self.target_size:
            try:
                import cv2
                img_array = cv2.resize(img_array, (self.target_size[1], self.target_size[0]))
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
            except ImportError:
                # Fallback sederhana tanpa cv2: crop atau pad
                th, tw = self.target_size
                img_resized = np.zeros((th, tw, C), dtype=np.float32)
                h_copy = min(H, th)
                w_copy = min(W, tw)
                img_resized[:h_copy, :w_copy] = img_array[:h_copy, :w_copy]
                img_array = img_resized
        
        H, W, C = img_array.shape
        
        # === Step 4: 2D FFT per channel ===
        all_amplitudes = []
        all_phases     = []
        
        for c in range(C):
            channel = torch.tensor(img_array[:, :, c], dtype=torch.float32)
            
            # 2D FFT: mendapatkan representasi frekuensi 2 dimensi
            fft_2d = torch.fft.rfft2(channel, norm="ortho")
            # Shape: (H, W//2 + 1) — bilangan kompleks
            
            # Flatten ke 1D
            fft_flat = fft_2d.flatten()  # (H * (W//2 + 1),)
            
            # Sample freq_dim elemen
            if len(fft_flat) >= self.freq_dim:
                # Ambil elemen-elemen dengan amplitudo terbesar (paling informatif)
                magnitudes = fft_flat.abs()
                _, top_indices = torch.topk(magnitudes, self.freq_dim)
                fft_sampled = fft_flat[top_indices]
            else:
                # Pad dengan nol jika kurang
                pad_size   = self.freq_dim - len(fft_flat)
                fft_sampled = torch.nn.functional.pad(fft_flat, (0, pad_size))
            
            amplitude = fft_sampled.abs()
            phase     = fft_sampled.angle()
            
            all_amplitudes.append(amplitude)
            all_phases.append(phase)
        
        # Stack semua channel: (C, freq_dim)
        amplitude_tensor = torch.stack(all_amplitudes, dim=0)
        phase_tensor     = torch.stack(all_phases, dim=0)
        
        # Normalisasi
        if self.normalize:
            max_amp = amplitude_tensor.max() + 1e-8
            amplitude_tensor = amplitude_tensor / max_amp
        
        # Tambah batch dimension: (1, C, freq_dim)
        amplitude_tensor = amplitude_tensor.unsqueeze(0).to(self.device)
        phase_tensor     = phase_tensor.unsqueeze(0).to(self.device)
        
        freq_tensor = FrequencyTensor(
            amplitude=amplitude_tensor,
            phase=phase_tensor,
            modality=Modality.IMAGE,
            metadata={
                "original_shape": (H, W, C),
                "target_size":    self.target_size,
                "n_channels":     C,
            }
        )
        
        self._validate_output(freq_tensor)
        return freq_tensor


# ============================================================
# AUDIO CONVERTER (disatukan di file ini untuk kemudahan)
# ============================================================

class AudioConverter(BaseConverter):
    """
    Konverter Audio → FrequencyTensor menggunakan Short-Time Fourier Transform (STFT).
    
    Audio adalah sinyal 1D yang secara natural sangat cocok untuk analisis Fourier.
    STFT digunakan (bukan FFT biasa) karena memungkinkan analisis frekuensi
    yang berubah seiring waktu — seperti spektrogram.
    
    Representasi:
    - Axis frekuensi: Pitch/nada dari audio (Hz)
    - Axis waktu    : Perubahan spektrum seiring waktu
    - Amplitude     : Loudness pada frekuensi tertentu
    
    Args:
        freq_dim    (int): Target dimensi frekuensi output
        sample_rate (int): Sample rate audio yang diharapkan (Hz)
        n_fft       (int): Panjang FFT window untuk STFT
        hop_length  (int): Step antar window STFT
    """
    
    def __init__(
        self,
        freq_dim:    int  = 512,
        sample_rate: int  = 22050,
        n_fft:       int  = 1024,
        hop_length:  int  = 256,
        normalize:   bool = True,
        device:      str  = "cpu",
    ):
        super().__init__(freq_dim=freq_dim, normalize=normalize, device=device)
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
    
    @property
    def modality(self) -> Modality:
        return Modality.AUDIO
    
    def to_frequency_tensor(
        self,
        data: Union[np.ndarray, torch.Tensor, str],
        sr:   Optional[int] = None,
        **kwargs
    ) -> FrequencyTensor:
        """
        Konversi audio ke FrequencyTensor menggunakan STFT.
        
        Input yang diterima:
        - numpy array: shape (samples,) untuk mono, (2, samples) untuk stereo
        - torch Tensor: shape (samples,) atau (channels, samples)
        - str: path ke file audio (.wav, .mp3, dll)
        
        Args:
            data: Audio data
            sr  : Sample rate (Hz). Jika None, gunakan self.sample_rate
            
        Returns:
            FrequencyTensor dengan shape (1, 1, freq_dim) untuk mono audio
        """
        actual_sr = sr or self.sample_rate
        
        # === Step 1: Load audio ===
        if isinstance(data, str):
            try:
                import librosa
                audio_array, actual_sr = librosa.load(data, sr=actual_sr, mono=True)
            except ImportError:
                raise ImportError("librosa diperlukan untuk membaca file audio: pip install librosa")
        elif isinstance(data, np.ndarray):
            audio_array = data.astype(np.float32)
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=0)  # Stereo → Mono
        elif isinstance(data, torch.Tensor):
            audio_array = data.cpu().numpy().astype(np.float32)
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=0)
        else:
            raise TypeError(f"Tipe data tidak didukung: {type(data)}")
        
        # === Step 2: Normalisasi amplitude audio ===
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # === Step 3: Terapkan STFT ===
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        # torch.stft menghasilkan spektrogram kompleks
        stft_output = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft),  # Hann window untuk sidelobe suppression
            normalized=True,
        )
        # Shape: (freq_bins, time_frames) — bilangan kompleks
        # freq_bins = n_fft//2 + 1
        
        # === Step 4: Ambil magnitude dan phase spektrogram ===
        magnitude    = stft_output.abs()   # (freq_bins, time_frames)
        phase_stft   = stft_output.angle() # (freq_bins, time_frames)
        
        # === Step 5: Rata-ratakan over time → mendapatkan profil frekuensi rata-rata ===
        # (Untuk kasus di mana kita ingin satu vektor frekuensi per klip audio)
        avg_magnitude = magnitude.mean(dim=-1)  # (freq_bins,)
        avg_phase     = phase_stft.mean(dim=-1)  # (freq_bins,)
        
        # === Step 6: Sample/interpolasi ke freq_dim ===
        freq_bins = avg_magnitude.shape[0]
        
        if freq_bins != self.freq_dim:
            # Interpolasi linear ke freq_dim target
            avg_magnitude = torch.nn.functional.interpolate(
                avg_magnitude.unsqueeze(0).unsqueeze(0),
                size=self.freq_dim,
                mode='linear',
                align_corners=False
            ).squeeze()
            avg_phase = torch.nn.functional.interpolate(
                avg_phase.unsqueeze(0).unsqueeze(0),
                size=self.freq_dim,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Normalisasi
        if self.normalize:
            max_amp = avg_magnitude.max() + 1e-8
            avg_magnitude = avg_magnitude / max_amp
        
        # Tambah batch dan channel dimension: (1, 1, freq_dim)
        amplitude_tensor = avg_magnitude.unsqueeze(0).unsqueeze(0).to(self.device)
        phase_tensor     = avg_phase.unsqueeze(0).unsqueeze(0).to(self.device)
        
        freq_tensor = FrequencyTensor(
            amplitude=amplitude_tensor,
            phase=phase_tensor,
            modality=Modality.AUDIO,
            metadata={
                "sample_rate": actual_sr,
                "duration":    len(audio_array) / actual_sr,
                "n_fft":       self.n_fft,
                "hop_length":  self.hop_length,
            }
        )
        
        self._validate_output(freq_tensor)
        return freq_tensor
