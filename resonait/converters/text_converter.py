"""
resonait/converters/text_converter.py
======================================
Konverter Teks → FrequencyTensor

Pendekatan yang digunakan:
1. Encode teks ke embedding numerik (via hash atau model embedding)
2. Terapkan FFT pada embedding untuk mendapatkan representasi frekuensi
3. Interpretasi: frekuensi rendah = makna semantik global, frekuensi tinggi = detail sintaksis
"""

import torch
import numpy as np
from typing import Any, List, Optional
from resonait.converters.base import BaseConverter
from resonait.core.frequency_space import FrequencyTensor, Modality


class TextConverter(BaseConverter):
    """
    Konverter Teks ke FrequencyTensor.
    
    Pipeline:
        Teks → Character/Subword encoding → Embedding numerik → FFT → FrequencyTensor
    
    Tidak memerlukan model embedding eksternal dalam mode dasar.
    Mode lanjutan mendukung integrasi HuggingFace embeddings.
    
    Args:
        freq_dim       (int): Target dimensi frekuensi output
        vocab_size     (int): Ukuran vocabulary untuk encoding
        embedding_dim  (int): Dimensi embedding sebelum FFT
        use_pretrained (bool): Gunakan model embedding pre-trained (memerlukan transformers)
    """
    
    def __init__(
        self,
        freq_dim:       int  = 512,
        vocab_size:     int  = 50257,  # Sama dengan GPT-2/tiktoken
        embedding_dim:  int  = 256,
        use_pretrained: bool = False,
        normalize:      bool = True,
        device:         str  = "cpu",
    ):
        super().__init__(freq_dim=freq_dim, normalize=normalize, device=device)
        
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        
        if use_pretrained:
            # Mode lanjutan: gunakan sentence transformer untuk embedding yang lebih kaya
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._mode = "pretrained"
                print("[TextConverter] ✓ Menggunakan sentence-transformers untuk embedding.")
            except ImportError:
                print("[TextConverter] ⚠ sentence-transformers tidak tersedia. Fallback ke mode hash.")
                self._mode = "hash"
        else:
            self._mode = "hash"
    
    @property
    def modality(self) -> Modality:
        return Modality.TEXT
    
    def to_frequency_tensor(self, data: str, **kwargs) -> FrequencyTensor:
        """
        Konversi string teks ke FrequencyTensor.
        
        Alur (mode hash):
        1. Encode setiap karakter ke nilai numerik
        2. Pad/crop ke panjang tetap
        3. Terapkan FFT untuk mendapatkan komponen frekuensi
        4. Bungkus dalam FrequencyTensor
        
        Args:
            data: String teks input
            
        Returns:
            FrequencyTensor dengan shape (1, 1, freq_dim)
        """
        if self._mode == "pretrained":
            return self._convert_pretrained(data)
        else:
            return self._convert_hash(data)
    
    def _convert_hash(self, text: str) -> FrequencyTensor:
        """
        Konversi teks ke frekuensi menggunakan hash karakter.
        
        Ini adalah metode cepat yang tidak memerlukan model eksternal.
        Setiap karakter di-hash ke nilai numerik, lalu FFT diterapkan.
        
        Keterbatasan: Tidak menangkap semantik, hanya statistik karakter.
        Untuk kualitas lebih baik, gunakan use_pretrained=True.
        """
        if not text:
            text = " "  # Minimal 1 karakter
        
        # Encode teks ke array numerik menggunakan ordinal values
        # Setiap karakter dipetakan ke nilai ASCII/Unicode-nya
        char_values = np.array([ord(c) for c in text], dtype=np.float32)
        
        # Normalisasi ke [-1, 1] — karakter ASCII biasanya 32-127
        char_values = (char_values - 64.0) / 64.0
        
        # Convert ke tensor PyTorch
        signal = torch.tensor(char_values, dtype=torch.float32)
        
        # Terapkan FFT untuk mendapatkan representasi frekuensi
        # Shape signal: (len(text),) → setelah FFT: (freq_dim,)
        amplitude, phase = self._apply_dft(signal.unsqueeze(0))
        # Setelah unsqueeze: (1, len) → setelah dft: (1, freq_dim)
        
        # Tambahkan batch dimension: (1, 1, freq_dim)
        amplitude = amplitude.unsqueeze(0).to(self.device)
        phase     = phase.unsqueeze(0).to(self.device)
        
        freq_tensor = FrequencyTensor(
            amplitude=amplitude,
            phase=phase,
            modality=Modality.TEXT,
            metadata={
                "original_text": text[:100],  # Simpan 100 char pertama sebagai metadata
                "text_length":   len(text),
                "encoding_mode": "hash",
            }
        )
        
        self._validate_output(freq_tensor)
        return freq_tensor
    
    def _convert_pretrained(self, text: str) -> FrequencyTensor:
        """
        Konversi teks menggunakan model embedding pre-trained.
        
        Menghasilkan representasi semantik yang jauh lebih kaya.
        Embedding dari sentence-transformer kemudian di-FFT-kan.
        """
        # Dapatkan embedding dari model pre-trained
        embedding = self._embedding_model.encode(text, convert_to_tensor=True)
        # embedding shape: (embedding_dim,)
        
        # Terapkan FFT pada embedding vector
        amplitude, phase = self._apply_dft(embedding.unsqueeze(0))
        
        amplitude = amplitude.unsqueeze(0).to(self.device)
        phase     = phase.unsqueeze(0).to(self.device)
        
        return FrequencyTensor(
            amplitude=amplitude,
            phase=phase,
            modality=Modality.TEXT,
            metadata={
                "original_text": text[:100],
                "encoding_mode": "pretrained",
            }
        )
    
    def from_frequency_tensor(self, freq_tensor: FrequencyTensor) -> str:
        """
        Rekonstruksi teks dari FrequencyTensor via IFFT.
        
        CATATAN: Rekonstruksi sempurna tidak mungkin (FFT dari karakter ordinal
        adalah representasi lossy). Untuk text generation, gunakan dekoder
        language model yang terpisah.
        
        Ini mengembalikan representasi karakter terdekat dari IFFT.
        """
        # IFFT untuk mendapatkan kembali sinyal karakter
        complex_repr = freq_tensor.complex_repr[:, 0, :]  # (batch, freq_dim)
        
        # Pad ke panjang yang diperlukan dan IFFT
        reconstructed = torch.fft.irfft(complex_repr, n=self.freq_dim * 2, norm="ortho")
        
        # Konversi kembali ke karakter (denormalisasi)
        char_values = (reconstructed[0].cpu().numpy() * 64.0 + 64.0).astype(int)
        
        # Clamp ke range karakter yang valid (32-126 untuk printable ASCII)
        char_values = np.clip(char_values, 32, 126)
        
        # Konversi ke string
        reconstructed_text = "".join([chr(c) for c in char_values])
        
        return reconstructed_text.strip()
