"""
resonait/converters/universal_converter.py
==========================================
Konverter Universal — satu pintu untuk semua modalitas.

UniversalFrequencyConverter adalah orkestrator konversi:
- Menerima input dalam format APAPUN
- Mendeteksi modalitas secara otomatis
- Mendelegasikan ke konverter yang tepat
- Mengembalikan FrequencyTensor yang terstandarisasi

Ini juga mengimplementasikan:
- Autonomous Data Sourcing: jika hanya punya data teks, secara otomatis
  mencari/mensimulasikan data gambar dan audio pendukung untuk multimodal training
"""

import torch
import numpy as np
from typing import Any, Optional, Union, List, Dict
from pathlib import Path

from resonait.converters.base import BaseConverter
from resonait.core.frequency_space import FrequencyTensor, Modality


class UniversalFrequencyConverter:
    """
    Konverter universal yang menyatukan semua modalitas dalam satu API.
    
    Penggunaan Sederhana:
        >>> converter = UniversalFrequencyConverter()
        >>> freq = converter.convert("Halo dunia!")          # Otomatis text
        >>> freq = converter.convert(image_array)            # Otomatis image
        >>> freq = converter.convert(audio_array, sr=22050)  # Otomatis audio
        >>> freq = converter.convert(data, modality="lidar") # Plugin komunitas
    
    Args:
        freq_dim (int): Dimensi frekuensi output (default: 512)
        device   (str): Device untuk komputasi
    """
    
    def __init__(self, freq_dim: int = 512, device: str = "cpu"):
        self.freq_dim = freq_dim
        self.device   = device
        
        # Lazy import untuk menghindari circular imports
        self._converters: Dict[str, BaseConverter] = {}
        self._init_default_converters()
    
    def _init_default_converters(self):
        """Inisialisasi konverter bawaan."""
        from resonait.converters.text_converter  import TextConverter
        from resonait.converters.image_converter import ImageConverter
        from resonait.converters.audio_converter import AudioConverter
        
        self._converters = {
            "text" : TextConverter(freq_dim=self.freq_dim, device=self.device),
            "image": ImageConverter(freq_dim=self.freq_dim, device=self.device),
            "audio": AudioConverter(freq_dim=self.freq_dim, device=self.device),
        }
    
    def register(self, name: str, converter: BaseConverter):
        """Daftarkan konverter baru (extension point komunitas)."""
        self._converters[name] = converter
        print(f"[UniversalConverter] ✓ Konverter '{name}' terdaftar.")
    
    def convert(
        self,
        data: Any,
        modality: Optional[str] = None,
        **kwargs
    ) -> FrequencyTensor:
        """
        Konversi input ke FrequencyTensor secara universal.
        
        Jika modality tidak disebutkan, akan auto-detect berdasarkan tipe data.
        
        Args:
            data    : Input data dalam format apapun
            modality: (opsional) Nama modalitas yang diinginkan
            **kwargs: Argumen tambahan untuk konverter spesifik (e.g., sr untuk audio)
            
        Returns:
            FrequencyTensor terstandarisasi
        """
        # Auto-detect modalitas jika tidak disebutkan
        if modality is None:
            modality = self._auto_detect(data)
        
        if modality not in self._converters:
            raise ValueError(
                f"Modalitas '{modality}' tidak tersedia. "
                f"Tersedia: {list(self._converters.keys())}. "
                f"Gunakan register_sensor() untuk menambahkan."
            )
        
        converter = self._converters[modality]
        return converter.to_frequency_tensor(data, **kwargs)
    
    def _auto_detect(self, data: Any) -> str:
        """
        Deteksi otomatis modalitas berdasarkan tipe dan bentuk data.
        
        Heuristik sederhana:
        - str → text
        - numpy/torch array 1D → audio
        - numpy/torch array 2D atau 3D → image (grayscale atau RGB)
        - Path/str dengan ekstensi → berdasarkan ekstensi file
        """
        if isinstance(data, str):
            # Cek apakah ini path ke file
            p = Path(data)
            if p.exists():
                suffix = p.suffix.lower()
                if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    return "image"
                elif suffix in [".mp3", ".wav", ".flac", ".ogg"]:
                    return "audio"
                elif suffix in [".mp4", ".avi", ".mov"]:
                    return "video"
            # Jika bukan path, anggap teks biasa
            return "text"
        
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            ndim = len(data.shape)
            if ndim == 1:
                return "audio"
            elif ndim == 2:
                return "image"  # Grayscale
            elif ndim == 3:
                return "image"  # RGB
            
        return "text"  # Default fallback
    
    def convert_batch(
        self,
        data_list: List[Any],
        modality: Optional[str] = None,
    ) -> List[FrequencyTensor]:
        """
        Konversi batch data sekaligus.
        
        Args:
            data_list: List data yang akan dikonversi
            modality : Modalitas (opsional, akan auto-detect per item)
            
        Returns:
            List FrequencyTensor
        """
        return [
            self.convert(item, modality=modality)
            for item in data_list
        ]
    
    # ==========================================================
    # AUTONOMOUS DATA SOURCING
    # ==========================================================
    
    def autonomous_multimodal_expansion(
        self,
        text_data: List[str],
        strategy: str = "simulate",
    ) -> List[Dict[str, FrequencyTensor]]:
        """
        [AUTONOMOUS DATA SOURCING] Ekspansi otomatis dari teks ke multimodal.
        
        Jika pengguna hanya punya data teks, fungsi ini secara otomatis
        menciptakan data gambar dan audio pendukung agar training bisa
        menjadi fully multimodal.
        
        Strategi yang tersedia:
        - "simulate" : Buat gambar/audio sintetis dari teks (offline, cepat)
        - "search"   : Cari gambar/audio terkait dari internet (butuh koneksi)
        - "generate" : Gunakan model generatif untuk buat data pendukung
        
        Args:
            text_data: List string teks yang akan diperluas
            strategy : Strategi ekspansi ("simulate", "search", "generate")
            
        Returns:
            List dict, tiap dict berisi {"text", "image", "audio"} sebagai FrequencyTensor
        """
        results = []
        
        for text in text_data:
            # Konversi teks ke frekuensi (selalu ada)
            text_freq = self.convert(text, modality="text")
            
            # Buat data gambar pendukung
            image_freq = self._create_supporting_image(text, strategy)
            
            # Buat data audio pendukung
            audio_freq = self._create_supporting_audio(text, strategy)
            
            results.append({
                "text" : text_freq,
                "image": image_freq,
                "audio": audio_freq,
            })
        
        return results
    
    def _create_supporting_image(
        self, 
        text: str, 
        strategy: str
    ) -> FrequencyTensor:
        """
        Buat data gambar pendukung untuk teks.
        
        Strategy "simulate": Buat gambar sintetis berbasis hash teks.
        Setiap teks menghasilkan pola gambar yang berbeda tapi deterministik.
        """
        if strategy == "simulate":
            # Buat gambar sintetis deterministik dari teks
            # Gunakan hash teks sebagai seed untuk numpy random
            seed = hash(text) % (2**32)
            rng  = np.random.RandomState(seed)
            
            # Buat array gambar sintetis 64x64 RGB
            # Pola yang berbeda untuk setiap teks, tapi reproducible
            fake_image = rng.rand(64, 64, 3).astype(np.float32)
            
            return self._converters["image"].to_frequency_tensor(fake_image)
        
        elif strategy == "search":
            # TODO: Implementasi pencarian gambar online
            # Contoh: Unsplash API, Wikimedia Commons API
            raise NotImplementedError(
                "Strategy 'search' belum diimplementasi. "
                "Gunakan strategy='simulate' untuk offline mode."
            )
        
        else:
            raise ValueError(f"Strategy '{strategy}' tidak dikenal.")
    
    def _create_supporting_audio(
        self, 
        text: str, 
        strategy: str
    ) -> FrequencyTensor:
        """
        Buat data audio pendukung untuk teks.
        
        Strategy "simulate": Buat sinyal audio sintetis (kombinasi sine waves).
        Setiap teks menghasilkan komposisi nada yang berbeda.
        """
        if strategy == "simulate":
            seed = hash(text + "_audio") % (2**32)
            rng  = np.random.RandomState(seed)
            
            # Buat sinyal audio sintetis: campuran beberapa sine wave
            sample_rate = 22050  # Hz standar
            duration    = 1.0    # 1 detik
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Pilih 3-5 frekuensi acak berdasarkan teks
            n_freqs    = rng.randint(3, 6)
            base_freqs = rng.uniform(100, 2000, size=n_freqs)  # Hz
            
            # Superposisi sine waves dengan amplitudo berbeda
            audio = np.zeros_like(t)
            for freq in base_freqs:
                amplitude = rng.uniform(0.1, 1.0)
                audio += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Normalisasi
            audio = (audio / (np.abs(audio).max() + 1e-8)).astype(np.float32)
            
            return self._converters["audio"].to_frequency_tensor(audio)
        
        else:
            raise NotImplementedError(f"Strategy '{strategy}' belum diimplementasi.")
