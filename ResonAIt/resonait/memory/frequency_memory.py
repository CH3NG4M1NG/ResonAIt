"""
resonait/memory/frequency_memory.py
=====================================
SISTEM MEMORI BERBASIS FREKUENSI

Modul ini mengimplementasikan dua jenis memori untuk ResonAIt:

1. ShortTermMemory (STM):
   - Buffer memori jangka pendek (seperti RAM)
   - Menyimpan FrequencyTensor dari beberapa step terakhir
   - Menghilang setelah kapasitas terlampaui (FIFO)
   - Analogi: "Apa yang barusan terjadi?"

2. LongTermMemory (LTM):
   - Penyimpanan memori jangka panjang (seperti HDD)
   - Mengkompresi dan menyimpan pola frekuensi penting
   - Retrieval via similarity search di ruang frekuensi
   - Analogi: "Pernah mengalami hal seperti ini sebelumnya?"

Filosofi Frekuensi:
    Memori bukan disimpan sebagai "fakta" tapi sebagai "pola getaran".
    Saat melihat kucing, otak menyimpan pola frekuensi BUKAN kata "kucing".
    Retrieval terjadi saat pola baru beresonansi dengan pola lama.
    
    Ini menjelaskan mengapa kita lebih mudah mengingat sesuatu yang
    "beresonansi" dengan pengalaman sebelumnya.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass, field
import time

from resonait.core.frequency_space import FrequencyTensor, Modality


# ============================================================
# DATACLASS: MemoryEntry
# ============================================================

@dataclass
class MemoryEntry:
    """
    Satu unit memori yang tersimpan dalam ResonAIt.
    
    Attributes:
        freq_tensor  : FrequencyTensor yang merepresentasikan memori ini
        importance   : Seberapa penting memori ini (0.0-1.0)
        timestamp    : Kapan memori ini dibuat
        access_count : Berapa kali memori ini diakses (semakin sering = semakin kuat)
        context_tags : Tag konteks untuk mempermudah retrieval
        modality     : Modalitas asal memori
    """
    freq_tensor:  FrequencyTensor
    importance:   float = 0.5
    timestamp:    float = field(default_factory=time.time)
    access_count: int   = 0
    context_tags: List[str] = field(default_factory=list)
    
    @property
    def modality(self) -> Modality:
        return self.freq_tensor.modality
    
    def strengthen(self, amount: float = 0.1):
        """
        Kuatkan memori ini (dipanggil saat diakses/direview).
        
        Semakin sering memori diakses, semakin penting dan kuat ia menjadi.
        Ini mensimulasikan proses konsolidasi memori pada otak biologis.
        """
        self.access_count += 1
        self.importance    = min(1.0, self.importance + amount)
    
    def decay(self, rate: float = 0.01):
        """
        Perlahan-lahan melemahkan memori yang tidak diakses.
        Ini mensimulasikan natural forgetting — ingatan memudar jika tidak diulang.
        """
        self.importance = max(0.0, self.importance - rate)


# ============================================================
# SHORT TERM MEMORY
# ============================================================

class ShortTermMemory:
    """
    Memori jangka pendek yang menyimpan FrequencyTensor terbaru.
    
    Bekerja seperti "working memory" manusia — buffer sementara
    yang menyimpan informasi yang sedang aktif diproses.
    
    Implementasi: Fixed-size FIFO queue (deque)
    
    Args:
        capacity  (int)  : Jumlah maksimum FrequencyTensor yang tersimpan
        freq_dim  (int)  : Dimensi frekuensi
        decay_rate(float): Kecepatan decay memori lama (per step)
    """
    
    def __init__(
        self,
        capacity:   int   = 32,
        freq_dim:   int   = 512,
        decay_rate: float = 0.05,
    ):
        self.capacity   = capacity
        self.freq_dim   = freq_dim
        self.decay_rate = decay_rate
        
        # Buffer FIFO — item terlama otomatis hilang saat penuh
        self._buffer: deque = deque(maxlen=capacity)
    
    def store(
        self,
        freq_tensor: FrequencyTensor,
        importance:  float = 0.5,
        tags:        Optional[List[str]] = None,
    ):
        """
        Simpan FrequencyTensor baru ke memori jangka pendek.
        
        Jika buffer sudah penuh, entri terlama otomatis dihapus (FIFO).
        
        Args:
            freq_tensor: FrequencyTensor yang akan disimpan
            importance : Tingkat kepentingan (memori penting lebih lama bertahan)
            tags       : Tag konteks opsional
        """
        entry = MemoryEntry(
            freq_tensor=freq_tensor,
            importance=importance,
            context_tags=tags or [],
        )
        self._buffer.append(entry)
    
    def recall_recent(self, n: int = 5) -> List[MemoryEntry]:
        """
        Ambil n memori terbaru.
        
        Args:
            n: Jumlah memori yang ingin diambil
            
        Returns:
            List MemoryEntry, dari terbaru ke terlama
        """
        recent = list(self._buffer)[-n:]
        recent.reverse()  # Terbaru duluan
        
        # Kuatkan memori yang diakses
        for entry in recent:
            entry.strengthen(amount=0.05)
        
        return recent
    
    def recall_similar(
        self,
        query_freq: FrequencyTensor,
        top_k: int = 3,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Cari memori yang paling mirip dengan query.
        
        Similarity diukur via koherensi frekuensi antara query dan memori tersimpan.
        Ini adalah "associative recall" — mencari memori yang beresonansi.
        
        Args:
            query_freq: FrequencyTensor sebagai query pencarian
            top_k     : Jumlah memori paling mirip yang dikembalikan
            
        Returns:
            List of (MemoryEntry, similarity_score), diurutkan dari paling mirip
        """
        if not self._buffer:
            return []
        
        scored_memories = []
        for entry in self._buffer:
            # Hitung koherensi antara query dan memori tersimpan
            try:
                # Pastikan shape kompatibel sebelum membandingkan
                if entry.freq_tensor.amplitude.shape[-1] == query_freq.amplitude.shape[-1]:
                    similarity = query_freq.coherence_with(entry.freq_tensor)
                    # Bobot dengan importance — memori penting lebih mudah di-recall
                    weighted_sim = similarity * (0.5 + 0.5 * entry.importance)
                    scored_memories.append((entry, weighted_sim))
            except Exception:
                continue
        
        # Urutkan dari similarity tertinggi
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Kuatkan memori yang ditemukan
        results = scored_memories[:top_k]
        for entry, score in results:
            entry.strengthen(amount=0.05)
        
        return results
    
    def aggregate_to_context(self) -> Optional[torch.Tensor]:
        """
        Agregasi semua memori di buffer menjadi satu vektor konteks.
        
        Berguna untuk memberikan "konteks historis" ke otak sebelum memproses
        input baru. Ini seperti "ingat semua yang sudah terjadi" sebelum memutuskan.
        
        Returns:
            Tensor konteks teragregasi. Shape: (1, freq_dim)
            None jika buffer kosong.
        """
        if not self._buffer:
            return None
        
        amplitudes = []
        weights    = []
        
        for entry in self._buffer:
            amp    = entry.freq_tensor.amplitude.mean(dim=(0, 1))  # (freq_dim,)
            weight = entry.importance
            amplitudes.append(amp)
            weights.append(weight)
        
        # Weighted average — memori lebih penting berkontribusi lebih besar
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / (weights_tensor.sum() + 1e-8)
        
        stacked = torch.stack(amplitudes, dim=0)  # (n, freq_dim)
        context = (stacked * weights_tensor.unsqueeze(1)).sum(dim=0, keepdim=True)
        # Shape: (1, freq_dim)
        
        return context
    
    def step_decay(self):
        """
        Aplikasikan decay ke semua memori di buffer.
        Dipanggil setiap step untuk mensimulasikan natural forgetting.
        """
        for entry in self._buffer:
            entry.decay(rate=self.decay_rate)
    
    @property
    def size(self) -> int:
        return len(self._buffer)
    
    def clear(self):
        """Kosongkan seluruh memori jangka pendek."""
        self._buffer.clear()


# ============================================================
# LONG TERM MEMORY
# ============================================================

class LongTermMemory(nn.Module):
    """
    Memori jangka panjang yang mengkompresi dan menyimpan pola frekuensi penting.
    
    Berbeda dari STM yang menyimpan raw FrequencyTensor,
    LTM mengkompresi representasi menggunakan learned autoencoder kecil.
    
    Ini memungkinkan penyimpanan jutaan "ingatan" yang efisien
    karena setiap ingatan dikompres ke vektor kecil.
    
    Retrieval: Approximate Nearest Neighbor via cosine similarity
    
    Args:
        freq_dim      (int): Dimensi frekuensi input
        memory_dim    (int): Dimensi representasi terkompresi
        max_memories  (int): Batas maksimum ingatan tersimpan
        importance_threshold (float): Minimum importance untuk masuk LTM
    """
    
    def __init__(
        self,
        freq_dim:              int   = 512,
        memory_dim:            int   = 64,
        max_memories:          int   = 10000,
        importance_threshold:  float = 0.6,
    ):
        super().__init__()
        
        self.freq_dim             = freq_dim
        self.memory_dim           = memory_dim
        self.max_memories         = max_memories
        self.importance_threshold = importance_threshold
        
        # === ENCODER: FrequencyTensor → Compressed Memory Vector ===
        # Kompres frekuensi besar ke vektor kecil yang efisien
        self.encoder = nn.Sequential(
            nn.Linear(freq_dim, freq_dim // 2),
            nn.GELU(),
            nn.Linear(freq_dim // 2, memory_dim),
            nn.LayerNorm(memory_dim),
        )
        
        # === DECODER: Compressed Vector → FrequencyTensor (untuk rekonstruksi) ===
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, freq_dim // 2),
            nn.GELU(),
            nn.Linear(freq_dim // 2, freq_dim),
        )
        
        # === MEMORY BANK ===
        # Menyimpan semua memori terkompresi sebagai matrix
        # Shape: (max_memories, memory_dim)
        self.register_buffer(
            'memory_bank',
            torch.zeros(max_memories, memory_dim)
        )
        self.register_buffer(
            'memory_importance',
            torch.zeros(max_memories)
        )
        
        # Pointer ke slot memori berikutnya
        self.register_buffer('write_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))
        
        # Metadata untuk setiap slot (tidak bisa di-register sebagai buffer karena non-tensor)
        self._metadata: List[Dict] = [{}] * max_memories
    
    def consolidate(
        self,
        stm: ShortTermMemory,
        min_importance: Optional[float] = None,
    ) -> int:
        """
        Konsolidasi memori dari Short-Term ke Long-Term Memory.
        
        Ini adalah proses yang terjadi selama "tidur" pada otak biologis —
        memori penting dari STM dipindahkan ke penyimpanan jangka panjang.
        
        Hanya memori dengan importance di atas threshold yang dikonsolidasi.
        
        Args:
            stm           : ShortTermMemory yang akan dikonsolidasi
            min_importance: Override threshold minimum
            
        Returns:
            Jumlah memori yang berhasil dikonsolidasi
        """
        threshold  = min_importance or self.importance_threshold
        consolidated = 0
        
        for entry in stm._buffer:
            if entry.importance >= threshold:
                self.store(
                    freq_tensor=entry.freq_tensor,
                    importance=entry.importance,
                    metadata={
                        "tags":        entry.context_tags,
                        "access_count": entry.access_count,
                        "modality":    entry.modality.value,
                    }
                )
                consolidated += 1
        
        return consolidated
    
    def store(
        self,
        freq_tensor: FrequencyTensor,
        importance:  float = 0.5,
        metadata:    Optional[Dict] = None,
    ):
        """
        Simpan satu memori ke Long-Term Memory.
        
        Proses:
        1. Encode FrequencyTensor ke compressed vector
        2. Simpan di memory_bank dengan LRU replacement
        
        Args:
            freq_tensor: FrequencyTensor yang akan disimpan
            importance : Tingkat kepentingan memori ini
            metadata   : Metadata tambahan
        """
        # Encode frekuensi ke compressed vector
        with torch.no_grad():
            amplitude_flat = freq_tensor.amplitude.mean(dim=(0, 1))  # (freq_dim,)
            compressed     = self.encoder(amplitude_flat)              # (memory_dim,)
        
        # Dapatkan slot untuk menulis
        ptr = self.write_ptr.item()
        
        # Tulis ke memory bank
        self.memory_bank[ptr]       = compressed.detach()
        self.memory_importance[ptr] = importance
        self._metadata[ptr]          = metadata or {}
        
        # Advance write pointer (circular buffer)
        self.write_ptr = torch.tensor(
            [(ptr + 1) % self.max_memories], dtype=torch.long
        )
        self.memory_count = torch.clamp(
            self.memory_count + 1, max=self.max_memories
        )
    
    def recall(
        self,
        query_freq: FrequencyTensor,
        top_k:      int   = 5,
        threshold:  float = 0.3,
    ) -> List[Tuple[torch.Tensor, float, Dict]]:
        """
        Cari memori terkait dari Long-Term Memory via similarity search.
        
        Menggunakan cosine similarity antara query yang di-encode
        dan semua memori tersimpan di memory_bank.
        
        Args:
            query_freq: FrequencyTensor sebagai query
            top_k     : Jumlah memori yang dikembalikan
            threshold : Minimum similarity untuk dianggap relevan
            
        Returns:
            List of (compressed_memory, similarity_score, metadata)
        """
        n_stored = min(self.memory_count.item(), self.max_memories)
        if n_stored == 0:
            return []
        
        # Encode query
        with torch.no_grad():
            query_amp        = query_freq.amplitude.mean(dim=(0, 1))
            query_compressed = self.encoder(query_amp)  # (memory_dim,)
        
        # Hitung cosine similarity dengan semua memori tersimpan
        stored_memories = self.memory_bank[:n_stored]  # (n_stored, memory_dim)
        
        # Normalize untuk cosine similarity
        query_norm   = F.normalize(query_compressed.unsqueeze(0), dim=-1)  # (1, memory_dim)
        memory_norms = F.normalize(stored_memories, dim=-1)                 # (n, memory_dim)
        
        similarities = torch.matmul(query_norm, memory_norms.T).squeeze(0)  # (n_stored,)
        
        # Bobot dengan importance
        weighted_sims = similarities * (0.3 + 0.7 * self.memory_importance[:n_stored])
        
        # Filter by threshold dan ambil top_k
        mask      = weighted_sims >= threshold
        if not mask.any():
            return []
        
        masked_sims    = weighted_sims * mask.float()
        top_vals, top_indices = torch.topk(
            masked_sims,
            k=min(top_k, mask.sum().item()),
        )
        
        results = []
        for idx, sim in zip(top_indices.tolist(), top_vals.tolist()):
            if sim > 0:
                results.append((
                    self.memory_bank[idx],  # Compressed memory vector
                    sim,                    # Similarity score
                    self._metadata[idx],    # Metadata
                ))
        
        return results
    
    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Rekonstruksi amplitude FrequencyTensor dari compressed vector.
        
        Berguna untuk "mengingat" kembali detail dari memori tersimpan.
        
        Args:
            compressed: Compressed memory vector. Shape: (memory_dim,)
            
        Returns:
            Reconstructed amplitude tensor. Shape: (freq_dim,)
        """
        with torch.no_grad():
            return self.decoder(compressed)
    
    @property
    def utilization(self) -> float:
        """Persentase kapasitas LTM yang terisi."""
        return min(self.memory_count.item() / self.max_memories, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistik penggunaan Long-Term Memory."""
        n_stored = min(self.memory_count.item(), self.max_memories)
        return {
            "stored_memories":  n_stored,
            "max_capacity":     self.max_memories,
            "utilization":      f"{self.utilization * 100:.1f}%",
            "avg_importance":   self.memory_importance[:n_stored].mean().item() if n_stored > 0 else 0.0,
            "memory_dim":       self.memory_dim,
        }


# ============================================================
# MEMORY SYSTEM — Gabungan STM + LTM
# ============================================================

class FrequencyMemorySystem:
    """
    Sistem memori lengkap yang menggabungkan STM dan LTM.
    
    Interface tunggal untuk semua operasi memori ResonAIt.
    
    Penggunaan:
        >>> memory = FrequencyMemorySystem(freq_dim=512)
        >>> memory.perceive(freq_tensor)           # Simpan ke STM
        >>> memory.consolidate()                    # Pindahkan penting ke LTM
        >>> context = memory.get_context(query)     # Ambil konteks relevan
    """
    
    def __init__(
        self,
        freq_dim:     int = 512,
        stm_capacity: int = 32,
        ltm_capacity: int = 10000,
        memory_dim:   int = 64,
    ):
        self.stm = ShortTermMemory(capacity=stm_capacity, freq_dim=freq_dim)
        self.ltm = LongTermMemory(
            freq_dim=freq_dim,
            memory_dim=memory_dim,
            max_memories=ltm_capacity,
        )
        self.freq_dim = freq_dim
        
        # Konsolidasi otomatis setiap N steps
        self._step_counter = 0
        self._consolidate_every = 50
    
    def perceive(
        self,
        freq_tensor: FrequencyTensor,
        importance:  float = 0.5,
        tags:        Optional[List[str]] = None,
    ):
        """
        Simpan persepsi baru ke STM.
        Secara otomatis trigger konsolidasi jika sudah waktunya.
        """
        self.stm.store(freq_tensor, importance=importance, tags=tags)
        self._step_counter += 1
        
        # Auto-konsolidasi periodik
        if self._step_counter % self._consolidate_every == 0:
            n = self.consolidate(min_importance=0.7)
            if n > 0:
                print(f"[MemorySystem] 💾 Auto-konsolidasi: {n} memori dipindahkan ke LTM.")
    
    def consolidate(self, min_importance: float = 0.6) -> int:
        """Transfer memori penting dari STM ke LTM."""
        return self.ltm.consolidate(self.stm, min_importance=min_importance)
    
    def get_context(
        self,
        query_freq: FrequencyTensor,
        use_stm: bool = True,
        use_ltm: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Ambil konteks memori yang relevan untuk query.
        
        Menggabungkan hasil dari STM (recent) dan LTM (long-term)
        menjadi satu vektor konteks untuk diumpankan ke otak.
        
        Args:
            query_freq: FrequencyTensor sebagai query
            use_stm   : Gunakan Short-Term Memory
            use_ltm   : Gunakan Long-Term Memory
            
        Returns:
            Konteks teragregasi. Shape: (1, freq_dim) atau None
        """
        contexts = []
        
        if use_stm:
            stm_context = self.stm.aggregate_to_context()
            if stm_context is not None:
                contexts.append(stm_context)
        
        if use_ltm:
            ltm_results = self.ltm.recall(query_freq, top_k=3)
            if ltm_results:
                # Rekonstruksi memori LTM dan rata-ratakan
                reconstructed = [
                    self.ltm.reconstruct(mem).unsqueeze(0)
                    for mem, _, _ in ltm_results
                ]
                ltm_context = torch.stack(reconstructed).mean(dim=0)
                contexts.append(ltm_context)
        
        if not contexts:
            return None
        
        # Rata-ratakan semua konteks
        return torch.stack(contexts).mean(dim=0)
    
    def get_stats(self) -> Dict:
        """Statistik lengkap sistem memori."""
        return {
            "stm": {
                "size":     self.stm.size,
                "capacity": self.stm.capacity,
            },
            "ltm": self.ltm.get_stats(),
            "total_steps": self._step_counter,
        }
