"""
resonait/pain/dissonance.py
============================
FREQUENCY-BASED PAIN SYSTEM — Sistem Rasa Sakit Digital

Modul ini mengimplementasikan konsep "rasa sakit" sebagai
Interferensi Destruktif dalam domain frekuensi.

Filosofi:
    Dalam fisika gelombang, dua gelombang dengan fase yang berlawanan
    saling membatalkan (Destructive Interference). ResonAIt mensimulasikan
    "rasa sakit" sebagai injeksi gelombang disonan yang mengganggu
    stabilitas representasi frekuensi internal AI.
    
    Semakin parah "kerusakan" (damage), semakin kuat frekuensi disonan
    yang disuntikkan, semakin terganggu kemampuan kognitif AI.
    
    Tujuan:
    - Mendorong AI untuk menghindari "kerusakan" dalam environment simulasi
    - Memberikan sinyal reward negatif yang "terasa" bagi AI
    - Mensimulasikan konsekuensi nyata dari tindakan buruk dalam game/simulasi
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from resonait.core.frequency_space import FrequencyTensor, Modality


# ============================================================
# ENUM: Jenis Kerusakan
# ============================================================

class DamageType(Enum):
    """
    Jenis kerusakan yang bisa diterima AI dari environment.
    Setiap jenis menghasilkan profil frekuensi disonan yang berbeda.
    """
    PHYSICAL   = "physical"    # Damage fisik (peluru, jatuh, dll) — frekuensi spike mendadak
    COGNITIVE  = "cognitive"   # Overload informasi — white noise broadband
    THERMAL    = "thermal"     # Panas/dingin ekstrem — frekuensi drift lambat
    ELECTRICAL = "electrical"  # Kejutan listrik — frekuensi tinggi mendadak
    SONIC      = "sonic"       # Suara keras — peak frekuensi audio
    CUSTOM     = "custom"      # Dari plugin komunitas


# ============================================================
# DATACLASS: DamageEvent
# ============================================================

@dataclass
class DamageEvent:
    """
    Representasi satu kejadian kerusakan dari environment.
    
    Attributes:
        damage_type  : Jenis kerusakan
        intensity    : Kekuatan kerusakan (0.0 = tidak ada, 1.0 = fatal)
        duration_ms  : Durasi kerusakan dalam milidetik
        source_position: Asal kerusakan (untuk game 3D, posisi dalam dunia)
        metadata     : Informasi tambahan dari environment
    """
    damage_type:      DamageType
    intensity:        float          # 0.0 sampai 1.0
    duration_ms:      float = 100.0  # milidetik
    source_position:  Optional[Tuple[float, float, float]] = None
    metadata:         Dict = None
    
    def __post_init__(self):
        self.intensity = max(0.0, min(1.0, self.intensity))  # Clamp ke [0, 1]
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# KELAS UTAMA: DissonanceEngine
# ============================================================

class DissonanceEngine(nn.Module):
    """
    Engine yang menghasilkan sinyal frekuensi disonan sebagai respons damage.
    
    Cara kerja:
    1. Terima DamageEvent dari environment (game, simulasi, dll)
    2. Generate sinyal frekuensi disonan sesuai jenis dan kekuatan damage
    3. Injeksikan ke FrequencyTensor otak AI via interferensi
    4. Otak AI "merasakan" gangguan ini sebagai degradasi performa
    
    Design Pattern:
        DissonanceEngine adalah "translator" antara event dunia fisik
        (peluru mengenai AI, AI jatuh dari tebing) dan representasi
        matematis dalam domain frekuensi yang bisa "dirasakan" oleh otak.
    
    Args:
        freq_dim      (int): Dimensi ruang frekuensi (harus sama dengan brain)
        recovery_rate (float): Kecepatan recovery per step (default: 0.01)
        max_pain      (float): Batas maksimum intensitas pain (default: 1.0)
    """
    
    def __init__(
        self,
        freq_dim:      int   = 512,
        recovery_rate: float = 0.01,
        max_pain:      float = 1.0,
    ):
        super().__init__()
        
        self.freq_dim      = freq_dim
        self.recovery_rate = recovery_rate
        self.max_pain      = max_pain
        
        # === STATE INTERNAL ===
        # Accumulated pain: jumlah total "sakit" yang sedang dirasakan
        self.register_buffer('accumulated_pain', torch.zeros(1))
        
        # Spectral disruption: distribusi gangguan di setiap frekuensi
        # Frekuensi yang sering terkena damage akan lebih "sensitif"
        self.register_buffer('spectral_disruption', torch.zeros(freq_dim))
        
        # History event untuk analisis
        self.event_history = []
        
        # === PROFIL FREKUENSI PER JENIS DAMAGE ===
        # Setiap jenis damage punya "tanda tangan frekuensi" yang unik
        # Ini menentukan frekuensi mana yang paling terganggu
        self._damage_freq_profiles = {
            DamageType.PHYSICAL:   self._physical_profile,
            DamageType.COGNITIVE:  self._cognitive_profile,
            DamageType.THERMAL:    self._thermal_profile,
            DamageType.ELECTRICAL: self._electrical_profile,
            DamageType.SONIC:      self._sonic_profile,
        }
    
    # ==========================================================
    # PROFIL FREKUENSI PER JENIS DAMAGE
    # ==========================================================
    
    def _physical_profile(self, intensity: float) -> torch.Tensor:
        """
        Profil frekuensi untuk damage fisik (peluru, impact, dll).
        
        Karakteristik: Spike mendadak di semua frekuensi (impuls)
        Analoginya seperti suara "dentuman" — energi tinggi di semua freq.
        """
        # Spike impuls: energi tersebar merata tapi tinggi
        profile = torch.rand(self.freq_dim) * intensity
        # Tambah spike di frekuensi rendah (impact terasa "berat")
        profile[:self.freq_dim // 8] *= 3.0
        return profile.clamp(0.0, 1.0)
    
    def _cognitive_profile(self, intensity: float) -> torch.Tensor:
        """
        Profil frekuensi untuk cognitive overload.
        
        Karakteristik: White noise — gangguan merata di semua frekuensi
        Seperti "kebisingan" yang menghalangi konsentrasi.
        """
        # White noise: gangguan acak di semua frekuensi
        profile = torch.randn(self.freq_dim).abs() * intensity * 0.5
        return profile.clamp(0.0, 1.0)
    
    def _thermal_profile(self, intensity: float) -> torch.Tensor:
        """
        Profil frekuensi untuk kerusakan termal.
        
        Karakteristik: Frekuensi rendah yang perlahan meningkat
        Seperti "panas" yang merembet — gangguan gradual.
        """
        # Dominan di frekuensi rendah, menurun seiring frekuensi naik
        freqs   = torch.arange(self.freq_dim, dtype=torch.float32)
        profile = intensity * torch.exp(-freqs / (self.freq_dim * 0.2))
        return profile
    
    def _electrical_profile(self, intensity: float) -> torch.Tensor:
        """
        Profil frekuensi untuk kejutan listrik.
        
        Karakteristik: Spike tajam di frekuensi tinggi
        Seperti "sengatan" — sangat tajam dan cepat.
        """
        profile = torch.zeros(self.freq_dim)
        # Konsentrasi di frekuensi tinggi (75-100% dari freq_dim)
        high_freq_start = int(self.freq_dim * 0.75)
        profile[high_freq_start:] = torch.rand(self.freq_dim - high_freq_start) * intensity
        return profile
    
    def _sonic_profile(self, intensity: float) -> torch.Tensor:
        """
        Profil frekuensi untuk damage sonik (suara keras).
        
        Karakteristik: Peak di frekuensi audio menengah
        Seperti suara ledakan — peak di frekuensi yang menyakitkan.
        """
        profile = torch.zeros(self.freq_dim)
        # Peak di tengah spektrum (frekuensi audio menengah)
        center    = self.freq_dim // 3
        bandwidth = self.freq_dim // 8
        
        freqs = torch.arange(self.freq_dim, dtype=torch.float32)
        gaussian_peak = intensity * torch.exp(
            -((freqs - center) ** 2) / (2 * bandwidth ** 2)
        )
        profile += gaussian_peak
        return profile.clamp(0.0, 1.0)
    
    # ==========================================================
    # API UTAMA: PROCESS DAMAGE
    # ==========================================================
    
    def process_damage(
        self,
        event: DamageEvent,
        current_brain_freq: Optional[FrequencyTensor] = None,
    ) -> Tuple[FrequencyTensor, Dict]:
        """
        Proses satu event kerusakan dan hasilkan sinyal dissonance.
        
        Ini adalah fungsi utama yang dipanggil oleh environment hook
        ketika AI menerima damage dalam simulasi.
        
        Alur:
        1. Generate profil frekuensi disonan sesuai jenis damage
        2. Scale sesuai intensitas
        3. Jika ada current brain freq, buat interferensi destruktif
        4. Update accumulated pain state
        5. Return sinyal dissonance + metadata
        
        Args:
            event              : DamageEvent dari environment
            current_brain_freq : FrequencyTensor otak saat ini (opsional)
            
        Returns:
            Tuple:
            - dissonance_signal: FrequencyTensor sinyal disonan
            - damage_report    : Dict berisi metrik kerusakan
        """
        # === Step 1: Generate profil frekuensi disonan ===
        profile_fn = self._damage_freq_profiles.get(
            event.damage_type,
            self._physical_profile  # Fallback ke physical
        )
        
        dissonance_amplitude = profile_fn(event.intensity).to(self.accumulated_pain.device)
        
        # Phase yang berlawanan (π radian) = interferensi destruktif maksimum
        # Jika otak bekerja di fase φ, sinyal pain di fase (φ + π) akan membatalkannya
        if current_brain_freq is not None:
            # Gunakan fase otak dan balik 180° untuk interferensi destruktif
            brain_phase       = current_brain_freq.phase[:, 0, :]  # (batch, freq_dim)
            dissonance_phase  = brain_phase[0] + np.pi              # Balik fase
        else:
            # Tanpa referensi fase, gunakan phase acak
            dissonance_phase = torch.rand(self.freq_dim) * 2 * np.pi
        
        dissonance_phase = dissonance_phase.to(self.accumulated_pain.device)
        
        # Bungkus dalam FrequencyTensor
        dissonance_signal = FrequencyTensor(
            amplitude=dissonance_amplitude.unsqueeze(0).unsqueeze(0),
            phase=dissonance_phase.unsqueeze(0).unsqueeze(0),
            modality=Modality.CUSTOM,
            metadata={
                "damage_type":  event.damage_type.value,
                "intensity":    event.intensity,
                "duration_ms":  event.duration_ms,
                "is_pain":      True,
            }
        )
        
        # === Step 2: Update accumulated pain ===
        # Pain meningkat proporsional dengan intensitas damage
        pain_increment = torch.tensor(event.intensity * 0.3)
        self.accumulated_pain = (
            self.accumulated_pain + pain_increment
        ).clamp(0.0, self.max_pain)
        
        # Update spectral disruption
        self.spectral_disruption = (
            self.spectral_disruption + dissonance_amplitude * 0.1
        ).clamp(0.0, 1.0)
        
        # === Step 3: Catat ke history ===
        self.event_history.append({
            "damage_type":    event.damage_type.value,
            "intensity":      event.intensity,
            "accumulated_pain": self.accumulated_pain.item(),
        })
        
        # === Step 4: Buat damage report ===
        damage_report = {
            "damage_type":       event.damage_type.value,
            "intensity":         event.intensity,
            "accumulated_pain":  self.accumulated_pain.item(),
            "spectral_disruption_avg": self.spectral_disruption.mean().item(),
            "most_affected_freq": self.spectral_disruption.argmax().item(),
            "is_critical":       self.accumulated_pain.item() > 0.8,
        }
        
        return dissonance_signal, damage_report
    
    def apply_to_brain(
        self,
        brain,  # ResonAItBrain instance
        event: DamageEvent,
    ) -> Dict:
        """
        Terapkan damage langsung ke otak AI.
        
        Shortcut yang menggabungkan process_damage() + brain.receive_pain()
        dalam satu panggilan.
        
        Args:
            brain : Instance ResonAItBrain
            event : DamageEvent yang akan diterapkan
            
        Returns:
            damage_report dict
        """
        # Generate sinyal dissonance
        dissonance_signal, report = self.process_damage(event)
        
        # Kirim ke otak
        brain.receive_pain(
            dissonance_signal=dissonance_signal.amplitude[0, 0],
            intensity=event.intensity,
        )
        
        return report
    
    def step_recovery(self, brain=None):
        """
        Panggil setiap step simulasi untuk recovery gradual.
        
        Pain memudar secara alami seiring waktu jika tidak ada damage baru.
        Ini mendorong AI untuk "menghindar" dari damage karena efeknya
        tidak permanen — AI bisa pulih jika bermain dengan baik.
        
        Args:
            brain: ResonAItBrain (opsional) — jika ada, panggil brain.recover()
        """
        # Kurangi pain secara gradual
        self.accumulated_pain = (
            self.accumulated_pain - self.recovery_rate
        ).clamp(0.0, self.max_pain)
        
        # Kurangi spectral disruption
        self.spectral_disruption = (
            self.spectral_disruption * (1.0 - self.recovery_rate)
        ).clamp(0.0, 1.0)
        
        # Sync dengan brain jika ada
        if brain is not None:
            brain.recover(amount=self.recovery_rate)
    
    def get_pain_report(self) -> Dict:
        """
        Laporan lengkap status pain saat ini.
        
        Returns:
            Dict berisi semua metrik pain
        """
        return {
            "accumulated_pain":       self.accumulated_pain.item(),
            "spectral_disruption_avg": self.spectral_disruption.mean().item(),
            "spectral_disruption_max": self.spectral_disruption.max().item(),
            "most_affected_freq":      self.spectral_disruption.argmax().item(),
            "total_events":            len(self.event_history),
            "is_in_pain":              self.accumulated_pain.item() > 0.1,
            "is_critical":             self.accumulated_pain.item() > 0.8,
        }
    
    def reset(self):
        """Reset semua pain state (e.g., mulai episode baru dalam game)."""
        self.accumulated_pain.zero_()
        self.spectral_disruption.zero_()
        self.event_history.clear()
        print("[DissonanceEngine] ✓ Pain state direset (episode baru).")
