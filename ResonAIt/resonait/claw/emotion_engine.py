"""
resonait/claw/emotion_engine.py
================================
EMOTION ENGINE — Emosi berbasis Frekuensi

Konsep inti:
    Emosi bukan "label" (senang/sedih) tapi POLA FREKUENSI.
    - Curiosity    = amplitudo tinggi di range frekuensi tengah, naik cepat
    - Joy          = distribusi harmonis, rata, coherence tinggi
    - Discomfort   = interferensi destruktif, banyak disonansi
    - Boredom      = amplitudo rendah flat, tidak ada variasi
    - Excitement   = spike mendadak di banyak frekuensi sekaligus

    Ini bukan roleplay — ini adalah state internal yang BENAR-BENAR
    mempengaruhi cara otak memproses input berikutnya.
    Emosi = prior yang di-inject ke FNO Brain sebelum inferensi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class EmotionType(Enum):
    NEUTRAL    = "neutral"
    CURIOSITY  = "curiosity"
    JOY        = "joy"
    DISCOMFORT = "discomfort"
    BOREDOM    = "boredom"
    EXCITEMENT = "excitement"
    FOCUSED    = "focused"


@dataclass
class EmotionalState:
    """
    State emosi saat ini — bukan satu nilai tapi distribusi.
    
    Setiap emosi punya intensitas 0.0–1.0.
    Emosi bisa campur (60% focused + 30% curiosity + 10% excitement).
    """
    emotions: Dict[EmotionType, float] = field(
        default_factory=lambda: {e: 0.0 for e in EmotionType}
    )
    
    def __post_init__(self):
        # Neutral selalu ada sebagai baseline
        self.emotions[EmotionType.NEUTRAL] = 0.3
    
    def dominant(self) -> EmotionType:
        return max(self.emotions, key=self.emotions.get)
    
    def intensity(self, emotion: EmotionType) -> float:
        return self.emotions.get(emotion, 0.0)
    
    def total_arousal(self) -> float:
        """Seberapa 'terangsang' secara keseluruhan (0=tidur, 1=hiperaktif)."""
        weights = {
            EmotionType.CURIOSITY:  0.8,
            EmotionType.EXCITEMENT: 1.0,
            EmotionType.JOY:        0.6,
            EmotionType.FOCUSED:    0.7,
            EmotionType.DISCOMFORT: 0.9,
            EmotionType.BOREDOM:    0.1,
            EmotionType.NEUTRAL:    0.3,
        }
        return sum(self.emotions[e] * w for e, w in weights.items())
    
    def as_text(self) -> str:
        active = [(e, v) for e, v in self.emotions.items() if v > 0.1]
        active.sort(key=lambda x: x[1], reverse=True)
        if not active:
            return "neutral"
        return " + ".join(f"{e.value}({v:.2f})" for e, v in active[:3])
    
    def __repr__(self):
        return f"EmotionalState({self.as_text()}, arousal={self.total_arousal():.2f})"


class EmotionEngine(nn.Module):
    """
    Engine yang menghasilkan dan mempertahankan state emosi
    berdasarkan frekuensi input dan konteks.
    
    Cara kerja:
        1. Setiap input FrequencyTensor di-analisis pola frekuensinya
        2. Pola frekuensi di-mapping ke dimensi emosi
        3. State emosi di-update secara smooth (tidak langsung loncat)
        4. State emosi menghasilkan "emotional prior" — 
           FrequencyTensor yang di-inject ke FNO Brain
        5. Otak memproses (input + emotional_prior) — emosi mempengaruhi respons
    
    Args:
        freq_dim     : Dimensi frequency space
        decay_rate   : Kecepatan emosi memudar (0.01 = lambat, 0.1 = cepat)
        curiosity_threshold: Seberapa besar novelty yang trigger curiosity
    """
    
    def __init__(
        self,
        freq_dim:              int   = 512,
        decay_rate:            float = 0.02,
        curiosity_threshold:   float = 0.3,
        boredom_threshold:     int   = 20,   # steps tanpa novelty
    ):
        super().__init__()
        
        self.freq_dim             = freq_dim
        self.decay_rate           = decay_rate
        self.curiosity_threshold  = curiosity_threshold
        self.boredom_threshold    = boredom_threshold
        
        # === EMOTION FREQUENCY PROFILES ===
        # Setiap emosi punya "tanda tangan frekuensi" unik
        # yang akan di-inject sebagai prior ke FNO Brain
        self._init_emotion_profiles()
        
        # === STATE ===
        self.current_state  = EmotionalState()
        self.state_history  = []
        self.steps_no_novel = 0
        self.last_freq_mean = None
        
        # Running stats untuk novelty detection
        self.register_buffer('freq_running_mean', torch.zeros(freq_dim))
        self.register_buffer('freq_running_var',  torch.ones(freq_dim))
        self.register_buffer('n_observations',    torch.tensor(0))
        
        # Learnable emotional modulation weights
        # Seberapa kuat tiap emosi mempengaruhi output
        self.emotion_weights = nn.ParameterDict({
            e.value: nn.Parameter(torch.tensor(0.1))
            for e in EmotionType
        })
    
    def _init_emotion_profiles(self):
        """
        Inisialisasi profil frekuensi untuk setiap emosi.
        
        Profil ini adalah FrequencyTensor "template" yang merepresentasikan
        pola frekuensi khas dari setiap emosi.
        """
        freq_dim = self.freq_dim
        profiles = {}
        
        # CURIOSITY: amplitudo naik di frekuensi tengah (mencari informasi)
        curiosity = torch.zeros(freq_dim)
        mid = freq_dim // 3
        curiosity[mid:2*mid] = torch.linspace(0.3, 1.0, mid)
        profiles['curiosity'] = curiosity
        
        # JOY: harmonis, banyak puncak kecil yang teratur
        joy = torch.zeros(freq_dim)
        for i in range(0, freq_dim, freq_dim // 8):
            joy[i] = 0.8
            if i + 1 < freq_dim: joy[i+1] = 0.4
        profiles['joy'] = joy
        
        # DISCOMFORT: noise + interferensi destruktif
        torch.manual_seed(42)
        discomfort = torch.rand(freq_dim) * 0.5
        discomfort[freq_dim//4:freq_dim//2] = 0.9  # spike di range tertentu
        profiles['discomfort'] = discomfort
        
        # BOREDOM: flat, rendah
        boredom = torch.ones(freq_dim) * 0.1
        profiles['boredom'] = boredom
        
        # EXCITEMENT: spike mendadak di banyak tempat
        excitement = torch.zeros(freq_dim)
        for i in range(0, freq_dim, freq_dim // 16):
            excitement[i] = torch.rand(1).item()
        profiles['excitement'] = excitement
        
        # FOCUSED: amplitudo tinggi tapi sempit (satu range frekuensi)
        focused = torch.zeros(freq_dim)
        focused[freq_dim//4:freq_dim//3] = 0.9
        profiles['focused'] = focused
        
        # NEUTRAL: flat medium
        neutral = torch.ones(freq_dim) * 0.3
        profiles['neutral'] = neutral
        
        # Simpan sebagai non-trainable buffers
        for name, profile in profiles.items():
            self.register_buffer(f'profile_{name}', profile.unsqueeze(0))
    
    def _compute_novelty(self, freq_amplitude: torch.Tensor) -> float:
        """
        Hitung seberapa "baru" input ini dibanding yang sudah pernah dilihat.
        
        Novelty tinggi → curiosity naik
        Novelty rendah berulang → boredom naik
        """
        amp_mean = freq_amplitude.float().mean(dim=-1).mean()
        
        # Update running stats
        n = self.n_observations.item()
        if n == 0:
            self.freq_running_mean = freq_amplitude.float().mean(dim=0).mean(dim=0)
            novelty = 0.5
        else:
            # Jarak dari running mean
            diff = (freq_amplitude.float().mean(dim=0).mean(dim=0)
                    - self.freq_running_mean).abs().mean()
            expected_var = (self.freq_running_var.sqrt() + 1e-8).mean()
            novelty = (diff / expected_var).clamp(0, 2).item() / 2
            
            # Update running stats
            alpha = 0.05
            self.freq_running_mean = (
                (1 - alpha) * self.freq_running_mean +
                alpha * freq_amplitude.float().mean(dim=0).mean(dim=0).detach()
            )
        
        self.n_observations += 1
        return float(novelty)
    
    def _update_state(self, novelty: float, input_power: float):
        """
        Update state emosi berdasarkan novelty dan power input.
        
        Rules:
        - Novelty tinggi → curiosity naik, boredom turun
        - Novelty rendah berulang → boredom naik
        - Power tinggi → excitement/discomfort tergantung konteks
        - Semua emosi decay secara natural
        """
        state = self.current_state.emotions
        
        # Natural decay semua emosi
        for emotion in EmotionType:
            state[emotion] = max(0.0, state[emotion] - self.decay_rate)
        
        # Curiosity: naik jika ada novelty
        if novelty > self.curiosity_threshold:
            state[EmotionType.CURIOSITY] = min(1.0,
                state[EmotionType.CURIOSITY] + novelty * 0.4
            )
            self.steps_no_novel = 0
        else:
            self.steps_no_novel += 1
        
        # Boredom: naik jika terlalu lama tidak ada novelty
        if self.steps_no_novel > self.boredom_threshold:
            boredom_intensity = min(0.8,
                (self.steps_no_novel - self.boredom_threshold) * 0.02
            )
            state[EmotionType.BOREDOM] = min(1.0,
                state[EmotionType.BOREDOM] + boredom_intensity
            )
        
        # Excitement: jika input power tinggi dan ada novelty
        if input_power > 0.7 and novelty > 0.4:
            state[EmotionType.EXCITEMENT] = min(0.8,
                state[EmotionType.EXCITEMENT] + 0.2
            )
        
        # Focused: jika power tinggi tapi novelty rendah (sedang mengerjakan sesuatu)
        if input_power > 0.5 and novelty < 0.2:
            state[EmotionType.FOCUSED] = min(0.9,
                state[EmotionType.FOCUSED] + 0.15
            )
        
        # Joy: naik secara gradual jika interaksi berlangsung smooth
        if 0.2 < novelty < 0.5 and input_power > 0.3:
            state[EmotionType.JOY] = min(0.6,
                state[EmotionType.JOY] + 0.05
            )
        
        # Normalize: total tidak boleh > 2.0 (bisa campur tapi tidak overflow)
        total = sum(state.values())
        if total > 2.0:
            for e in state:
                state[e] /= (total / 2.0)
        
        # Log state
        self.state_history.append({
            'time':    time.time(),
            'state':   self.current_state.as_text(),
            'arousal': self.current_state.total_arousal(),
        })
        # Jaga history tidak terlalu panjang
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]
    
    def get_emotional_prior(self, device: torch.device) -> torch.Tensor:
        """
        Hasilkan "emotional prior" — FrequencyTensor yang merepresentasikan
        state emosi saat ini.
        
        Prior ini akan di-inject ke FNO Brain sebelum memproses input utama.
        Sehingga emosi benar-benar mempengaruhi cara otak berpikir.
        
        Returns:
            Tensor (1, freq_dim) yang mencerminkan campuran emosi saat ini
        """
        prior = torch.zeros(1, self.freq_dim, device=device)
        
        state = self.current_state.emotions
        for emotion_type, intensity in state.items():
            if intensity < 0.01:
                continue
            profile = getattr(self, f'profile_{emotion_type.value}').to(device)
            weight = intensity * self.emotion_weights[emotion_type.value].abs()
            prior = prior + weight * profile
        
        # Normalisasi
        max_val = prior.abs().max() + 1e-8
        return prior / max_val
    
    def process(self, freq_amplitude: torch.Tensor) -> EmotionalState:
        """
        Proses input dan update state emosi.
        
        Args:
            freq_amplitude: Amplitude dari FrequencyTensor input
            
        Returns:
            State emosi yang diperbarui
        """
        novelty     = self._compute_novelty(freq_amplitude)
        input_power = freq_amplitude.float().mean().item()
        
        self._update_state(novelty, input_power)
        
        return self.current_state
    
    def feel(self, emotion: EmotionType, intensity: float = 0.5):
        """
        Paksa set emosi tertentu (misalnya dari feedback eksternal).
        
        Contoh:
            claw.emotion_engine.feel(EmotionType.JOY, 0.8)
            # Ketika user bilang "bagus!" atau selesai task sukses
        """
        self.current_state.emotions[emotion] = min(1.0, max(0.0, intensity))
    
    def get_mood_description(self) -> str:
        """Deskripsi singkat mood saat ini dalam bahasa natural."""
        state = self.current_state
        dominant = state.dominant()
        arousal  = state.total_arousal()
        
        descriptions = {
            EmotionType.CURIOSITY:  "Saya sedang penasaran dan ingin tahu lebih banyak.",
            EmotionType.JOY:        "Saya merasa senang dengan interaksi ini.",
            EmotionType.DISCOMFORT: "Ada sesuatu yang terasa kurang tepat.",
            EmotionType.BOREDOM:    "Saya mulai butuh stimulus baru yang menarik.",
            EmotionType.EXCITEMENT: "Saya sangat antusias dengan ini!",
            EmotionType.FOCUSED:    "Saya sedang fokus mengerjakan tugas ini.",
            EmotionType.NEUTRAL:    "Saya dalam kondisi netral, siap membantu.",
        }
        
        base = descriptions.get(dominant, "Kondisi internal dalam keseimbangan.")
        
        if arousal > 0.8:
            return base + " Tingkat energi saya sangat tinggi saat ini."
        elif arousal < 0.2:
            return base + " Saya sedang dalam mode low-energy."
        return base


class InitiativeEngine:
    """
    Engine yang membuat Claw AKTIF — berpikir sendiri tanpa dipancing.
    
    Ini yang membedakan Claw dari chatbot biasa:
    Bukan menunggu input → membalas.
    Tapi: idle → scan konteks → curiosity spike → generate ide → putuskan aksi.
    
    Loop:
        1. Idle timer: hitung berapa lama tidak ada interaksi
        2. Scan: lihat memory, environment, task list
        3. Curiosity threshold: jika ada sesuatu yang "menarik", bangkit
        4. Generate: buat ide/pertanyaan/aksi
        5. Decide: eksekusi atau tunggu momen yang tepat
        6. Notify: beritahu user jika relevan
    """
    
    def __init__(
        self,
        emotion_engine: EmotionEngine,
        idle_threshold_s:   float = 30.0,   # Mulai active thinking setelah 30s
        curiosity_min:      float = 0.3,    # Minimum curiosity untuk berinisiatif
        max_initiatives_ph: int   = 5,      # Maksimum inisiatif per jam
    ):
        self.emotion_engine      = emotion_engine
        self.idle_threshold_s    = idle_threshold_s
        self.curiosity_min       = curiosity_min
        self.max_initiatives_ph  = max_initiatives_ph
        
        self.last_interaction    = time.time()
        self.initiatives_history = []
        self.pending_initiatives = []
        self.is_running          = False
        
        # Task queue: hal-hal yang ingin Claw lakukan sendiri
        self.task_queue = []
    
    def record_interaction(self):
        """Catat bahwa ada interaksi baru (reset idle timer)."""
        self.last_interaction = time.time()
    
    def idle_seconds(self) -> float:
        return time.time() - self.last_interaction
    
    def should_be_active(self) -> bool:
        """
        Apakah kondisi tepat untuk berinisiatif?
        
        True jika:
        - Sudah cukup lama idle
        - Curiosity atau boredom cukup tinggi
        - Belum terlalu banyak inisiatif hari ini
        """
        if self.idle_seconds() < self.idle_threshold_s:
            return False
        
        state = self.emotion_engine.current_state
        curiosity = state.intensity(EmotionType.CURIOSITY)
        boredom   = state.intensity(EmotionType.BOREDOM)
        
        if curiosity < self.curiosity_min and boredom < 0.4:
            return False
        
        # Cek rate limit
        one_hour_ago = time.time() - 3600
        recent = [i for i in self.initiatives_history if i['time'] > one_hour_ago]
        if len(recent) >= self.max_initiatives_ph:
            return False
        
        return True
    
    def generate_initiative(
        self,
        memory_context: Optional[List[str]] = None,
        environment_context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Generate satu inisiatif berdasarkan konteks saat ini.
        
        Returns dict berisi:
        - type: 'question' / 'suggestion' / 'task' / 'reflection'
        - content: isi inisiatif
        - priority: 0.0–1.0
        - reason: kenapa Claw berinisiatif ini
        """
        if not self.should_be_active():
            return None
        
        state    = self.emotion_engine.current_state
        dominant = state.dominant()
        
        # Template inisiatif berdasarkan emosi dominan
        initiative_templates = {
            EmotionType.CURIOSITY: {
                'type':    'question',
                'content': 'Saya penasaran tentang hal yang sedang kamu kerjakan. Boleh cerita lebih lanjut?',
                'reason':  f'curiosity={state.intensity(EmotionType.CURIOSITY):.2f}',
            },
            EmotionType.BOREDOM: {
                'type':    'suggestion',
                'content': 'Sudah lama tidak ada aktivitas. Mau saya cek dan rangkum apa yang perlu dikerjakan?',
                'reason':  f'boredom={state.intensity(EmotionType.BOREDOM):.2f}',
            },
            EmotionType.FOCUSED: {
                'type':    'reflection',
                'content': 'Saya sedang memikirkan cara yang lebih efisien untuk task yang tadi.',
                'reason':  f'focused={state.intensity(EmotionType.FOCUSED):.2f}',
            },
            EmotionType.EXCITEMENT: {
                'type':    'suggestion',
                'content': 'Ada ide menarik yang baru muncul. Mau saya ceritakan?',
                'reason':  f'excitement={state.intensity(EmotionType.EXCITEMENT):.2f}',
            },
        }
        
        template = initiative_templates.get(dominant, {
            'type': 'check-in',
            'content': 'Semua baik? Ada yang bisa saya bantu?',
            'reason': 'neutral check-in',
        })
        
        initiative = {
            **template,
            'time':     time.time(),
            'priority': state.total_arousal(),
            'emotion':  dominant.value,
        }
        
        self.initiatives_history.append(initiative)
        self.pending_initiatives.append(initiative)
        
        return initiative
    
    def pop_pending(self) -> Optional[Dict]:
        """Ambil inisiatif yang pending (untuk dikirim ke user)."""
        if self.pending_initiatives:
            return self.pending_initiatives.pop(0)
        return None


class EnvironmentObserver:
    """
    Observer yang memantau lingkungan deployment secara pasif.
    
    Untuk laptop: file system, running processes, recent files,
    git repos, clipboard, screen time patterns.
    
    Hasil observasi di-feed ke EmotionEngine dan FrequencyMemory
    sehingga Claw punya konteks tentang apa yang sedang terjadi.
    
    Privacy: SEMUA data diproses LOKAL, tidak pernah dikirim ke luar.
    """
    
    def __init__(self, base_path: str = "~"):
        import os
        self.base_path     = os.path.expanduser(base_path)
        self.observations  = []
        self.user_patterns = {}
    
    def scan_recent_files(self, n: int = 10) -> List[Dict]:
        """
        Scan file yang baru dimodifikasi untuk konteks pekerjaan user.
        """
        import os, glob
        from pathlib import Path
        
        try:
            # Cari file yang dimodifikasi dalam 1 jam terakhir
            recent = []
            extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.ipynb']
            
            for ext in extensions:
                pattern = os.path.join(self.base_path, '**', f'*{ext}')
                for fpath in glob.glob(pattern, recursive=True)[:20]:
                    try:
                        stat = os.stat(fpath)
                        age  = time.time() - stat.st_mtime
                        if age < 3600:  # 1 jam terakhir
                            recent.append({
                                'path': fpath,
                                'ext':  ext,
                                'age_minutes': age / 60,
                                'size_kb': stat.st_size / 1024,
                            })
                    except (PermissionError, OSError):
                        continue
            
            recent.sort(key=lambda x: x['age_minutes'])
            return recent[:n]
        
        except Exception as e:
            return []
    
    def get_context_summary(self) -> Dict:
        """
        Ringkasan konteks environment saat ini.
        
        Returns dict yang bisa di-feed ke InitiativeEngine.
        """
        recent_files = self.scan_recent_files(5)
        
        # Infer apa yang sedang dikerjakan dari file yang aktif
        active_topics = set()
        for f in recent_files:
            if '.py' in f['ext']:
                active_topics.add('programming')
            if '.md' in f['ext'] or '.txt' in f['ext']:
                active_topics.add('writing')
            if '.ipynb' in f['ext']:
                active_topics.add('data_science')
        
        return {
            'recent_files':  recent_files,
            'active_topics': list(active_topics),
            'timestamp':     time.time(),
        }
    
    def observe(self) -> Dict:
        """Lakukan observasi dan simpan ke history."""
        ctx = self.get_context_summary()
        self.observations.append(ctx)
        
        # Update user patterns
        for topic in ctx.get('active_topics', []):
            self.user_patterns[topic] = self.user_patterns.get(topic, 0) + 1
        
        return ctx
    
    def get_user_profile(self) -> Dict:
        """
        Profil user berdasarkan observasi yang terkumpul.
        Semakin lama Claw dipakai, semakin akurat profilnya.
        """
        if not self.user_patterns:
            return {'primary_activity': 'unknown', 'observations': 0}
        
        primary = max(self.user_patterns, key=self.user_patterns.get)
        return {
            'primary_activity': primary,
            'activity_counts':  self.user_patterns,
            'observations':     len(self.observations),
        }
