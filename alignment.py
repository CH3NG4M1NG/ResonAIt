"""
resonait/core/unified_model.py
================================
RESONAIT UNIFIED MODEL — Satu Model, Semua Kemampuan

Ini adalah jantung dari visi ResonAIt:
    Chatbot + Image Gen + Video Gen + TTS + ASR + Music
    → Semuanya di-compress ke Universal Frequency Space
    → Lalu di-reconstruct oleh decoder yang tepat
    → SATU model dengan BANYAK kemampuan (multi-tools)

Analogi:
    Bayangkan otak manusia. Kita tidak punya "modul bicara" terpisah
    dari "modul melihat". Semua persepsi masuk ke satu sistem saraf,
    diproses dalam satu bahasa (sinyal elektrik), lalu output-nya
    berbeda-beda (gerak, suara, pikiran).

    ResonAItUnified melakukan hal yang sama:
    - INPUT  : semua modalitas → FrequencyTensor (satu bahasa)
    - CORE   : FNO Brain memproses di frequency space (satu sistem)
    - OUTPUT : routing ke decoder yang tepat berdasarkan task

Arsitektur:
    ┌─────────────────────────────────────────────┐
    │              INPUT LAYER                    │
    │  Text │ Image │ Audio │ Video │ Custom...   │
    └───────────────┬─────────────────────────────┘
                    │ Universal Frequency Converter
                    ▼
    ┌─────────────────────────────────────────────┐
    │         UNIVERSAL FREQUENCY SPACE           │
    │   (Semua input jadi FrequencyTensor yang    │
    │    bisa dibandingkan & dikombinasikan)       │
    └───────────────┬─────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────────┐
    │         RESONAIT FNO BRAIN CORE             │
    │  Logic │ Imagination │ Memory (paralel)     │
    │  + Task Router (deteksi task apa yg diminta)│
    └───────────────┬─────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          │   TASK ROUTER      │
          │ (pilih decoder)    │
          └─────┬──────────────┘
                │
    ┌───────────┴──────────────────────────────────┐
    │              OUTPUT DECODERS                  │
    │  TextDec │ ImageDec │ AudioDec │ VideoDec...  │
    └──────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from resonait.core.frequency_space import FrequencyTensor, Modality, UniversalFrequencySpace
from resonait.core.brain import ResonAItBrain, FourierNeuralOperatorBlock


# ============================================================
# ENUM: Semua task yang bisa dilakukan UnifiedModel
# ============================================================

class TaskType(Enum):
    """
    Semua kemampuan ResonAIt Unified Model.
    Tambahkan task baru di sini untuk extend kemampuan.
    """
    # === TEXT TASKS ===
    CHAT          = "chat"           # Chatbot / conversation
    SUMMARIZE     = "summarize"      # Ringkasan teks
    TRANSLATE     = "translate"      # Terjemahan bahasa
    CODE          = "code"           # Generate / debug kode
    REASONING     = "reasoning"      # Berpikir step-by-step

    # === IMAGE TASKS ===
    TEXT_TO_IMAGE = "text_to_image"  # Generate gambar dari teks
    IMAGE_CAPTION = "image_caption"  # Deskripsi gambar
    IMAGE_EDIT    = "image_edit"     # Edit gambar dengan instruksi
    IMAGE_QA      = "image_qa"       # Tanya-jawab tentang gambar

    # === AUDIO TASKS ===
    TEXT_TO_SPEECH = "text_to_speech" # TTS: teks → suara
    SPEECH_TO_TEXT = "speech_to_text" # ASR: suara → teks
    TEXT_TO_MUSIC  = "text_to_music"  # Generate musik dari deskripsi

    # === VIDEO TASKS ===
    TEXT_TO_VIDEO  = "text_to_video"  # Generate video dari teks
    VIDEO_CAPTION  = "video_caption"  # Deskripsi konten video

    # === MULTIMODAL TASKS ===
    MULTIMODAL_CHAT = "multimodal_chat" # Chat dengan input campuran
    AUTO            = "auto"            # Auto-detect task dari input


# ============================================================
# DATACLASS: UnifiedInput — Satu format input untuk semua task
# ============================================================

@dataclass
class UnifiedInput:
    """
    Format input universal untuk ResonAIt Unified Model.

    Tidak peduli mau generate teks, gambar, atau video —
    semua dimasukkan lewat format yang sama.

    Attributes:
        task         : Apa yang ingin dilakukan
        text_prompt  : Instruksi teks (hampir selalu ada)
        image_input  : Input gambar (numpy array jika ada)
        audio_input  : Input audio (numpy array jika ada)
        video_input  : Input video (list of frames jika ada)
        output_config: Konfigurasi output (resolusi, durasi, dll)
        context      : Riwayat percakapan sebelumnya
    """
    task:          TaskType
    text_prompt:   Optional[str]            = None
    image_input:   Optional[Any]            = None  # numpy (H,W,3)
    audio_input:   Optional[Any]            = None  # numpy (samples,)
    video_input:   Optional[List[Any]]      = None  # list of frames
    output_config: Dict[str, Any]           = field(default_factory=dict)
    context:       List[Dict[str, str]]     = field(default_factory=list)


@dataclass
class UnifiedOutput:
    """
    Format output universal dari ResonAIt Unified Model.

    Semua hasil dikembalikan dalam format yang sama,
    field yang tidak relevan akan None.

    Attributes:
        task          : Task yang dijalankan
        text_output   : Output teks (untuk chat, caption, ASR, dll)
        image_output  : Output gambar (untuk T2I, image edit)
        audio_output  : Output audio numpy array (untuk TTS, music)
        video_output  : Output video frames (untuk T2V)
        freq_hidden   : FrequencyTensor representasi internal (untuk debugging/chaining)
        confidence    : Confidence score model
        metadata      : Info tambahan
    """
    task:          TaskType
    text_output:   Optional[str]       = None
    image_output:  Optional[Any]       = None  # numpy (H,W,3)
    audio_output:  Optional[Any]       = None  # numpy (samples,)
    video_output:  Optional[List[Any]] = None
    freq_hidden:   Optional[FrequencyTensor] = None
    confidence:    float               = 0.0
    metadata:      Dict[str, Any]      = field(default_factory=dict)


# ============================================================
# TASK ROUTER — Otak yang memilih decoder
# ============================================================

class TaskRouter(nn.Module):
    """
    Router yang memutuskan decoder mana yang harus diaktifkan
    berdasarkan representasi frekuensi internal.

    Bekerja seperti "lobus prefrontal" — bagian otak yang
    memutuskan "sekarang kita harus berbicara, menggambar, atau bernyanyi".

    Args:
        hidden_dim : Dimensi representasi internal
        n_tasks    : Jumlah task yang didukung
    """
    def __init__(self, hidden_dim: int, n_tasks: int):
        super().__init__()

        self.n_tasks = n_tasks

        # Classifier task dari representasi frekuensi
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_tasks),
        )

        # Gate per task: seberapa besar kontribusi tiap decoder
        # (soft routing — tidak harus pilih satu decoder saja)
        self.task_gate = nn.Sequential(
            nn.Linear(hidden_dim, n_tasks),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        forced_task_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route hidden state ke task yang tepat.

        Args:
            hidden          : Representasi internal dari FNO brain.
                             Shape: (batch, hidden_dim, length)
            forced_task_idx : Jika user sudah spesifik minta task tertentu,
                             paksa routing ke sana (bypass classifier)

        Returns:
            Tuple:
            - task_logits : Distribusi probabilitas tiap task
            - task_gates  : Gate weight tiap decoder
        """
        # Global average pooling: (batch, hidden_dim, length) → (batch, hidden_dim)
        pooled = hidden.mean(dim=-1)

        if forced_task_idx is not None:
            # Hard routing: langsung ke task yang diminta
            task_logits = torch.zeros(hidden.shape[0], self.n_tasks, device=hidden.device)
            task_logits[:, forced_task_idx] = 10.0  # Sangat yakin
            task_gates  = torch.softmax(task_logits, dim=-1)
        else:
            # Soft routing: biarkan model memilih sendiri
            task_logits = self.task_classifier(pooled)
            task_gates  = self.task_gate(pooled)

        return task_logits, task_gates


# ============================================================
# BASE DECODER — Semua decoder mewarisi ini
# ============================================================

class BaseDecoder(nn.Module):
    """
    Kelas dasar untuk semua decoder output.

    Setiap decoder mengambil representasi frekuensi internal
    dan menghasilkan output dalam format modalitasnya sendiri.

    Komunitas dapat menambahkan decoder baru dengan mewarisi kelas ini.
    """
    def __init__(self, hidden_dim: int, freq_dim: int, task: TaskType):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_dim   = freq_dim
        self.task       = task

    def forward(self, freq_hidden: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        raise NotImplementedError

    def decode_to_output(self, freq_hidden: torch.Tensor, input_data: UnifiedInput) -> UnifiedOutput:
        """
        Konversi hidden state ke UnifiedOutput.
        Setiap subkelas mengimplementasikan versi spesifiknya.
        """
        raise NotImplementedError


# ============================================================
# DECODER: TEXT (untuk chat, summarize, translate, dll)
# ============================================================

class TextDecoder(BaseDecoder):
    """
    Decoder dari Frequency Space → Token logits → Teks.

    Menggunakan lightweight autoregressive head di atas
    representasi frekuensi. Untuk kualitas terbaik, bisa
    digantikan dengan full LM head dari model yang sudah di-align.

    Alternatif: gunakan LLM yang sudah di-align (via AlignmentTool)
    sebagai decoder — lebih besar tapi jauh lebih bagus.
    """
    def __init__(
        self,
        hidden_dim:  int,
        freq_dim:    int,
        vocab_size:  int = 50257,  # Kompatibel dengan tiktoken/GPT-2
        max_seq_len: int = 2048,
    ):
        super().__init__(hidden_dim, freq_dim, TaskType.CHAT)

        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len

        # Proyeksikan frequency hidden ke dimensi yang sesuai untuk LM head
        self.freq_to_lm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # LM Head: hidden → logit per token di vocab
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Positional encoding untuk sequence generation
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

    def forward(self, freq_hidden: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        """
        freq_hidden: (batch, hidden_dim, freq_length)
        Returns: logits (batch, seq_len, vocab_size)
        """
        # Transpose: (batch, freq_length, hidden_dim)
        x = freq_hidden.transpose(-1, -2)

        # Proyeksi ke LM space
        x = self.freq_to_lm(x)

        # Tambah positional encoding
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)

        # Project ke vocab
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits

    def decode_to_output(self, freq_hidden: torch.Tensor, input_data: UnifiedInput) -> UnifiedOutput:
        """
        Decode frequency representation menjadi teks.

        NOTE: Ini adalah simplified version. Untuk production,
        gunakan proper autoregressive decoding dengan beam search/sampling.
        """
        logits = self.forward(freq_hidden)

        # Greedy decoding: ambil token dengan probabilitas tertinggi
        token_ids = logits.argmax(dim=-1)[0]  # (seq_len,)

        # Decode token IDs ke string
        # (dalam implementasi nyata, pakai tokenizer yang proper)
        text_output = self._simple_decode(token_ids)

        return UnifiedOutput(
            task=input_data.task,
            text_output=text_output,
            freq_hidden=None,
            confidence=torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item(),
        )

    def _simple_decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs ke string.
        Dalam implementasi nyata, pakai AutoTokenizer dari HuggingFace.
        """
        # Placeholder — akan di-replace oleh tokenizer LLM yang di-align
        return f"[TextDecoder output — {token_ids.shape[0]} tokens]"


# ============================================================
# DECODER: IMAGE (untuk text-to-image)
# ============================================================

class ImageDecoder(BaseDecoder):
    """
    Decoder dari Frequency Space → Gambar.

    Menggunakan Inverse FFT + Convolutional Upsampler untuk
    merekonstruksi gambar dari representasi frekuensi.

    Untuk kualitas production, ini bisa digantikan dengan
    Stable Diffusion VAE decoder yang sudah di-align.
    """
    def __init__(
        self,
        hidden_dim:  int,
        freq_dim:    int,
        image_size:  int = 256,  # Output (image_size, image_size, 3)
        channels:    int = 3,
    ):
        super().__init__(hidden_dim, freq_dim, TaskType.TEXT_TO_IMAGE)

        self.image_size = image_size
        self.channels   = channels

        # Proyeksikan frequency hidden ke spatial feature map
        self.freq_to_spatial = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, (image_size // 16) ** 2 * hidden_dim),
        )

        # Convolutional upsampler: spatial features → gambar penuh
        # Setiap ConvTranspose2d 2x upscale
        self.conv_upsample = nn.Sequential(
            # (hidden_dim, 16, 16) → (hidden_dim//2, 32, 32)
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            # → (hidden_dim//4, 64, 64)
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            # → (hidden_dim//8, 128, 128)
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 8),
            nn.GELU(),
            # → (channels, 256, 256)
            nn.ConvTranspose2d(hidden_dim // 8, channels, 4, stride=2, padding=1),
            nn.Tanh(),  # Output -1 sampai 1 (standard untuk gambar)
        )

    def forward(self, freq_hidden: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        """
        freq_hidden: (batch, hidden_dim, freq_length)
        Returns: image tensor (batch, channels, H, W)
        """
        batch = freq_hidden.shape[0]

        # Pool over frequency dimension
        pooled = freq_hidden.mean(dim=-1)  # (batch, hidden_dim)

        # Proyeksi ke spatial feature map
        spatial = self.freq_to_spatial(pooled)  # (batch, H*W*hidden_dim)

        # Reshape ke feature map 2D
        h = w = self.image_size // 16
        spatial = spatial.view(batch, self.hidden_dim, h, w)  # (batch, hidden, h, w)

        # Upsample ke ukuran gambar penuh
        image = self.conv_upsample(spatial)  # (batch, channels, image_size, image_size)

        return image

    def decode_to_output(self, freq_hidden: torch.Tensor, input_data: UnifiedInput) -> UnifiedOutput:
        import numpy as np

        image_tensor = self.forward(freq_hidden)  # (batch, C, H, W)

        # Konversi ke numpy array: (H, W, C), range [0, 255]
        image_np = image_tensor[0].detach().cpu()
        image_np = ((image_np + 1.0) / 2.0).clamp(0, 1)  # [-1,1] → [0,1]
        image_np = (image_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return UnifiedOutput(
            task=input_data.task,
            image_output=image_np,
            metadata={
                "size": f"{self.image_size}x{self.image_size}",
                "channels": self.channels,
                "prompt": input_data.text_prompt,
            }
        )


# ============================================================
# DECODER: AUDIO (untuk TTS dan Music)
# ============================================================

class AudioDecoder(BaseDecoder):
    """
    Decoder dari Frequency Space → Audio waveform.

    Menggunakan IFFT untuk merekonstruksi sinyal audio dari
    representasi frekuensi. Ini secara alami cocok karena
    audio ADALAH sinyal frekuensi (via Fourier).

    Untuk TTS berkualitas tinggi: integrasikan dengan Kokoro/Chatterbox vocoder.
    Untuk music: integrasikan dengan MusicGen decoder.
    """
    def __init__(
        self,
        hidden_dim:  int,
        freq_dim:    int,
        sample_rate: int = 22050,
        duration_s:  float = 5.0,
    ):
        super().__init__(hidden_dim, freq_dim, TaskType.TEXT_TO_SPEECH)

        self.sample_rate = sample_rate
        self.n_samples   = int(sample_rate * duration_s)

        # Proyeksikan hidden ke spectral representation
        self.hidden_to_spectral = nn.Sequential(
            nn.Linear(hidden_dim, freq_dim * 2),
            nn.GELU(),
            nn.Linear(freq_dim * 2, freq_dim * 2),  # amplitude + phase
        )

        # Post-processing: smoothing filter untuk mengurangi artifacts
        self.smoothing = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        nn.init.constant_(self.smoothing.weight, 1.0 / 5)

    def forward(self, freq_hidden: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        """
        freq_hidden: (batch, hidden_dim, freq_length)
        Returns: waveform (batch, n_samples)
        """
        # Pool dan proyeksikan ke spectral representation
        pooled   = freq_hidden.mean(dim=-1)             # (batch, hidden_dim)
        spectral = self.hidden_to_spectral(pooled)      # (batch, freq_dim*2)

        # Split menjadi amplitude dan phase
        amplitude = spectral[:, :self.freq_dim]          # (batch, freq_dim)
        phase     = spectral[:, self.freq_dim:]          # (batch, freq_dim)

        # Buat representasi kompleks
        complex_spec = torch.complex(
            amplitude * torch.cos(phase),
            amplitude * torch.sin(phase),
        )

        # IFFT: frequency domain → time domain
        waveform = torch.fft.irfft(complex_spec, n=self.n_samples, norm="ortho")
        # (batch, n_samples)

        # Smoothing untuk mengurangi high-frequency artifacts
        waveform = self.smoothing(waveform.unsqueeze(1)).squeeze(1)

        # Normalisasi ke [-1, 1]
        max_val  = waveform.abs().max(dim=-1, keepdim=True).values + 1e-8
        waveform = waveform / max_val

        return waveform

    def decode_to_output(self, freq_hidden: torch.Tensor, input_data: UnifiedInput) -> UnifiedOutput:
        import numpy as np

        waveform = self.forward(freq_hidden)  # (batch, n_samples)
        audio_np = waveform[0].detach().cpu().numpy()

        return UnifiedOutput(
            task=input_data.task,
            audio_output=audio_np,
            metadata={
                "sample_rate": self.sample_rate,
                "duration_s":  len(audio_np) / self.sample_rate,
                "prompt":      input_data.text_prompt,
            }
        )


# ============================================================
# DECODER: VIDEO (untuk text-to-video)
# ============================================================

class VideoDecoder(BaseDecoder):
    """
    Decoder dari Frequency Space → Sequence of Image Frames.

    Menggunakan ImageDecoder per frame dengan temporal consistency
    layer untuk memastikan frame-to-frame coherence.

    Untuk kualitas production: integrasikan dengan CogVideoX/Wan2.2 decoder.
    """
    def __init__(
        self,
        hidden_dim:  int,
        freq_dim:    int,
        image_size:  int = 256,
        n_frames:    int = 16,
        fps:         int = 8,
    ):
        super().__init__(hidden_dim, freq_dim, TaskType.TEXT_TO_VIDEO)

        self.n_frames  = n_frames
        self.fps       = fps
        self.image_size = image_size

        # Per-frame decoder (shared weights untuk efisiensi)
        self.frame_decoder = ImageDecoder(hidden_dim, freq_dim, image_size)

        # Temporal consistency: pastikan frame berurutan koheren
        # Menggunakan 1D conv over temporal dimension
        self.temporal_smooth = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3, padding=1,
            bias=False,
        )

        # Frame-specific positional embedding
        self.frame_pos_embed = nn.Embedding(n_frames, hidden_dim)

    def forward(self, freq_hidden: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        """
        freq_hidden: (batch, hidden_dim, freq_length)
        Returns: video frames (batch, n_frames, channels, H, W)
        """
        batch     = freq_hidden.shape[0]
        all_frames = []

        # Temporal smoothing over frequency axis
        smoothed = self.temporal_smooth(freq_hidden)

        for t in range(self.n_frames):
            # Tambahkan frame positional embedding
            t_embed = self.frame_pos_embed(
                torch.tensor([t], device=freq_hidden.device)
            ).unsqueeze(-1)  # (1, hidden_dim, 1)

            # Frame-specific hidden state
            frame_hidden = smoothed + t_embed  # (batch, hidden_dim, freq_length)

            # Decode frame ini
            frame = self.frame_decoder(frame_hidden)  # (batch, 3, H, W)
            all_frames.append(frame)

        # Stack frames: (batch, n_frames, 3, H, W)
        video = torch.stack(all_frames, dim=1)
        return video

    def decode_to_output(self, freq_hidden: torch.Tensor, input_data: UnifiedInput) -> UnifiedOutput:
        import numpy as np

        video_tensor = self.forward(freq_hidden)  # (batch, T, C, H, W)

        # Konversi ke list of numpy arrays
        frames = []
        for t in range(self.n_frames):
            frame = video_tensor[0, t].detach().cpu()
            frame = ((frame + 1.0) / 2.0).clamp(0, 1)
            frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames.append(frame)

        return UnifiedOutput(
            task=input_data.task,
            video_output=frames,
            metadata={
                "n_frames":   self.n_frames,
                "fps":        self.fps,
                "duration_s": self.n_frames / self.fps,
                "size":       f"{self.image_size}x{self.image_size}",
                "prompt":     input_data.text_prompt,
            }
        )


# ============================================================
# KELAS UTAMA: ResonAItUnified
# ============================================================

class ResonAItUnified(nn.Module):
    """
    RESONAIT UNIFIED MODEL — Satu Model, Semua Kemampuan.

    Ini adalah model final yang menggabungkan:
    - ResonAItBrain (FNO core)
    - TaskRouter (pilih decoder)
    - Semua decoder (text, image, audio, video)

    Cara kerja:
        1. Input (teks/gambar/audio/video) → FrequencyTensor
        2. FNO Brain memproses di frequency space
        3. TaskRouter memilih decoder yang tepat
        4. Decoder menghasilkan output dalam format yang diminta
        5. Output dikembalikan dalam UnifiedOutput

    Penggunaan:
        >>> model = ResonAItUnified.from_scratch()
        >>> result = model.run(UnifiedInput(
        ...     task=TaskType.TEXT_TO_IMAGE,
        ...     text_prompt="Kucing bermain di taman"
        ... ))
        >>> # result.image_output → numpy array gambar

    Args:
        freq_dim    : Dimensi frequency space
        hidden_dim  : Dimensi representasi internal
        n_modes     : Mode Fourier yang dipertahankan
        n_fno_layers: Kedalaman FNO encoder
        image_size  : Ukuran output gambar (untuk T2I dan T2V)
        n_frames    : Jumlah frame video (untuk T2V)
        vocab_size  : Ukuran vocabulary (untuk text decoder)
        sample_rate : Sample rate audio (untuk TTS/music)
    """

    # Mapping TaskType → index untuk router
    TASK_LIST = [
        TaskType.CHAT, TaskType.SUMMARIZE, TaskType.TRANSLATE,
        TaskType.CODE, TaskType.REASONING,
        TaskType.TEXT_TO_IMAGE, TaskType.IMAGE_CAPTION,
        TaskType.IMAGE_EDIT, TaskType.IMAGE_QA,
        TaskType.TEXT_TO_SPEECH, TaskType.SPEECH_TO_TEXT,
        TaskType.TEXT_TO_MUSIC,
        TaskType.TEXT_TO_VIDEO, TaskType.VIDEO_CAPTION,
        TaskType.MULTIMODAL_CHAT,
    ]

    def __init__(
        self,
        freq_dim:     int = 512,
        hidden_dim:   int = 256,
        n_modes:      int = 64,
        n_fno_layers: int = 4,
        image_size:   int = 256,
        n_frames:     int = 16,
        vocab_size:   int = 50257,
        sample_rate:  int = 22050,
    ):
        super().__init__()

        # Simpan config
        self.freq_dim    = freq_dim
        self.hidden_dim  = hidden_dim
        self.n_modes     = n_modes
        self.image_size  = image_size
        self.n_frames    = n_frames
        self.vocab_size  = vocab_size
        self.sample_rate = sample_rate
        self.n_tasks     = len(self.TASK_LIST)
        self.task_to_idx = {t: i for i, t in enumerate(self.TASK_LIST)}

        # ── 1. FREQUENCY SPACE ─────────────────────────────────────
        self.frequency_space = UniversalFrequencySpace(
            freq_dim=freq_dim,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
        )

        # ── 2. FNO BRAIN CORE ──────────────────────────────────────
        self.brain = ResonAItBrain(
            freq_dim=freq_dim,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_fno_layers=n_fno_layers,
        )

        # ── 3. CROSS-MODAL FUSION ──────────────────────────────────
        # Jika ada multiple input modalitas, perlu digabung
        self.modal_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # ── 4. TASK ROUTER ─────────────────────────────────────────
        self.task_router = TaskRouter(hidden_dim, self.n_tasks)

        # ── 5. OUTPUT DECODERS ─────────────────────────────────────
        self.decoders = nn.ModuleDict({
            # Text decoder (dipakai oleh: chat, summarize, translate, dll)
            "text": TextDecoder(
                hidden_dim=hidden_dim,
                freq_dim=freq_dim,
                vocab_size=vocab_size,
            ),
            # Image decoder (text-to-image, image edit)
            "image": ImageDecoder(
                hidden_dim=hidden_dim,
                freq_dim=freq_dim,
                image_size=image_size,
            ),
            # Audio decoder (TTS, music generation)
            "audio": AudioDecoder(
                hidden_dim=hidden_dim,
                freq_dim=freq_dim,
                sample_rate=sample_rate,
            ),
            # Video decoder (text-to-video)
            "video": VideoDecoder(
                hidden_dim=hidden_dim,
                freq_dim=freq_dim,
                image_size=image_size,
                n_frames=n_frames,
            ),
        })

        # ── 6. TASK → DECODER MAPPING ──────────────────────────────
        self._task_decoder_map = {
            TaskType.CHAT:           "text",
            TaskType.SUMMARIZE:      "text",
            TaskType.TRANSLATE:      "text",
            TaskType.CODE:           "text",
            TaskType.REASONING:      "text",
            TaskType.IMAGE_CAPTION:  "text",
            TaskType.IMAGE_QA:       "text",
            TaskType.SPEECH_TO_TEXT: "text",
            TaskType.VIDEO_CAPTION:  "text",
            TaskType.MULTIMODAL_CHAT:"text",
            TaskType.TEXT_TO_IMAGE:  "image",
            TaskType.IMAGE_EDIT:     "image",
            TaskType.TEXT_TO_SPEECH: "audio",
            TaskType.TEXT_TO_MUSIC:  "audio",
            TaskType.TEXT_TO_VIDEO:  "video",
        }

        # ── 7. SPECIALIST MODEL SLOTS ──────────────────────────────
        # Slot untuk model specialist yang sudah di-align
        # (Llama, SD, Whisper, dll) — diisi oleh UnifiedTrainer
        self._specialist_models: Dict[str, nn.Module] = {}
        self._specialist_aligned: Dict[str, bool]     = {}

    def register_specialist(self, name: str, model: nn.Module, task: str):
        """
        Daftarkan model specialist yang sudah di-align ke frequency space.

        Specialist model (misalnya Llama-3 yang sudah di-align) akan
        digunakan sebagai "expert decoder" yang jauh lebih powerful
        dibanding decoder bawaan.

        Args:
            name  : Nama identifier specialist
            model : Model yang sudah di-align (via AlignmentTool)
            task  : Task yang di-handle specialist ini
        """
        self._specialist_models[name]  = model
        self._specialist_aligned[name] = True
        print(f"[UnifiedModel] ✓ Specialist '{name}' terdaftar untuk task '{task}'")

    def _get_converter(self):
        """Lazy-load universal converter."""
        if not hasattr(self, '_converter') or self._converter is None:
            from resonait.converters.universal_converter import UniversalFrequencyConverter
            self._converter = UniversalFrequencyConverter(freq_dim=self.freq_dim)
        return self._converter

    def _encode_inputs(self, input_data: UnifiedInput) -> torch.Tensor:
        """
        Encode semua input modalitas ke representasi frekuensi internal.

        Jika ada multiple modalitas, akan di-fuse menggunakan
        cross-modal attention.

        Returns:
            hidden: (batch, hidden_dim, freq_dim) representasi internal
        """
        converter = self._get_converter()
        freq_tensors = []

        # Encode setiap modalitas yang ada
        if input_data.text_prompt:
            freq = converter.convert(input_data.text_prompt, modality="text")
            projected = self.frequency_space.project(freq)  # (1, freq_dim, hidden_dim)
            freq_tensors.append(projected)

        if input_data.image_input is not None:
            freq = converter.convert(input_data.image_input, modality="image")
            # Rata-rata over channel → (1, freq_dim, hidden_dim)
            projected = self.frequency_space.project(freq)
            freq_tensors.append(projected.mean(dim=1, keepdim=True))

        if input_data.audio_input is not None:
            freq = converter.convert(input_data.audio_input, modality="audio")
            projected = self.frequency_space.project(freq)
            freq_tensors.append(projected)

        if not freq_tensors:
            # Fallback: zero tensor
            freq_tensors.append(
                torch.zeros(1, 1, self.hidden_dim, device=next(self.parameters()).device)
            )

        if len(freq_tensors) == 1:
            # Single modality — tidak perlu fusion
            combined = freq_tensors[0]  # (1, freq_dim, hidden_dim)
        else:
            # Multi-modal: fuse via cross-modal attention
            # Stack semua: (1, n_modalities * freq_dim, hidden_dim)
            stacked = torch.cat(freq_tensors, dim=1)  # (1, total_len, hidden_dim)

            # Self-attention untuk cross-modal fusion
            fused, _ = self.modal_fusion(stacked, stacked, stacked)
            combined = fused[:, :freq_tensors[0].shape[1], :]  # Ambil length pertama

        # Transpose untuk FNO: (batch, hidden_dim, freq_dim)
        hidden = combined.transpose(-1, -2)

        return hidden

    def forward(
        self,
        input_data: UnifiedInput,
        return_hidden: bool = False,
    ) -> UnifiedOutput:
        """
        Forward pass utama Unified Model.

        Ini adalah satu-satunya fungsi yang perlu dipanggil
        untuk semua task. Model akan menangani sisanya.

        Args:
            input_data   : UnifiedInput dengan task dan data
            return_hidden: Jika True, sertakan freq_hidden di output

        Returns:
            UnifiedOutput dengan hasil yang sesuai task
        """
        # ── Step 1: Encode semua input ke frequency space ──────────
        hidden = self._encode_inputs(input_data)

        # ── Step 2: Proses lewat FNO Brain (core processing) ───────
        brain_output = self.brain.think(hidden)
        processed_hidden = brain_output["output"]
        # Shape: (batch, hidden_dim, freq_dim) — tetap sama

        # ── Step 3: Task Routing ────────────────────────────────────
        task = input_data.task

        if task == TaskType.AUTO:
            # Auto-detect: biarkan router yang memilih
            task_logits, _ = self.task_router(processed_hidden)
            task_idx = task_logits.argmax(dim=-1).item()
            task     = self.TASK_LIST[task_idx]
        else:
            # User sudah spesifik — forced routing
            task_idx = self.task_to_idx.get(task, 0)

        # ── Step 4: Decode dengan decoder yang tepat ────────────────
        decoder_key = self._task_decoder_map.get(task, "text")

        # Cek apakah ada specialist model yang lebih powerful
        specialist_key = f"{decoder_key}_specialist"
        if specialist_key in self._specialist_models:
            # Gunakan specialist model yang sudah di-align
            result = self._decode_with_specialist(
                specialist_key, processed_hidden, input_data
            )
        else:
            # Gunakan decoder bawaan
            decoder = self.decoders[decoder_key]
            result  = decoder.decode_to_output(processed_hidden, input_data)

        # ── Step 5: Tambahkan metadata ──────────────────────────────
        result.metadata.update({
            "task_detected":  task.value,
            "decoder_used":   decoder_key,
            "logic_gate":     brain_output.get("logic_gate", 0),
            "imagination":    brain_output.get("imagination_gate", 0),
            "pain_level":     brain_output.get("pain_level", 0),
            "has_specialist": specialist_key in self._specialist_models,
        })

        if return_hidden:
            result.freq_hidden = None  # Bisa di-set jika diperlukan

        return result

    def _decode_with_specialist(
        self,
        specialist_key: str,
        hidden: torch.Tensor,
        input_data: UnifiedInput,
    ) -> UnifiedOutput:
        """
        Decode menggunakan specialist model yang sudah di-align.

        Specialist model (Llama-3, SD, Whisper, dll) menerima
        representasi frekuensi dan menghasilkan output berkualitas tinggi.
        """
        specialist = self._specialist_models[specialist_key]
        # Delegate ke specialist — interface akan berbeda per specialist
        # Ini adalah extension point untuk integrasi model besar
        return specialist(hidden, input_data)

    def run(self, input_data: UnifiedInput) -> UnifiedOutput:
        """
        Shortcut untuk forward() — interface yang lebih bersih.

        Contoh:
            >>> model.run(UnifiedInput(task=TaskType.CHAT, text_prompt="Hai!"))
        """
        with torch.no_grad():
            return self.forward(input_data)

    @classmethod
    def from_scratch(
        cls,
        freq_dim:   int = 512,
        hidden_dim: int = 256,
        **kwargs
    ) -> "ResonAItUnified":
        """Buat model baru dari nol (random weights)."""
        model = cls(freq_dim=freq_dim, hidden_dim=hidden_dim, **kwargs)
        total = sum(p.numel() for p in model.parameters())
        print(f"[ResonAItUnified] ✓ Model dibuat dari nol")
        print(f"[ResonAItUnified]   Total parameter: {total:,}")
        print(f"[ResonAItUnified]   Task yang didukung: {len(cls.TASK_LIST)}")
        return model

    @classmethod
    def from_checkpoint(cls, path: str) -> "ResonAItUnified":
        """Load model dari checkpoint."""
        ckpt  = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        print(f"[ResonAItUnified] ✓ Model dimuat dari: {path}")
        return model

    def save(self, path: str):
        """Simpan model ke checkpoint."""
        torch.save({
            "config": {
                "freq_dim":    self.freq_dim,
                "hidden_dim":  self.hidden_dim,
                "n_modes":     self.n_modes,
                "image_size":  self.image_size,
                "n_frames":    self.n_frames,
                "vocab_size":  self.vocab_size,
                "sample_rate": self.sample_rate,
            },
            "state_dict": self.state_dict(),
        }, path)
        print(f"[ResonAItUnified] ✓ Model disimpan ke: {path}")

    def get_model_card(self) -> str:
        """Cetak ringkasan model yang indah."""
        total      = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        specialists = list(self._specialist_models.keys())

        card = f"""
╔══════════════════════════════════════════════════════════╗
║           ResonAIt UNIFIED MODEL — Model Card            ║
╠══════════════════════════════════════════════════════════╣
║  Arsitektur   : Fourier Neural Operator + Multi-Decoder  ║
║  Total params : {total:>15,}                         ║
║  Trainable    : {trainable:>15,}                         ║
║  Freq dim     : {self.freq_dim:>15}                         ║
║  Hidden dim   : {self.hidden_dim:>15}                         ║
╠══════════════════════════════════════════════════════════╣
║  TASK YANG DIDUKUNG ({self.n_tasks} tasks):                    ║
║    Text  : chat, summarize, translate, code, reasoning   ║
║    Image : text_to_image, caption, edit, QA              ║
║    Audio : TTS, ASR, music                               ║
║    Video : text_to_video, caption                        ║
╠══════════════════════════════════════════════════════════╣
║  DECODERS:                                               ║
║    [✓] TextDecoder   [✓] ImageDecoder                    ║
║    [✓] AudioDecoder  [✓] VideoDecoder                    ║
╠══════════════════════════════════════════════════════════╣
║  SPECIALIST MODELS ({len(specialists)} terdaftar):                  ║
║    {', '.join(specialists) if specialists else 'Belum ada — jalankan UnifiedTrainer'}
╚══════════════════════════════════════════════════════════╝
"""
        return card
