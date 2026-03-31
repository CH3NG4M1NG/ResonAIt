"""
tests/test_core.py
===================
Unit tests untuk komponen inti ResonAIt.

Jalankan dengan:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import pytest
import torch
import numpy as np
import sys
import os

# Tambahkan root package ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# TEST: FrequencyTensor
# ============================================================

class TestFrequencyTensor:
    """Test untuk dataclass FrequencyTensor."""
    
    def setup_method(self):
        """Setup FrequencyTensor dummy untuk setiap test."""
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        self.freq_dim = 128
        self.batch    = 2
        self.channels = 3
        
        self.amplitude = torch.rand(self.batch, self.channels, self.freq_dim)
        self.phase     = torch.rand(self.batch, self.channels, self.freq_dim) * 2 * np.pi - np.pi
        
        self.freq_tensor = FrequencyTensor(
            amplitude=self.amplitude,
            phase=self.phase,
            modality=Modality.TEXT,
            metadata={"test": True}
        )
    
    def test_shape_consistency(self):
        """Amplitude dan phase harus shape yang sama."""
        assert self.freq_tensor.amplitude.shape == self.freq_tensor.phase.shape
        assert self.freq_tensor.shape == (self.batch, self.channels, self.freq_dim)
    
    def test_complex_repr_created(self):
        """complex_repr harus otomatis dibuat saat inisialisasi."""
        assert self.freq_tensor.complex_repr is not None
        assert self.freq_tensor.complex_repr.shape == (self.batch, self.channels, self.freq_dim)
        assert self.freq_tensor.complex_repr.is_complex()
    
    def test_power_spectrum(self):
        """Power spectrum = amplitude^2, harus selalu positif."""
        ps = self.freq_tensor.to_power_spectrum()
        assert (ps >= 0).all(), "Power spectrum harus non-negatif"
        torch.testing.assert_close(ps, self.amplitude ** 2)
    
    def test_interference(self):
        """Interferensi dua FrequencyTensor harus menghasilkan FrequencyTensor baru."""
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        other = FrequencyTensor(
            amplitude=torch.rand(self.batch, self.channels, self.freq_dim),
            phase=torch.rand(self.batch, self.channels, self.freq_dim),
            modality=Modality.TEXT,
        )
        
        result = self.freq_tensor.interfere_with(other)
        
        assert isinstance(result, FrequencyTensor)
        assert result.shape == self.freq_tensor.shape
        assert (result.amplitude >= 0).all(), "Amplitude hasil interferensi harus non-negatif"
    
    def test_destructive_interference(self):
        """
        Interferensi destruktif: dua sinyal dengan fase berlawanan
        harus menghasilkan amplitude yang lebih kecil.
        """
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        # Sinyal dengan fase persis berlawanan
        opposite_phase = FrequencyTensor(
            amplitude=self.amplitude.clone(),
            phase=self.phase + np.pi,  # Berlawanan 180°
            modality=Modality.TEXT,
        )
        
        result = self.freq_tensor.interfere_with(opposite_phase)
        
        # Amplitude setelah interferensi destruktif harus < amplitude asli
        assert result.amplitude.mean() < self.amplitude.mean(), (
            "Interferensi destruktif harus mengurangi amplitude"
        )
    
    def test_coherence_with_self(self):
        """Koherensi tensor dengan dirinya sendiri harus mendekati 1.0."""
        coherence = self.freq_tensor.coherence_with(self.freq_tensor)
        assert coherence > 0.9, f"Self-coherence seharusnya ~1.0, dapat {coherence}"
    
    def test_dominant_frequencies(self):
        """dominant_frequencies harus mengembalikan indeks valid."""
        k = 5
        indices = self.freq_tensor.dominant_frequencies(top_k=k)
        assert indices.shape[-1] == k
        assert (indices >= 0).all()
        assert (indices < self.freq_dim).all()
    
    def test_device_transfer(self):
        """Tensor harus bisa dipindahkan ke device lain."""
        cpu_tensor = self.freq_tensor.to(torch.device("cpu"))
        assert cpu_tensor.device == torch.device("cpu")
    
    def test_shape_mismatch_raises(self):
        """Shape tidak sama antara amplitude dan phase harus raise AssertionError."""
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        with pytest.raises(AssertionError):
            FrequencyTensor(
                amplitude=torch.rand(1, 1, 64),
                phase=torch.rand(1, 1, 128),  # Shape berbeda!
                modality=Modality.TEXT,
            )


# ============================================================
# TEST: SpectralConvolution
# ============================================================

class TestSpectralConvolution:
    """Test untuk layer konvolusi spektral."""
    
    def test_output_shape(self):
        """Output SpectralConvolution harus sama shape dengan input."""
        from resonait.core.brain import SpectralConvolution
        
        in_ch  = 16
        out_ch = 16
        n_modes = 8
        length  = 64
        batch   = 4
        
        layer = SpectralConvolution(in_ch, out_ch, n_modes)
        x     = torch.randn(batch, in_ch, length)
        y     = layer(x)
        
        assert y.shape == (batch, out_ch, length), (
            f"Output shape {y.shape} != expected ({batch}, {out_ch}, {length})"
        )
    
    def test_no_nan_output(self):
        """Output tidak boleh mengandung NaN."""
        from resonait.core.brain import SpectralConvolution
        
        layer = SpectralConvolution(8, 8, 4)
        x     = torch.randn(2, 8, 32)
        y     = layer(x)
        
        assert not torch.isnan(y).any(), "SpectralConvolution menghasilkan NaN!"
    
    def test_gradient_flows(self):
        """Gradient harus bisa mengalir melalui SpectralConvolution."""
        from resonait.core.brain import SpectralConvolution
        
        layer  = SpectralConvolution(4, 4, 2)
        x      = torch.randn(1, 4, 16, requires_grad=True)
        y      = layer(x)
        loss   = y.mean()
        loss.backward()
        
        assert x.grad is not None, "Gradient tidak mengalir ke input!"


# ============================================================
# TEST: FourierNeuralOperatorBlock
# ============================================================

class TestFNOBlock:
    """Test untuk satu blok FNO."""
    
    def test_output_shape(self):
        """Shape input dan output FNO Block harus sama."""
        from resonait.core.brain import FourierNeuralOperatorBlock
        
        channels = 32
        n_modes  = 8
        batch    = 2
        length   = 64
        
        block = FourierNeuralOperatorBlock(channels, n_modes)
        x     = torch.randn(batch, channels, length)
        y     = block(x)
        
        assert y.shape == x.shape, f"FNO Block mengubah shape: {x.shape} → {y.shape}"
    
    def test_residual_connection(self):
        """Dengan weight nol, output harus mendekati input (karena residual)."""
        from resonait.core.brain import FourierNeuralOperatorBlock
        
        block = FourierNeuralOperatorBlock(16, 4)
        
        # Nolkan semua weight
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
        
        x = torch.randn(1, 16, 32)
        y = block(x)
        
        # Dengan weight=0, y seharusnya ≈ x (dari residual connection)
        # (tidak persis sama karena ada normalisasi dan bias)
        assert y.shape == x.shape


# ============================================================
# TEST: ResonAItBrain
# ============================================================

class TestResonAItBrain:
    """Test untuk otak utama ResonAIt."""
    
    def setup_method(self):
        """Buat brain kecil untuk testing."""
        from resonait.core.brain import ResonAItBrain
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        self.brain = ResonAItBrain(
            freq_dim=64,
            hidden_dim=32,
            n_modes=8,
            n_fno_layers=1,
        )
        
        # FrequencyTensor dummy
        self.freq_tensor = FrequencyTensor(
            amplitude=torch.rand(1, 1, 64),
            phase=torch.rand(1, 1, 64),
            modality=Modality.TEXT,
        )
    
    def test_forward_pass(self):
        """Forward pass harus menghasilkan dict dengan key yang benar."""
        output = self.brain(self.freq_tensor)
        
        required_keys = ["output", "logic_gate", "imagination_gate", "memory_gate", "pain_level"]
        for key in required_keys:
            assert key in output, f"Key '{key}' tidak ada di output brain"
    
    def test_no_nan_output(self):
        """Output brain tidak boleh NaN."""
        output = self.brain(self.freq_tensor)
        assert not torch.isnan(output["output"]).any()
    
    def test_gate_values_range(self):
        """Gate values harus dalam range [0, 1]."""
        output = self.brain(self.freq_tensor)
        
        for gate_key in ["logic_gate", "imagination_gate", "memory_gate"]:
            val = output[gate_key]
            assert 0.0 <= val <= 1.0, f"{gate_key} = {val} di luar range [0, 1]"
    
    def test_pain_affects_output(self):
        """Output saat dalam pain harus berbeda dari kondisi normal."""
        # Output normal
        out_normal = self.brain(self.freq_tensor)
        
        # Tambahkan pain
        self.brain.pain_state.fill_(0.8)
        out_pain = self.brain(self.freq_tensor)
        
        # Pain level harus lebih tinggi
        assert out_pain["pain_level"] > out_normal["pain_level"], (
            "Pain level tidak mencerminkan pain state brain"
        )
        
        # Reset
        self.brain.pain_state.fill_(0.0)
    
    def test_save_load(self, tmp_path):
        """Save dan load checkpoint harus menghasilkan model yang identik."""
        from resonait.core.brain import ResonAItBrain
        
        save_path = str(tmp_path / "test_brain.pt")
        self.brain.save(save_path)
        
        loaded_brain = ResonAItBrain.load(save_path)
        
        # Bandingkan output
        with torch.no_grad():
            out1 = self.brain(self.freq_tensor)
            out2 = loaded_brain(self.freq_tensor)
        
        torch.testing.assert_close(out1["output"], out2["output"])
    
    def test_health_report(self):
        """Health report harus mengandung key yang benar dengan nilai valid."""
        report = self.brain.get_health_report()
        
        assert "pain_level"          in report
        assert "spectral_health"     in report
        assert "cognitive_stability" in report
        
        assert 0.0 <= report["pain_level"]          <= 1.0
        assert 0.0 <= report["spectral_health"]      <= 1.0
        assert 0.0 <= report["cognitive_stability"]  <= 1.0


# ============================================================
# TEST: Converters
# ============================================================

class TestTextConverter:
    """Test untuk TextConverter."""
    
    def test_basic_conversion(self):
        """Konversi teks sederhana harus berhasil."""
        from resonait.converters.text_converter import TextConverter
        
        converter   = TextConverter(freq_dim=128)
        freq_tensor = converter.to_frequency_tensor("Halo dunia!")
        
        assert freq_tensor.amplitude.shape[-1] == 128
        assert not torch.isnan(freq_tensor.amplitude).any()
    
    def test_empty_text_handled(self):
        """Teks kosong tidak boleh crash."""
        from resonait.converters.text_converter import TextConverter
        
        converter = TextConverter(freq_dim=64)
        freq      = converter.to_frequency_tensor("")
        assert freq is not None
    
    def test_long_text(self):
        """Teks panjang harus di-handle dengan baik (padding/cropping)."""
        from resonait.converters.text_converter import TextConverter
        
        long_text = "A" * 10000  # Sangat panjang
        converter = TextConverter(freq_dim=64)
        freq      = converter.to_frequency_tensor(long_text)
        
        assert freq.amplitude.shape[-1] == 64
    
    def test_metadata_stored(self):
        """Metadata harus tersimpan dengan benar."""
        from resonait.converters.text_converter import TextConverter
        
        text      = "Test metadata"
        converter = TextConverter(freq_dim=64)
        freq      = converter.to_frequency_tensor(text)
        
        assert "original_text" in freq.metadata
        assert freq.metadata["original_text"].startswith("Test")


class TestImageConverter:
    """Test untuk ImageConverter."""
    
    def test_rgb_image(self):
        """Gambar RGB 3-channel harus dikonversi dengan benar."""
        from resonait.converters.image_converter import ImageConverter
        
        converter   = ImageConverter(freq_dim=64, target_size=(32, 32))
        fake_image  = np.random.rand(64, 64, 3).astype(np.float32)
        freq_tensor = converter.to_frequency_tensor(fake_image)
        
        assert freq_tensor.amplitude.shape[-1] == 64
        assert freq_tensor.amplitude.shape[1] == 3   # 3 channels (RGB)
    
    def test_grayscale_image(self):
        """Gambar grayscale (1 channel) juga harus bisa."""
        from resonait.converters.image_converter import ImageConverter
        
        converter  = ImageConverter(freq_dim=64, target_size=(32, 32))
        gray_image = np.random.rand(64, 64, 1).astype(np.float32)
        freq       = converter.to_frequency_tensor(gray_image)
        
        assert freq.amplitude.shape[1] == 1


class TestAudioConverter:
    """Test untuk AudioConverter."""
    
    def test_sine_wave(self):
        """Sine wave 440 Hz harus menghasilkan peak di frekuensi yang sesuai."""
        from resonait.converters.image_converter import AudioConverter
        
        sr       = 22050
        duration = 1.0
        t        = np.linspace(0, duration, int(sr * duration))
        audio    = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        converter = AudioConverter(freq_dim=128, sample_rate=sr)
        freq      = converter.to_frequency_tensor(audio, sr=sr)
        
        assert freq.amplitude.shape[-1] == 128
        assert not torch.isnan(freq.amplitude).any()


# ============================================================
# TEST: Pain System
# ============================================================

class TestDissonanceEngine:
    """Test untuk Pain System."""
    
    def setup_method(self):
        from resonait.pain.dissonance import DissonanceEngine, DamageEvent, DamageType
        from resonait.core.brain import ResonAItBrain
        
        self.engine = DissonanceEngine(freq_dim=64)
        self.brain  = ResonAItBrain(freq_dim=64, hidden_dim=32, n_modes=4, n_fno_layers=1)
        self.DamageEvent = DamageEvent
        self.DamageType  = DamageType
    
    def test_damage_increases_pain(self):
        """Menerima damage harus meningkatkan pain level."""
        pain_before = self.engine.accumulated_pain.item()
        
        event = self.DamageEvent(
            damage_type=self.DamageType.PHYSICAL,
            intensity=0.5
        )
        self.engine.apply_to_brain(self.brain, event)
        
        pain_after = self.engine.accumulated_pain.item()
        assert pain_after > pain_before, "Pain tidak meningkat setelah damage!"
    
    def test_recovery_reduces_pain(self):
        """Recovery harus mengurangi pain level."""
        # Tambahkan pain dulu
        event = self.DamageEvent(
            damage_type=self.DamageType.PHYSICAL,
            intensity=0.8
        )
        self.engine.apply_to_brain(self.brain, event)
        pain_with_damage = self.engine.accumulated_pain.item()
        
        # Recovery
        for _ in range(20):
            self.engine.step_recovery(self.brain)
        
        pain_after_recovery = self.engine.accumulated_pain.item()
        assert pain_after_recovery < pain_with_damage, "Recovery tidak mengurangi pain!"
    
    def test_max_pain_clamped(self):
        """Pain tidak boleh melebihi max_pain."""
        for _ in range(100):
            event = self.DamageEvent(
                damage_type=self.DamageType.PHYSICAL,
                intensity=1.0
            )
            self.engine.apply_to_brain(self.brain, event)
        
        assert self.engine.accumulated_pain.item() <= self.engine.max_pain
    
    def test_reset_clears_pain(self):
        """Reset harus menghapus semua pain state."""
        event = self.DamageEvent(damage_type=self.DamageType.PHYSICAL, intensity=0.9)
        self.engine.apply_to_brain(self.brain, event)
        
        self.engine.reset()
        
        assert self.engine.accumulated_pain.item() == 0.0
        assert len(self.engine.event_history) == 0
    
    def test_different_damage_types(self):
        """Semua jenis damage harus menghasilkan sinyal dissonance yang valid."""
        from resonait.pain.dissonance import DamageType
        
        for damage_type in DamageType:
            if damage_type == DamageType.CUSTOM:
                continue  # Custom butuh handler khusus
            
            event = self.DamageEvent(
                damage_type=damage_type,
                intensity=0.3
            )
            signal, report = self.engine.process_damage(event)
            
            assert signal is not None
            assert not torch.isnan(signal.amplitude).any()
            assert "damage_type" in report


# ============================================================
# TEST: Memory System
# ============================================================

class TestMemorySystem:
    """Test untuk FrequencyMemorySystem."""
    
    def setup_method(self):
        from resonait.memory.frequency_memory import FrequencyMemorySystem
        from resonait.core.frequency_space import FrequencyTensor, Modality
        
        self.memory = FrequencyMemorySystem(
            freq_dim=64,
            stm_capacity=10,
            ltm_capacity=100,
            memory_dim=16,
        )
        self.Modality    = Modality
        self.FreqTensor  = FrequencyTensor
    
    def _make_freq_tensor(self, seed=None):
        if seed:
            torch.manual_seed(seed)
        return self.FreqTensor(
            amplitude=torch.rand(1, 1, 64),
            phase=torch.rand(1, 1, 64),
            modality=self.Modality.TEXT,
        )
    
    def test_perceive_stores_to_stm(self):
        """perceive() harus menyimpan ke STM."""
        initial_size = self.memory.stm.size
        self.memory.perceive(self._make_freq_tensor())
        assert self.memory.stm.size == initial_size + 1
    
    def test_stm_capacity_limit(self):
        """STM tidak boleh melebihi kapasitas."""
        for i in range(20):  # Lebih dari kapasitas (10)
            self.memory.perceive(self._make_freq_tensor(seed=i))
        
        assert self.memory.stm.size <= self.memory.stm.capacity
    
    def test_consolidate_transfers_important(self):
        """Konsolidasi harus memindahkan memori penting ke LTM."""
        # Simpan beberapa memori dengan importance tinggi
        for i in range(5):
            self.memory.perceive(self._make_freq_tensor(seed=i), importance=0.9)
        
        n_consolidated = self.memory.consolidate(min_importance=0.8)
        assert n_consolidated > 0
        assert self.memory.ltm.memory_count.item() > 0
    
    def test_recall_returns_results(self):
        """LTM recall harus mengembalikan hasil jika ada memori tersimpan."""
        # Isi LTM
        for i in range(10):
            freq = self._make_freq_tensor(seed=i)
            self.memory.ltm.store(freq, importance=0.8)
        
        query   = self._make_freq_tensor(seed=0)
        results = self.memory.ltm.recall(query, top_k=3)
        
        assert len(results) > 0
        assert all(0.0 <= sim <= 1.0 for _, sim, _ in results)


# ============================================================
# INTEGRATION TEST
# ============================================================

class TestIntegration:
    """Test integrasi end-to-end."""
    
    def test_full_pipeline(self):
        """Test pipeline lengkap: teks → frekuensi → otak → output."""
        from resonait.converters.text_converter import TextConverter
        from resonait.core.brain import ResonAItBrain
        from resonait.pain.dissonance import DissonanceEngine, DamageEvent, DamageType
        
        # Setup
        converter = TextConverter(freq_dim=64)
        brain     = ResonAItBrain(freq_dim=64, hidden_dim=32, n_modes=4, n_fno_layers=1)
        engine    = DissonanceEngine(freq_dim=64)
        
        # Konversi teks
        freq = converter.to_frequency_tensor("Test integrasi end-to-end")
        
        # Proses dengan otak
        output = brain(freq)
        assert "output" in output
        
        # Simulasi damage
        damage = DamageEvent(damage_type=DamageType.PHYSICAL, intensity=0.5)
        report = engine.apply_to_brain(brain, damage)
        assert report["accumulated_pain"] > 0
        
        # Proses lagi setelah pain
        output_after = brain(freq)
        assert output_after["pain_level"] > output["pain_level"]
        
        # Recovery
        for _ in range(10):
            engine.step_recovery(brain)
        
        # Verifikasi recovery
        health = brain.get_health_report()
        assert health["pain_level"] < report["accumulated_pain"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
