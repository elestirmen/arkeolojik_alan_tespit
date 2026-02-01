"""
Arkeolojik Alan Tespiti - Unit Testler

Bu dosya archaeo_detect.py modülünün temel fonksiyonlarını test eder.
Testleri çalıştırmak için: pytest tests/ -v
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

# Proje kök dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from archaeo_detect import (
    PipelineDefaults,
    build_config_from_args,
    percentile_clip,
    compute_ndsm,
    _otsu_threshold_0to1,
    robust_norm,
    robust_norm_fixed,
    fill_nodata,
    _local_variance,
    _hessian_response,
    _norm01,
    make_feather_weights,
    generate_windows,
    build_filename_with_params,
    compute_fused_probability,
)


# ============================================================================
# PipelineDefaults Validation Testleri
# ============================================================================

class TestPipelineDefaultsValidation:
    """PipelineDefaults __post_init__ doğrulama testleri."""

    def test_valid_defaults(self):
        """Varsayılan değerler geçerli olmalı."""
        config = PipelineDefaults()
        assert config.th == 0.6
        assert config.tile == 1024

    def test_invalid_th_above_1(self):
        """th > 1.0 hata vermeli."""
        with pytest.raises(ValueError, match="th değeri 0-1 arasında olmalı"):
            PipelineDefaults(th=1.5)

    def test_invalid_th_below_0(self):
        """th < 0.0 hata vermeli."""
        with pytest.raises(ValueError, match="th değeri 0-1 arasında olmalı"):
            PipelineDefaults(th=-0.1)

    def test_invalid_alpha(self):
        """alpha 0-1 dışında hata vermeli."""
        with pytest.raises(ValueError, match="alpha değeri 0-1 arasında olmalı"):
            PipelineDefaults(alpha=2.0)

    def test_invalid_tile_negative(self):
        """Negatif tile hata vermeli."""
        with pytest.raises(ValueError, match="tile pozitif olmalı"):
            PipelineDefaults(tile=-100)

    def test_invalid_overlap_greater_than_tile(self):
        """overlap >= tile hata vermeli."""
        with pytest.raises(ValueError, match="overlap.*tile.*küçük olmalı"):
            PipelineDefaults(tile=512, overlap=600)

    def test_invalid_percentiles(self):
        """percentile_low >= percentile_high hata vermeli."""
        with pytest.raises(ValueError, match="percentile_low.*percentile_high.*küçük olmalı"):
            PipelineDefaults(percentile_low=99.0, percentile_high=1.0)

    def test_invalid_connectivity(self):
        """label_connectivity 4 veya 8 dışında hata vermeli."""
        with pytest.raises(ValueError, match="label_connectivity 4 veya 8 olmalı"):
            PipelineDefaults(label_connectivity=6)

    def test_invalid_yolo_conf(self):
        """yolo_conf 0-1 dışında hata vermeli."""
        with pytest.raises(ValueError, match="yolo_conf değeri 0-1 arasında olmalı"):
            PipelineDefaults(yolo_conf=1.5)

    def test_invalid_verbose(self):
        """verbose 0-2 dışında hata vermeli."""
        with pytest.raises(ValueError, match="verbose 0-2 arasında olmalı"):
            PipelineDefaults(verbose=5)

    def test_empty_sigma_scales(self):
        """Boş sigma_scales hata vermeli."""
        with pytest.raises(ValueError, match="sigma_scales en az bir değer içermeli"):
            PipelineDefaults(sigma_scales=())

    def test_valid_device_cuda(self):
        """'cuda' geçerli bir device olmalı."""
        config = PipelineDefaults(device="cuda")
        assert config.device == "cuda"

    def test_valid_device_cpu(self):
        """'cpu' geçerli bir device olmalı."""
        config = PipelineDefaults(device="cpu")
        assert config.device == "cpu"

    def test_valid_device_cuda_index(self):
        """'cuda:0' geçerli bir device olmalı."""
        config = PipelineDefaults(device="cuda:0")
        assert config.device == "cuda:0"


# ============================================================================
# Normalizasyon ve Veri İşleme Testleri
# ============================================================================

class TestNormalizationFunctions:
    """Normalizasyon fonksiyonları testleri."""

    def test_percentile_clip_basic(self):
        """Temel percentile_clip işlemi."""
        arr = np.array([0, 10, 50, 90, 100], dtype=np.float32)
        result = percentile_clip(arr, low=2.0, high=98.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_percentile_clip_with_nan(self):
        """NaN değerli array ile percentile_clip."""
        arr = np.array([0, np.nan, 50, np.nan, 100], dtype=np.float32)
        result = percentile_clip(arr, low=2.0, high=98.0)
        # NaN'lar 0 olmalı
        assert result[1] == 0.0
        assert result[3] == 0.0
        # Geçerli değerler 0-1 arasında
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[2] <= 1.0
        assert 0.0 <= result[4] <= 1.0

    def test_percentile_clip_empty_array(self):
        """Boş array için percentile_clip."""
        arr = np.array([], dtype=np.float32)
        result = percentile_clip(arr)
        assert len(result) == 0

    def test_percentile_clip_all_nan(self):
        """Tamamen NaN array için percentile_clip."""
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        result = percentile_clip(arr)
        assert np.allclose(result, 0.0)

    def test_robust_norm_shape(self):
        """robust_norm çıktı şekli doğru olmalı."""
        arr = np.random.rand(9, 100, 100).astype(np.float32)
        result = robust_norm(arr)
        assert result.shape == arr.shape
        assert result.dtype == np.float32

    def test_robust_norm_range(self):
        """robust_norm değerleri 0-1 arasında olmalı."""
        arr = np.random.rand(3, 50, 50).astype(np.float32) * 255
        result = robust_norm(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_robust_norm_fixed_with_given_bounds(self):
        """robust_norm_fixed verilen sınırlarla çalışmalı."""
        arr = np.array([[[0, 50, 100]], [[10, 60, 110]], [[20, 70, 120]]], dtype=np.float32)
        lows = np.array([0, 10, 20], dtype=np.float32)
        highs = np.array([100, 110, 120], dtype=np.float32)
        result = robust_norm_fixed(arr, lows, highs)
        assert result.shape == arr.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ============================================================================
# nDSM Hesaplama Testleri
# ============================================================================

class TestComputeNDSM:
    """compute_ndsm fonksiyonu testleri."""

    def test_ndsm_basic(self):
        """Temel nDSM hesaplama."""
        dsm = np.array([[20, 30], [40, 50]], dtype=np.float32)
        dtm = np.array([[10, 20], [30, 40]], dtype=np.float32)
        ndsm = compute_ndsm(dsm, dtm)
        expected = np.array([[10, 10], [10, 10]], dtype=np.float32)
        assert np.allclose(ndsm, expected)

    def test_ndsm_with_none_dsm(self):
        """DSM None ise nDSM sıfır olmalı."""
        dtm = np.array([[10, 20], [30, 40]], dtype=np.float32)
        ndsm = compute_ndsm(None, dtm)
        assert np.allclose(ndsm[np.isfinite(ndsm)], 0.0)

    def test_ndsm_with_nan(self):
        """NaN değerler korunmalı."""
        dsm = np.array([[20, np.nan], [40, 50]], dtype=np.float32)
        dtm = np.array([[10, 20], [np.nan, 40]], dtype=np.float32)
        ndsm = compute_ndsm(dsm, dtm)
        # Sadece her iki değer de geçerli olduğunda hesaplanmalı
        assert np.isfinite(ndsm[0, 0])  # Her ikisi de geçerli
        assert np.isnan(ndsm[0, 1])     # DSM nan
        assert np.isnan(ndsm[1, 0])     # DTM nan
        assert np.isfinite(ndsm[1, 1])  # Her ikisi de geçerli


# ============================================================================
# Otsu Eşikleme Testleri
# ============================================================================

class TestOtsuThreshold:
    """Otsu eşikleme testleri."""

    def test_otsu_bimodal(self):
        """Bimodal dağılım için Otsu eşiği."""
        # İki farklı yoğunluk grubu
        low_values = np.ones(50) * 0.2
        high_values = np.ones(50) * 0.8
        arr = np.concatenate([low_values, high_values])
        valid = np.ones(100, dtype=bool)
        threshold = _otsu_threshold_0to1(arr, valid)
        # Eşik iki grup arasında olmalı
        assert 0.3 < threshold < 0.7

    def test_otsu_uniform(self):
        """Uniform dağılım için Otsu eşiği."""
        arr = np.linspace(0, 1, 100).astype(np.float32)
        valid = np.ones(100, dtype=bool)
        threshold = _otsu_threshold_0to1(arr, valid)
        assert 0.0 <= threshold <= 1.0

    def test_otsu_empty_valid(self):
        """Boş geçerli maske için varsayılan eşik."""
        arr = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        valid = np.zeros(3, dtype=bool)
        threshold = _otsu_threshold_0to1(arr, valid)
        assert threshold == 0.5  # Varsayılan


# ============================================================================
# Fill NoData Testleri
# ============================================================================

class TestFillNoData:
    """fill_nodata fonksiyonu testleri."""

    def test_fill_nodata_basic(self):
        """Temel NaN doldurma."""
        arr = np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)
        filled, valid_mask = fill_nodata(arr, fill_value=0.0)
        assert filled[1] == 0.0
        assert filled[3] == 0.0
        assert np.array_equal(valid_mask, [True, False, True, False, True])

    def test_fill_nodata_median(self):
        """Medyan ile NaN doldurma."""
        arr = np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32)
        filled, _ = fill_nodata(arr, fill_value=None)
        # Medyan 3.0 olmalı (1, 3, 5'in medyanı)
        assert filled[1] == 3.0
        assert filled[3] == 3.0

    def test_fill_nodata_no_nan(self):
        """NaN olmayan array için."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        filled, valid_mask = fill_nodata(arr)
        assert np.array_equal(filled, arr)
        assert np.all(valid_mask)


# ============================================================================
# Yerel Varyans Testleri
# ============================================================================

class TestLocalVariance:
    """_local_variance fonksiyonu testleri."""

    def test_local_variance_uniform(self):
        """Uniform array için varyans sıfır olmalı."""
        arr = np.ones((10, 10), dtype=np.float32) * 5.0
        var = _local_variance(arr, size=3)
        assert np.allclose(var, 0.0, atol=1e-5)

    def test_local_variance_varying(self):
        """Değişen array için varyans pozitif olmalı."""
        arr = np.random.rand(10, 10).astype(np.float32)
        var = _local_variance(arr, size=3)
        assert var.mean() > 0.0

    def test_local_variance_size_1(self):
        """Size 1 için sıfır döndürmeli."""
        arr = np.random.rand(10, 10).astype(np.float32)
        var = _local_variance(arr, size=1)
        assert np.allclose(var, 0.0)


# ============================================================================
# Hessian Response Testleri
# ============================================================================

class TestHessianResponse:
    """_hessian_response fonksiyonu testleri."""

    def test_hessian_response_shape(self):
        """Hessian response çıktı şekli doğru olmalı."""
        arr = np.random.rand(50, 50).astype(np.float32)
        response = _hessian_response(arr, sigma=2.0)
        assert response.shape == arr.shape

    def test_hessian_response_range(self):
        """Hessian response 0-1 arasında olmalı (normalize edilmiş)."""
        arr = np.random.rand(50, 50).astype(np.float32)
        response = _hessian_response(arr, sigma=2.0)
        assert response.min() >= 0.0
        assert response.max() <= 1.0


# ============================================================================
# Feather Weights Testleri
# ============================================================================

class TestFeatherWeights:
    """make_feather_weights fonksiyonu testleri."""

    def test_feather_weights_shape(self):
        """Feather weights doğru şekilde olmalı."""
        weights = make_feather_weights(h=100, w=100, tile=1024, overlap=256)
        assert weights.shape == (100, 100)

    def test_feather_weights_center_high(self):
        """Merkezdeki ağırlıklar yüksek olmalı."""
        weights = make_feather_weights(h=100, w=100, tile=1024, overlap=50)
        center = weights[50, 50]
        corner = weights[0, 0]
        assert center >= corner

    def test_feather_weights_no_overlap(self):
        """Overlap 0 ise tüm ağırlıklar 1 olmalı."""
        weights = make_feather_weights(h=100, w=100, tile=1024, overlap=0)
        assert np.allclose(weights, 1.0)


# ============================================================================
# Generate Windows Testleri
# ============================================================================

class TestGenerateWindows:
    """generate_windows fonksiyonu testleri."""

    def test_generate_windows_basic(self):
        """Temel pencere üretimi."""
        windows = list(generate_windows(width=100, height=100, tile=50, overlap=10))
        assert len(windows) > 0
        for win, row, col in windows:
            assert win.width > 0
            assert win.height > 0

    def test_generate_windows_overlap_error(self):
        """overlap >= tile hata vermeli."""
        with pytest.raises(ValueError):
            list(generate_windows(width=100, height=100, tile=50, overlap=60))

    def test_generate_windows_coverage(self):
        """Tüm alan kapsanmalı."""
        width, height = 100, 100
        tile, overlap = 40, 10
        windows = list(generate_windows(width, height, tile, overlap))
        
        covered = np.zeros((height, width), dtype=bool)
        for win, row, col in windows:
            r_end = min(row + int(win.height), height)
            c_end = min(col + int(win.width), width)
            covered[row:r_end, col:c_end] = True
        
        assert np.all(covered)


# ============================================================================
# Build Filename Testleri
# ============================================================================

class TestBuildFilename:
    """build_filename_with_params fonksiyonu testleri."""

    def test_build_filename_basic(self):
        """Temel dosya adı oluşturma."""
        name = build_filename_with_params(
            base_name="test",
            encoder="resnet34",
            threshold=0.5,
            tile=1024,
        )
        assert "test" in name
        assert "resnet34" in name
        assert "th0.5" in name
        assert "tile1024" in name

    def test_build_filename_with_alpha(self):
        """Alpha parametreli dosya adı."""
        name = build_filename_with_params(
            base_name="output",
            mode_suffix="fused",
            alpha=0.7,
        )
        assert "fused" in name
        assert "alpha0.7" in name

    def test_build_filename_minimal(self):
        """Sadece base_name ile dosya adı."""
        name = build_filename_with_params(base_name="minimal")
        assert name == "minimal"


# ============================================================================
# Fusion Probability Testleri
# ============================================================================

class TestFusionProbability:
    """compute_fused_probability fonksiyonu testleri."""

    def test_fusion_equal_weight(self):
        """alpha=0.5 ile eşit ağırlıklı fusion."""
        dl_prob = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float32)
        classic_prob = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        fused, mask = compute_fused_probability(dl_prob, classic_prob, alpha=0.5, threshold=0.5)
        
        expected_fused = (dl_prob + classic_prob) / 2
        assert np.allclose(fused, expected_fused)

    def test_fusion_dl_only(self):
        """alpha=1.0 ile sadece DL."""
        dl_prob = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float32)
        classic_prob = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        fused, _ = compute_fused_probability(dl_prob, classic_prob, alpha=1.0, threshold=0.5)
        
        assert np.allclose(fused, dl_prob)

    def test_fusion_classic_only(self):
        """alpha=0.0 ile sadece klasik."""
        dl_prob = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float32)
        classic_prob = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        fused, _ = compute_fused_probability(dl_prob, classic_prob, alpha=0.0, threshold=0.5)
        
        assert np.allclose(fused, classic_prob)

    def test_fusion_mask_threshold(self):
        """Eşikleme doğru çalışmalı."""
        dl_prob = np.array([[0.8, 0.3], [0.3, 0.8]], dtype=np.float32)
        classic_prob = np.array([[0.8, 0.3], [0.3, 0.8]], dtype=np.float32)
        _, mask = compute_fused_probability(dl_prob, classic_prob, alpha=0.5, threshold=0.5)
        
        expected_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        assert np.array_equal(mask, expected_mask)

    def test_fusion_with_nan(self):
        """NaN değerler ile fusion."""
        dl_prob = np.array([[0.8, np.nan], [0.4, 0.2]], dtype=np.float32)
        classic_prob = np.array([[0.2, 0.4], [np.nan, 0.8]], dtype=np.float32)
        fused, mask = compute_fused_probability(dl_prob, classic_prob, alpha=0.5, threshold=0.5)
        
        # Her iki değer de NaN olan yerlerde sonuç NaN olmalı
        # En az birinin geçerli olduğu yerlerde hesaplanmalı
        assert np.isfinite(fused[0, 0])


# ============================================================================
# Config/YAML Override Testleri
# ============================================================================

class TestConfigOverride:
    def test_cli_overrides_yaml_even_when_default_values(self, tmp_path: Path):
        """CLI override, YAML'daki ayarları default değerle de olsa ezebilmeli."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "enable_yolo: true",
                    "cache_derivatives: true",
                    "yolo_conf: 0.7",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        defaults = PipelineDefaults()
        args = argparse.Namespace(
            config=str(yaml_path),
            enable_yolo=False,
            cache_derivatives=False,
            yolo_conf=defaults.yolo_conf,  # default değere geri dön
        )
        config = build_config_from_args(
            args, cli_overrides={"enable_yolo", "cache_derivatives", "yolo_conf"}
        )

        assert config.enable_yolo is False
        assert config.cache_derivatives is False
        assert config.yolo_conf == pytest.approx(defaults.yolo_conf)


# ============================================================================
# Ana Test Çalıştırıcı
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

