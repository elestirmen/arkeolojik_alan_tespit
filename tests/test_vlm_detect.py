from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import vlm_detect
import vlm_lmstudio_detector as vlm_lmstudio


def _write_test_raster(path: Path, *, count: int) -> None:
    data = np.ones((count, 8, 8), dtype=np.uint8) * 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=count,
        dtype="uint8",
        transform=from_origin(0, 8, 1, 1),
    ) as dst:
        dst.write(data)


def test_auto_bands_reuse_single_band_as_rgb(tmp_path: Path):
    tif_path = tmp_path / "hillshade.tif"
    _write_test_raster(tif_path, count=1)

    assert vlm_detect.resolve_vlm_band_indexes(tif_path, "auto") == (1, 1, 1)


def test_auto_bands_delegate_three_band_raster_to_vlm_auto(tmp_path: Path):
    tif_path = tmp_path / "rgb.tif"
    _write_test_raster(tif_path, count=3)

    assert vlm_detect.resolve_vlm_band_indexes(tif_path, "auto") is None


def test_legacy_vlm_prefixed_config_keys_are_accepted(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'input: "data/sample.tif"',
                'bands: "auto"',
                'vlm_base_url: "http://127.0.0.1:8081"',
                'vlm_tile: 777',
                'vlm_overlap: 111',
                'vlm_source_kind: "hillshade"',
                'vlm_prompt_stage1_path: "prompts/stage1.txt"',
                'vlm_prompt_stage2_path: "prompts/stage2.txt"',
                'vlm_resume: false',
                'enable_vlm: true',
                'enable_deep_learning: false',
            ]
        ),
        encoding="utf-8",
    )

    args = vlm_detect.build_arg_parser().parse_args(["--config", str(cfg_path)])
    config = vlm_detect.build_config_from_args(args)

    assert config.input == str((tmp_path / "data" / "sample.tif").resolve(strict=False))
    assert config.base_url == "http://127.0.0.1:8081"
    assert config.tile == 777
    assert config.overlap == 111
    assert config.source_kind == "hillshade"
    assert config.prompt_stage1_path == str((tmp_path / "prompts" / "stage1.txt").resolve(strict=False))
    assert config.prompt_stage2_path == str((tmp_path / "prompts" / "stage2.txt").resolve(strict=False))
    assert config.resume is False


def test_cli_overrides_vlm_config_values(tmp_path: Path):
    cfg_path = tmp_path / "config_vlm.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'input: "data/sample.tif"',
                "tile: 512",
                "overlap: 128",
                'source_kind: "rgb"',
                'prompt_stage1_path: "prompts/from_yaml_stage1.txt"',
                'prompt_stage2_path: "prompts/from_yaml_stage2.txt"',
            ]
        ),
        encoding="utf-8",
    )

    args = vlm_detect.build_arg_parser().parse_args(
        [
            "--config",
            str(cfg_path),
            "--tile",
            "1024",
            "--overlap",
            "256",
            "--source-kind",
            "hillshade",
            "--prompt-stage1-path",
            "cli_stage1.txt",
            "--prompt-stage2-path",
            "cli_stage2.txt",
        ]
    )
    config = vlm_detect.build_config_from_args(args)

    assert config.tile == 1024
    assert config.overlap == 256
    assert config.source_kind == "hillshade"
    assert config.prompt_stage1_path == str((Path.cwd() / "cli_stage1.txt").resolve(strict=False))
    assert config.prompt_stage2_path == str((Path.cwd() / "cli_stage2.txt").resolve(strict=False))


def test_resume_records_require_matching_prompt_fingerprint(tmp_path: Path):
    resume_path = tmp_path / "resume.jsonl"
    prompt_fingerprint = vlm_lmstudio._prompt_fingerprint("stage 1 prompt", "stage 2 prompt")
    changed_prompt_fingerprint = vlm_lmstudio._prompt_fingerprint("changed stage 1 prompt", "stage 2 prompt")
    resume_path.write_text(
        (
            '{"tile_index":1,"tile_row":0,"tile_col":0,"tile_width":8,"tile_height":8,'
            '"used_views":["rgb"],"analysis_mode":"rgb_only","source_kind":"hillshade",'
            f'"prompt_fingerprint":"{prompt_fingerprint}"}}\n'
        ),
        encoding="utf-8",
    )

    records, processed = vlm_lmstudio._load_resume_records(
        resume_path,
        total_tiles=1,
        raster_width=8,
        raster_height=8,
        tile=8,
        overlap=0,
        selected_views=["rgb"],
        analysis_mode="rgb_only",
        source_kind="hillshade",
        prompt_fingerprint=prompt_fingerprint,
        logger=vlm_lmstudio.LOGGER,
    )
    assert [record["tile_index"] for record in records] == [1]
    assert processed == {1}

    records, processed = vlm_lmstudio._load_resume_records(
        resume_path,
        total_tiles=1,
        raster_width=8,
        raster_height=8,
        tile=8,
        overlap=0,
        selected_views=["rgb"],
        analysis_mode="rgb_only",
        source_kind="hillshade",
        prompt_fingerprint=changed_prompt_fingerprint,
        logger=vlm_lmstudio.LOGGER,
    )
    assert records == []
    assert processed == set()
