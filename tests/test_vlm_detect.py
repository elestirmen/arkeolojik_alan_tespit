import os
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


def test_auto_bands_accept_directory_input(tmp_path: Path):
    _write_test_raster(tmp_path / "rgb.tif", count=3)
    _write_test_raster(tmp_path / "hillshade.tif", count=1)

    assert vlm_detect.resolve_vlm_band_indexes(tmp_path, "auto") is None
    assert vlm_detect.resolve_vlm_band_indexes(tmp_path, "1,2,3") == (1, 2, 3)


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
                "vlm_max_candidates_per_tile: 5",
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
    assert config.max_candidates_per_tile == 5
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
                "max_candidates_per_tile: 2",
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
            "--max-candidates-per-tile",
            "4",
        ]
    )
    config = vlm_detect.build_config_from_args(args)

    assert config.tile == 1024
    assert config.overlap == 256
    assert config.source_kind == "hillshade"
    assert config.max_candidates_per_tile == 4
    assert config.prompt_stage1_path == str((Path.cwd() / "cli_stage1.txt").resolve(strict=False))
    assert config.prompt_stage2_path == str((Path.cwd() / "cli_stage2.txt").resolve(strict=False))


def test_batch_folder_name_uses_input_folder_stem(monkeypatch):
    monkeypatch.setattr(vlm_detect, "SESSION_RUN_ID", "20260526_120000")

    assert (
        vlm_detect.build_vlm_batch_folder_name(Path("E:/maps/cappadocia set"))
        == "20260526_120000_cappadocia_set_merged_batch"
    )


def test_find_latest_vlm_batch_folder_uses_matching_resume_jsonl(tmp_path: Path):
    input_dir = tmp_path / "kapadokya"
    input_dir.mkdir()
    output_root = tmp_path / "out"
    older = output_root / "20260523_212418_kapadokya_merged_batch"
    newer = output_root / "20260525_081500_kapadokya_merged_batch"
    other_input = output_root / "20260525_081500_other_merged_batch"
    empty = output_root / "20260526_081500_kapadokya_merged_batch"
    for folder in (older, newer, other_input, empty):
        folder.mkdir(parents=True)
    older_jsonl = older / "kapadokya_bing_map_1-4_vlm_candidates.jsonl"
    newer_jsonl = newer / "kapadokya_bing_map_1-4_vlm_candidates.jsonl"
    other_jsonl = other_input / "tile_vlm_candidates.jsonl"
    empty_jsonl = empty / "kapadokya_bing_map_1-4_vlm_candidates.jsonl"
    older_jsonl.write_text('{"tile_index":1}\n', encoding="utf-8")
    newer_jsonl.write_text('{"tile_index":2}\n', encoding="utf-8")
    other_jsonl.write_text('{"tile_index":1}\n', encoding="utf-8")
    empty_jsonl.write_text("", encoding="utf-8")
    os.utime(older_jsonl, (100.0, 100.0))
    os.utime(newer_jsonl, (200.0, 200.0))
    os.utime(other_jsonl, (300.0, 300.0))
    os.utime(empty_jsonl, (400.0, 400.0))
    os.utime(older, (100.0, 100.0))
    os.utime(newer, (100.0, 100.0))
    os.utime(other_input, (300.0, 300.0))
    os.utime(empty, (400.0, 400.0))

    assert vlm_detect.find_latest_vlm_batch_folder(input_dir, output_root) == newer


def test_resolve_batch_output_folder_reuses_latest_when_resume_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vlm_detect, "SESSION_RUN_ID", "20260526_120000")
    input_dir = tmp_path / "kapadokya"
    input_dir.mkdir()
    output_root = tmp_path / "out"
    resume_folder = output_root / "20260523_212418_kapadokya_merged_batch"
    resume_folder.mkdir(parents=True)
    (resume_folder / "kapadokya_bing_map_1-4_vlm_candidates.jsonl").write_text('{"tile_index":1}\n', encoding="utf-8")
    config = vlm_detect.StandaloneVlmConfig(
        input=str(input_dir),
        output_root=str(output_root),
        resume=True,
        batch=True,
    )

    assert vlm_detect.resolve_batch_output_folder(input_dir, config) == resume_folder

    fresh_config = vlm_detect.StandaloneVlmConfig(
        input=str(input_dir),
        output_root=str(output_root),
        resume=False,
        batch=True,
    )
    assert (
        vlm_detect.resolve_batch_output_folder(input_dir, fresh_config)
        == output_root / "20260526_120000_kapadokya_merged_batch"
    )


def test_run_info_txt_includes_parameters_and_prompt_text(tmp_path: Path):
    stage1_path = tmp_path / "stage1.txt"
    stage2_path = tmp_path / "stage2.txt"
    stage1_path.write_text("stage one prompt", encoding="utf-8")
    stage2_path.write_text("stage two prompt", encoding="utf-8")

    config = vlm_detect.StandaloneVlmConfig(
        input=str(tmp_path / "input.tif"),
        api_key="secret-key",
        tile=8,
        overlap=0,
        prompt_stage1_path=str(stage1_path),
        prompt_stage2_path=str(stage2_path),
    )

    info_path = vlm_detect.write_run_info_txt(tmp_path / "out" / "rgb", config, (1, 2, 3))
    text = info_path.read_text(encoding="utf-8")

    assert info_path.name == "rgb_vlm_parameters_prompts.txt"
    assert "tile: 8" in text
    assert "overlap: 0" in text
    assert "resolved_band_indexes: [1, 2, 3]" in text
    assert "api_key: <redacted>" in text
    assert "secret-key" not in text
    assert f"source: {stage1_path}" in text
    assert "stage one prompt" in text
    assert f"source: {stage2_path}" in text
    assert "stage two prompt" in text


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


def test_resume_invalid_json_errors_are_retried(tmp_path: Path):
    resume_path = tmp_path / "resume.jsonl"
    prompt_fingerprint = vlm_lmstudio._prompt_fingerprint(None, None)
    resume_path.write_text(
        (
            '{"tile_index":1,"tile_row":0,"tile_col":0,"tile_width":8,"tile_height":8,'
            '"used_views":["rgb"],"analysis_mode":"rgb_only","source_kind":"rgb",'
            f'"prompt_fingerprint":"{prompt_fingerprint}",'
            '"status":"error","error_type":"invalid_json"}\n'
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
        source_kind="rgb",
        prompt_fingerprint=prompt_fingerprint,
        logger=vlm_lmstudio.LOGGER,
    )

    assert records == []
    assert processed == set()
