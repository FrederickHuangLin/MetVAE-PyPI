"""Tests for the MetVAE command-line interface."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import metvae
from metvae import cli


class _StopRun(Exception):
    """Sentinel raised by the recording stub to end the command early."""


class _RecordingMetVAE:
    """Stub that records the constructor arguments and stops the pipeline."""

    calls = []

    def __init__(self, **kwargs):
        _RecordingMetVAE.calls.append(kwargs)
        raise _StopRun()


def _repo_root():
    """Return the directory that contains the metvae package."""
    return Path(metvae.__file__).resolve().parent.parent


@pytest.mark.parametrize("sparse_args, expected", [
    (['--sparse_method', 'pval'],
     ['model_state.pth', 'df_corr.csv', 'p_values.csv', 'q_values.csv', 'df_sparse_pval.csv']),
    (['--sparse_method', 'sec', '--rho', '1.0'],
     ['model_state.pth', 'df_corr.csv', 'df_sparse_sec.csv']),
])
def test_cli_smoke(tmp_path, monkeypatch, test_files, sparse_args, expected):
    """A short run writes its outputs under --save_path and leaves the working directory clean."""
    data_path, meta_path = test_files
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    argv = [
        '--data', str(data_path),
        '--meta', str(meta_path),
        '--continuous_covariate_keys', 'x1',
        '--categorical_covariate_keys', 'x2',
        '--save_path', str(out_dir),
        '--latent_dim', '8',
        '--batch_size', '100',
        '--max_epochs', '3',
        '--num_sim', '3',
        '--workers', '1',
        '--sec_max_iter', '50',
    ] + sparse_args

    cli.main(argv)

    for name in expected:
        assert (out_dir / name).exists(), f"{name} was not written to the save path"

    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize("threshold_args", [
    ['--feature_zero_threshold', 'none'],
    ['--no_feature_filter'],
])
def test_cli_none_feature_threshold(tmp_path, monkeypatch, test_files, threshold_args):
    """Both ways of disabling the feature filter pass feature_zero_threshold=None."""
    data_path, _ = test_files
    monkeypatch.setattr(cli, 'MetVAE', _RecordingMetVAE)
    _RecordingMetVAE.calls = []

    argv = ['--data', str(data_path), '--save_path', str(tmp_path)] + threshold_args
    with pytest.raises(_StopRun):
        cli.main(argv)

    assert len(_RecordingMetVAE.calls) == 1
    assert _RecordingMetVAE.calls[0]['feature_zero_threshold'] is None


def test_cli_covariate_keys_require_meta(tmp_path, test_files):
    """Covariate keys without --meta end the command with a parser error."""
    data_path, _ = test_files
    argv = ['--data', str(data_path), '--save_path', str(tmp_path),
            '--continuous_covariate_keys', 'x1']
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_cli_missing_file_errors(tmp_path):
    """A missing input file makes the command exit with a non-zero status."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(_repo_root())] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'metvae.cli',
         '--data', str(tmp_path / 'nonexistent.csv'),
         '--save_path', str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert proc.returncode != 0
