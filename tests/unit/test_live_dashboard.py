"""Tests for LiveDashboard Jupyter compatibility."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_tuner.reporting.live_dashboard import LiveDashboard, _is_jupyter


class TestIsJupyter:
    @pytest.mark.parametrize("env_var", ["COLAB_GPU", "JPY_PARENT_PID", "JPY_SESSION_NAME"])
    def test_returns_true_for_colab_subprocess_env(self, env_var):
        with patch.dict("os.environ", {env_var: "1"}, clear=True):
            assert _is_jupyter() is True

    def test_returns_false_without_ipython(self):
        with patch.dict("sys.modules", {"IPython": None}):
            # ImportError path
            assert _is_jupyter() is False

    def test_returns_false_when_no_active_shell(self):
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = None
        with patch.dict("sys.modules", {"IPython": mock_ipython}):
            assert _is_jupyter() is False

    def test_returns_false_for_terminal_ipython(self):
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "TerminalInteractiveShell"
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = mock_shell
        with patch.dict("sys.modules", {"IPython": mock_ipython}):
            assert _is_jupyter() is False

    def test_returns_true_for_jupyter_kernel(self):
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = mock_shell
        with patch.dict("sys.modules", {"IPython": mock_ipython}):
            assert _is_jupyter() is True


class TestLiveDashboardJupyterMode:
    def test_jupyter_mode_skips_live_display(self, sample_study_config):
        """In Jupyter mode, live_context should yield without creating a Live instance."""
        dash = LiveDashboard(sample_study_config)
        dash._use_live = False

        entered = False
        with dash.live_context():
            entered = True
            # No Live object should be created in Jupyter mode
            assert dash._live is None

        assert entered

    def test_non_jupyter_non_terminal_mode_skips_live(self, sample_study_config):
        """In non-Jupyter but non-terminal env, live mode should still be disabled."""
        with patch("vllm_tuner.reporting.live_dashboard._is_jupyter", return_value=False):
            dash = LiveDashboard(sample_study_config)
        assert dash._use_live is False

    def test_jupyter_mode_skips_live(self, sample_study_config):
        """In Jupyter mode live mode is disabled."""
        with patch("vllm_tuner.reporting.live_dashboard._is_jupyter", return_value=True):
            dash = LiveDashboard(sample_study_config)
        assert dash._use_live is False

    def test_non_jupyter_terminal_mode_enables_live(self, sample_study_config):
        """In non-Jupyter terminal mode, use Live dashboard rendering."""
        mock_console = MagicMock()
        mock_console.is_terminal = True
        with patch("vllm_tuner.reporting.live_dashboard._is_jupyter", return_value=False), patch(
            "vllm_tuner.reporting.live_dashboard.Console", return_value=mock_console
        ):
            dash = LiveDashboard(sample_study_config)
        assert dash._use_live is True

    def test_jupyter_event_callbacks_do_not_crash(
        self, sample_study_config, sample_benchmark_result, sample_trial_result
    ):
        """All event callbacks should run without errors in Jupyter mode."""
        dash = LiveDashboard(sample_study_config)
        dash._use_live = False

        with dash.live_context():
            dash.on_study_start()
            dash.on_baseline_start()
            dash.on_baseline_complete(sample_benchmark_result)
            dash.on_trial_start(1, {"gpu_memory_utilization": 0.9})
            dash.on_server_starting("vllm serve ...")
            dash.on_server_log("INFO: server starting")
            dash.on_server_ready()
            dash.on_benchmark_start(1)
            dash.on_trial_complete(sample_trial_result)
            dash.on_study_complete()
