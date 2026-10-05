"""Offline preflight and PowerShell orchestration checks. Docker is always fake."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.check_reranker_ollama import ENDPOINT, check_ollama, main

ROOT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def test_inventory_only_uses_internal_endpoint(self):
        for name in ("bge-m3", "bge-m3:latest"):
            opener = Mock()
            opener.open.return_value = io.BytesIO(json.dumps({"models": [{"name": name}]}).encode())
            self.assertEqual(check_ollama(opener=opener), 0)
            opener.open.assert_called_once_with(ENDPOINT, timeout=5)
            self.assertEqual(ENDPOINT, "http://ollama:11434/api/tags")

    def test_missing_default_model_does_not_pull(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"models":[{"name":"bge-m3:other"}]}')
        self.assertEqual(check_ollama(opener=opener), 2)
        self.assertEqual(opener.open.call_count, 1)

    def test_unavailable_retry_and_safe_error(self):
        opener = Mock(open=Mock(side_effect=OSError("PRIVATE_PATH")))
        pause = Mock()
        self.assertEqual(check_ollama(opener=opener, attempts=3, pause=pause), 3)
        self.assertEqual(pause.call_count, 2)
        with patch("scripts.check_reranker_ollama.check_ollama", return_value=3), patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(), 3)
            self.assertNotIn("PRIVATE", out.getvalue())

    def test_malformed_inventory_is_not_ready(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"PRIVATE":"invalid schema"}')
        self.assertEqual(check_ollama(opener=opener, attempts=1), 3)

    def test_compose_endpoint_and_persistent_volume(self):
        overlay = (ROOT / "compose.reranker.yaml").read_text(encoding="utf-8")
        self.assertIn("OLLAMA_BASE_URL: http://ollama:11434", overlay)
        self.assertIn("reranker-hf-cache:/cache/huggingface", overlay)
        self.assertIn("HF_HOME: /cache/huggingface", overlay)
        self.assertNotIn("host.docker.internal", overlay)
        self.assertNotRegex(overlay, r"(?m)^\s+ports:")


@unittest.skipUnless(shutil.which("powershell"), "PowerShell unavailable")
class RunnerTests(unittest.TestCase):
    def run_fake(self, *, running=True, preflight=0, daemon=0, device="auto"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "data/benchmark").mkdir(parents=True)
            for name in ("corpus", "gold"):
                (root / f"data/benchmark/{name}.json").write_text("{}")
            shutil.copy(ROOT / "scripts/run_local_reranker_benchmark.ps1", root / "scripts")
            harness = r'''
$calls = [System.Collections.Generic.List[object]]::new()
function git {
    $global:LASTEXITCODE = 0
    if ($args[0] -eq 'rev-parse') { 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' }
}
function docker {
    $calls.Add(@($args))
    $global:LASTEXITCODE = 0
    $command = $args -join ' '
    if ($args[0] -eq 'info') { $global:LASTEXITCODE = DAEMON; return }
    if ($args[0] -eq 'ps') { 'existing-custom-project'; return }
    if ($command -match 'ps --status') { if (RUNNING) { 'fake-ollama-id' }; return }
    if ($command -match 'check_reranker_ollama.py') { $global:LASTEXITCODE = PREFLIGHT; return }
    if ($command -match 'eval_retrieval.py') { 'Report: reports/20261005T010101Z-aaaaaaaaaaaa/REPORT.md'; return }
}
$outcome = ''
try { $output = @(& .\scripts\run_local_reranker_benchmark.ps1 -Device 'DEVICE'); $outcome = 'ok' }
catch { $outcome = $_.Exception.Message; $output = @() }
@{calls=@($calls.ToArray());outcome=$outcome;output=$output;restored=($null -eq $env:BENCHMARK_CODE_COMMIT_SHA)} | ConvertTo-Json -Depth 6 -Compress
'''.replace("DAEMON", str(daemon)).replace("RUNNING", "$true" if running else "$false").replace("PREFLIGHT", str(preflight)).replace("DEVICE", device)
            script = root / "harness.ps1"
            script.write_text(harness, encoding="utf-8")
            environment = {key: value for key, value in os.environ.items() if key not in (
                "COMPOSE_PROJECT_NAME", "BENCHMARK_CODE_COMMIT_SHA", "BENCHMARK_CODE_DIRTY")}
            result = subprocess.run([shutil.which("powershell"), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                                    cwd=root, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_one_command_running_project_no_restart_and_comparison(self):
        result = self.run_fake()
        self.assertEqual(result["outcome"], "ok")
        commands = [" ".join(args) for args in result["calls"]]
        self.assertFalse(any(" up " in command for command in commands))
        compose = [command for command in commands if "-f compose.yaml" in command]
        self.assertTrue(all("--project-name existing-custom-project" in command for command in compose))
        self.assertTrue(all("--env-file" in command and "-f compose.gpu.yaml" in command for command in compose))
        self.assertIn("check_reranker_ollama.py", commands[-3])
        self.assertIn("eval_retrieval.py", commands[-2])
        self.assertIn("compare_retrieval_reports.py", commands[-1])
        self.assertTrue(result["restored"])
        self.assertEqual(len(result["output"]), 3)

    def test_stopped_service_only_starts_ollama(self):
        result = self.run_fake(running=False, device="cpu")
        self.assertEqual(result["outcome"], "ok")
        commands = [" ".join(args) for args in result["calls"]]
        starts = [command for command in commands if " up " in command]
        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0].endswith("up -d --no-recreate ollama"))
        self.assertFalse(any("compose.reranker.gpu.yaml" in command for command in commands))
        self.assertFalse(any("prune" in command or "pull" in command or "down" in command for command in commands))

    def test_missing_model_stops_before_benchmark(self):
        result = self.run_fake(preflight=2)
        self.assertIn("BGE-M3 missing", result["outcome"])
        self.assertFalse(any("eval_retrieval.py" in " ".join(args) for args in result["calls"]))
        self.assertTrue(result["restored"])

    def test_docker_unavailable_safe_failure(self):
        result = self.run_fake(daemon=1)
        self.assertIn("Docker is unavailable", result["outcome"])
        self.assertEqual(len(result["calls"]), 1)
