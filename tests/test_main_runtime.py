import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
from hirag.telemetry import new_telemetry_session, summarize_telemetry


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse("hello")


class _FakeAsyncOpenAI:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        _FakeAsyncOpenAI.last_instance = self


class _FakeOpenAIModule:
    AsyncOpenAI = _FakeAsyncOpenAI


class _FakeHashingKV:
    def __init__(self, cached=None) -> None:
        self.cached = cached
        self.upserts = []

    async def get_by_id(self, _id):
        return self.cached

    async def upsert(self, data):
        self.upserts.append(data)


class MainRuntimeTests(unittest.TestCase):
    def test_internal_kwargs_are_removed_before_api_request(self) -> None:
        telemetry = new_telemetry_session("unit")
        runtime = {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "chat_model": "demo-model",
            "telemetry": telemetry,
            "benchmark_label": "unit",
        }

        with patch.object(main.importlib, "import_module", return_value=_FakeOpenAIModule):
            llm_func = main.create_vllm_chat_model(runtime)
            asyncio.run(
                llm_func(
                    "hello",
                    stage="entity_extract",
                    benchmark_label="override",
                    telemetry=telemetry,
                    max_tokens=32,
                )
            )

        request = _FakeAsyncOpenAI.last_instance.chat.completions.calls[0]
        self.assertNotIn("stage", request)
        self.assertNotIn("telemetry", request)
        self.assertNotIn("benchmark_label", request)
        self.assertEqual(request["max_tokens"], 32)

    def test_telemetry_summary_records_stage_usage_and_cache_hits(self) -> None:
        telemetry = new_telemetry_session("unit")
        runtime = {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "chat_model": "demo-model",
            "telemetry": telemetry,
            "benchmark_label": "unit",
        }

        with patch.object(main.importlib, "import_module", return_value=_FakeOpenAIModule):
            llm_func = main.create_vllm_chat_model(runtime)
            asyncio.run(
                llm_func(
                    "hello",
                    stage="entity_extract",
                    max_tokens=32,
                )
            )
            asyncio.run(
                llm_func(
                    "hello",
                    stage="entity_extract",
                    hashing_kv=_FakeHashingKV({"return": "cached"}),
                )
            )

        telemetry_payload = summarize_telemetry(telemetry)
        stage = telemetry_payload["summary"]["stages"]["entity_extract"]
        self.assertEqual(stage["call_count"], 2)
        self.assertEqual(stage["cache_hits"], 1)
        self.assertGreaterEqual(stage["total_tokens"], 18)

    def test_single_run_cli_parses_prompt_regime_and_gleaning(self) -> None:
        argv = [
            "main.py",
            "--prompt-regime",
            "lean",
            "--entity-extract-max-gleaning",
            "0",
            "--best-model-max-async",
            "4",
            "--cheap-model-max-async",
            "6",
            "--community-report-input-max-tokens",
            "7000",
            "--cluster-summary-input-max-tokens",
            "5000",
            "--telemetry-output",
            "/tmp/test-telemetry.json",
            "--benchmark-label",
            "demo",
        ]
        with patch("sys.argv", argv):
            args = main.parse_args()

        self.assertEqual(args.prompt_regime, "lean")
        self.assertEqual(args.entity_extract_max_gleaning, 0)
        self.assertEqual(args.best_model_max_async, 4)
        self.assertEqual(args.cheap_model_max_async, 6)
        self.assertEqual(args.community_report_input_max_tokens, 7000)
        self.assertEqual(args.cluster_summary_input_max_tokens, 5000)
        self.assertEqual(args.telemetry_output, "/tmp/test-telemetry.json")
        self.assertEqual(args.benchmark_label, "demo")


if __name__ == "__main__":
    unittest.main()
