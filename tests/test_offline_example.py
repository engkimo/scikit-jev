"""Keep the first-run example runnable under the ordinary no-network fixture."""

import runpy
from pathlib import Path


def test_offline_workflow_exposes_schema_metadata_and_logical_calls() -> None:
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "offline_workflow.py"))

    result = namespace["run_demo"]()

    assert result["feature_shape"] == [2, 6]
    assert len(result["feature_names"]) == 6
    assert result["feature_names"][1:3] == ["topic__p[technical]", "topic__p[billing]"]
    assert result["resolved_models"] == ["synthetic-demo-v1"] * 2
    assert result["usage"] == {"evaluation_logical_calls": 2, "input_tokens": None, "output_tokens": None}
    assert result["execution"] == {
        "fit_logical_calls": 0,
        "pipeline_training_logical_calls": 8,
        "total_logical_calls": 10,
        "client_opens": 2,
        "client_closes": 2,
    }
    assert "no semantic model" in result["mode"]
