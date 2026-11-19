"""Offline regression checks for the REST job timing contract."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import requests

from neurosnap.api import NeurosnapAPI


@pytest.fixture
def api():
  client = NeurosnapAPI.__new__(NeurosnapAPI)
  client.headers = {"X-API-KEY": "test-key"}
  return client


@pytest.fixture
def jobs():
  return [
    {"Job ID": "66aea7b235c83718af6161d5", "Status": "completed", "Start": 1722722250, "End": 1722722281},
    {"Job ID": "66ae52068d3b43e471e39c13", "Status": "running", "Start": 1722700320, "End": 0},
    {"Job ID": "66ae52068d3b43e471e39c14", "Status": "pending", "Start": 0, "End": 0},
  ]


@pytest.mark.parametrize("method,endpoint,keyword", [
  ("get_jobs", "/jobs", "format"),
  ("get_team_jobs", "/teams/jobs", "format_type"),
])
@pytest.mark.parametrize("output_format", [None, "TABLE", "JSON"])
def test_job_lists_preserve_raw_timestamps(api, jobs, monkeypatch, capsys, method, endpoint, keyword, output_format):
  original = deepcopy(jobs)
  requests_seen = []

  def get(url, headers):
    requests_seen.append((url, headers))
    return SimpleNamespace(status_code=200, json=lambda: jobs)

  monkeypatch.setattr(requests, "get", get)
  result = getattr(api, method)(**{keyword: output_format})
  assert result == original
  assert result is jobs
  assert requests_seen == [(api.BASE_URL + endpoint, api.headers)]
  for row in result:
    assert isinstance(row["Start"], int)
    assert isinstance(row["End"], int)
    assert "Runtime" not in row
    assert "Submitted" not in row
  output = capsys.readouterr().out
  if output_format is None:
    assert output == ""
  elif output_format == "JSON":
    assert json.loads(output) == original
  else:
    assert "Start" in output and "End" in output
    assert "1722722250" in output
    assert "Runtime" not in output and "Submitted" not in output


@pytest.mark.parametrize("method", ["get_jobs", "get_team_jobs"])
def test_default_job_list_formats_are_optional(api, jobs, monkeypatch, capsys, method):
  monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: jobs))
  assert getattr(api, method)() == jobs
  assert capsys.readouterr().out == ""


@pytest.mark.parametrize("method", ["get_jobs", "get_team_jobs"])
def test_job_lists_still_report_http_errors(api, monkeypatch, method):
  monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=403, text="Access denied"))
  with pytest.raises(requests.HTTPError, match="403"):
    getattr(api, method)()


def test_pipeline_jobs_preserve_timestamps(api, jobs, monkeypatch):
  payload = {"pipeline": {"id": "example"}, "jobs": jobs, "errors": []}
  original = deepcopy(payload)
  requests_seen = []

  def get(url, headers):
    requests_seen.append((url, headers))
    return SimpleNamespace(status_code=200, json=lambda: payload)

  monkeypatch.setattr(requests, "get", get)
  assert api.get_pipeline("example") == original
  assert requests_seen == [(api.BASE_URL + "/pipelines/status/example", api.headers)]
