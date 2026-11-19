"""Offline tests against the current backend request and response contracts."""

from email import policy
from email.parser import BytesParser
from io import BytesIO
from types import SimpleNamespace

import pytest
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from neurosnap.api import NeurosnapAPI


@pytest.fixture
def api():
  client = NeurosnapAPI.__new__(NeurosnapAPI)
  client.headers = {"X-API-KEY": "test-key", "User-Agent": "sdk-test"}
  return client


def response(payload=None, status=200, **kwargs):
  return SimpleNamespace(status_code=status, json=lambda: payload, text="server error", **kwargs)


def parts(encoder):
  message = BytesParser(policy=policy.default).parsebytes(
    f"Content-Type: {encoder.content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + encoder.to_string()
  )
  return [(part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True))
          for part in message.iter_parts()]


def test_submit_encodes_route_and_query_and_preserves_repeated_uploads(api, monkeypatch):
  calls = []

  def request(method, url, **kwargs):
    calls.append((method, url, kwargs, parts(kwargs["data"])))
    return response("job-id")

  monkeypatch.setattr(requests, "request", request)
  fields = [("Input Complexes", ("one.pdb", BytesIO(b"first"))),
            ("Input Complexes", ("two.cif", BytesIO(b"second"))), ("Count", "2")]
  assert api.submit_job("Model / A+B", fields, note="a & b? #1", project_id="project-id") == "job-id"
  method, url, kwargs, uploaded = calls[0]
  assert method == "POST"
  assert url.endswith("/job/submit/Model%20%2F%20A%2BB")
  assert kwargs["params"] == {"note": "a & b? #1", "project_id": "project-id"}
  assert uploaded == [("Input Complexes", "one.pdb", b"first"), ("Input Complexes", "two.cif", b"second"), ("Count", None, b"2")]
  assert kwargs["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
  assert "Content-Type" not in api.headers


def test_prebuilt_multipart_encoder_is_preserved(api, monkeypatch):
  encoder = MultipartEncoder(fields={"Input Sequence": '{"aa":{"protein":"MKTAYIAK"}}'})

  def request(method, url, **kwargs):
    assert kwargs["data"] is encoder
    return response("job-id")

  monkeypatch.setattr(requests, "request", request)
  assert api.submit_job("Example", encoder) == "job-id"


@pytest.mark.parametrize("method,args,path,payload", [
  ("copy_job", ("job",), "/job/copy/job", None),
  ("cancel_job", ("job",), "/job/cancel/job", None),
  ("create_project", ("Study",), "/job-project/create",
   {"name": "Study", "description": "", "visibility": "personal", "allow_public_sharing": False}),
  ("update_project", ("project", "Study"), "/job-project/update", {"project_id": "project", "name": "Study", "description": ""}),
  ("delete_project", ("project",), "/job-project/delete", {"project_id": "project"}),
  ("assign_jobs_to_project", (["job"],), "/job-project/assign", {"job_ids": ["job"], "project_id": "000000000000000000000000"}),
  ("assign_jobs_to_project", (["job"], "project"), "/job-project/assign", {"job_ids": ["job"], "project_id": "project"}),
  ("remove_team_member", ("member",), "/teams/user/remove/member", None),
  ("set_team_credit_limit", ("member", 0), "/teams/user/credit-limit/member", {"limit": 0}),
  ("set_team_credit_limit", ("member", None), "/teams/user/credit-limit/member", {"limit": None}),
  ("set_team_roles", ([], "role"), "/teams/roles", {"roles": [], "default_role_id": "role"}),
  ("set_team_member_role", ("member", "role"), "/teams/member-role", {"user_id": "member", "role_id": "role"}),
  ("set_pipeline_descriptor", ("pipeline", "Study"), "/pipeline/descriptor/set", {"pipeline_id": "pipeline", "descriptor": "Study"}),
])
def test_mutation_contracts(api, monkeypatch, method, args, path, payload):
  calls = []

  def request(verb, url, **kwargs):
    calls.append((verb, url, kwargs))
    return response({"ok": True})

  monkeypatch.setattr(requests, "request", request)
  getattr(api, method)(*args)
  assert len(calls) == 1
  verb, url, kwargs = calls[0]
  assert verb == "POST" and url == api.BASE_URL + path
  assert kwargs.get("json") == payload
  assert kwargs["headers"] == api.headers


def test_project_access_options_preserve_explicit_empty_maps_and_false(api, monkeypatch):
  calls = []
  monkeypatch.setattr(requests, "request", lambda *args, **kwargs: calls.append(kwargs) or response({}))
  api.create_project("Study", visibility="restricted", members={"user": "manager"}, roles={"role": "viewer"})
  assert calls[0]["json"]["visibility"] == "restricted"
  assert calls[0]["json"]["members"] == {"user": "manager"}
  assert calls[0]["json"]["roles"] == {"role": "viewer"}
  assert "team_visible" not in calls[0]["json"]
  api.update_project("project", "Study", members={}, roles={}, allow_public_sharing=False)
  assert calls[1]["json"] == {"project_id": "project", "name": "Study", "description": "",
                               "members": {}, "roles": {}, "allow_public_sharing": False}


def test_discovery_usage_and_invitation_contracts(api, monkeypatch):
  calls = []
  monkeypatch.setattr(requests, "request", lambda *args, **kwargs: calls.append((args, kwargs)) or response([]))
  api.get_projects(scope="account")
  api.get_pipelines()
  api.get_team_credit_usage("member")
  api.invite_team_member("researcher@example.com")
  assert [call[0] for call in calls] == [
    ("GET", api.BASE_URL + "/job-projects"), ("GET", api.BASE_URL + "/pipelines/list"),
    ("GET", api.BASE_URL + "/teams/user/credit-limit/member"), ("POST", api.BASE_URL + "/teams/user/invite/"),
  ]
  assert calls[0][1]["params"] == {"scope": "account"}
  assert calls[3][1]["data"] == "researcher@example.com"
  assert "json" not in calls[3][1]


@pytest.mark.parametrize("method,path,payload", [
  ("validate_pipeline", "/pipelines/validate", {"ok": False, "errors": [{"message": "Invalid draft"}]}),
  ("submit_pipeline", "/pipelines/submit", "pipeline-id"),
])
def test_pipeline_multipart_and_in_band_validation(api, monkeypatch, method, path, payload):
  def request(verb, url, **kwargs):
    assert verb == "POST" and url == api.BASE_URL + path
    assert parts(kwargs["data"]) == [("pipeline_graph", None, b'{"nodes":[]}')]
    return response(payload)

  monkeypatch.setattr(requests, "request", request)
  assert getattr(api, method)({"pipeline_graph": '{"nodes":[]}'}) == payload


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "deleted"])
def test_polling_respects_delay_and_terminal_states(api, monkeypatch, terminal):
  states = iter(["pending", "running", terminal])
  sleeps = []
  monkeypatch.setattr(api, "get_job_status", lambda job_id: next(states))
  monkeypatch.setattr("neurosnap.api.time.sleep", sleeps.append)
  assert api.wait_job_status("job", delay=2) == terminal
  assert sleeps == [2, 2]


def test_polling_deadline_does_not_cancel_or_resubmit(api, monkeypatch):
  clock = [0]
  sleeps = []

  def sleep(delay):
    sleeps.append(delay)
    clock[0] += delay

  monkeypatch.setattr(api, "get_job_status", lambda job_id: "running")
  monkeypatch.setattr("neurosnap.api.time.monotonic", lambda: clock[0])
  monkeypatch.setattr("neurosnap.api.time.sleep", sleep)
  with pytest.raises(TimeoutError, match="job.*not been cancelled"):
    api.wait_job_status("job", delay=3, timeout=5)
  assert sleeps == [3, 2]


@pytest.mark.parametrize("kwargs", [{"delay": 0}, {"delay": -1}, {"timeout": 0}, {"timeout": -1}, {"delay": float("nan")}, {"timeout": float("inf")}])
def test_invalid_polling_intervals_are_rejected(api, kwargs):
  with pytest.raises(ValueError):
    api.wait_job_status("job", **kwargs)


def test_signed_file_download_never_forwards_account_headers(api, monkeypatch, tmp_path):
  calls = []

  def get(url, **kwargs):
    calls.append((url, kwargs))
    if len(calls) == 1:
      return response(status=307, headers={"Location": "https://storage.example.com/file?signature=secret"})
    return response(content=b"\x00\xff")

  monkeypatch.setattr(requests, "get", get)
  target = tmp_path / "result.bin"
  api.get_job_file("job", "out", "a #? +.bin", str(target), share_id="a&b")
  assert target.read_bytes() == b"\x00\xff"
  assert calls[0][0].endswith("/job/file/job/out/a%20%23%3F%20%2B.bin")
  assert calls[0][1] == {"headers": api.headers, "params": {"share": "a&b"}, "allow_redirects": False}
  assert calls[1] == ("https://storage.example.com/file?signature=secret", {})


@pytest.mark.parametrize("location", ["http://storage.example.com/file", "//storage.example.com/file", "https://user:pass@storage.example.com/file", ""])
def test_invalid_file_redirects_are_rejected(api, monkeypatch, location):
  monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response(status=307, headers={"Location": location}))
  with pytest.raises(requests.HTTPError, match="Invalid signed download"):
    api.get_job_file("job", "out", "result")


def test_file_text_and_data_share_parameters(api, monkeypatch):
  calls = []

  def get(url, **kwargs):
    calls.append((url, kwargs))
    return response({"config": {}, "in": [], "out": []})

  monkeypatch.setattr(requests, "get", get)
  assert api.get_job_file("job", "in", "result") == "server error"
  assert api.get_job_data("job", share_id="a&b")["out"] == []
  assert calls[1][1]["params"] == {"share": "a&b"}


def test_failed_mutations_are_not_retried(api, monkeypatch):
  calls = []

  def request(*args, **kwargs):
    calls.append(args)
    return response(status=403)

  monkeypatch.setattr(requests, "request", request)
  with pytest.raises(requests.HTTPError) as error:
    api.copy_job("job")
  assert error.value.response.status_code == 403
  assert len(calls) == 1


@pytest.mark.parametrize("format_type", [None, "JSON", "TABLE"])
def test_team_info_current_fields_and_optional_format(api, monkeypatch, capsys, format_type):
  payload = {"name": "Team", "members": [], "roles": [], "computeCreditsRemaining": 123, "seatsMax": 5}
  monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response(payload))
  assert api.get_team_info(format_type) == payload
  output = capsys.readouterr().out
  if format_type == "TABLE":
    assert "Compute Credits Remaining: 123" in output
    assert "Jobs Count" not in output
  elif format_type is None:
    assert output == ""


def test_shared_job_status_encodes_share_token(api, monkeypatch):
  calls = []
  monkeypatch.setattr(requests, "get", lambda *args, **kwargs: calls.append((args, kwargs)) or response("completed"))
  assert api.get_job_status("job", share_id="a&b") == "completed"
  assert calls[0][1]["params"] == {"share": "a&b"}
