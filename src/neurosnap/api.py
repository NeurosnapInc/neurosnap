import json
import math
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tabulate import tabulate

from neurosnap.constants.api import USER_AGENT
from neurosnap.log import logger


MultipartFields = Union[MultipartEncoder, Dict[str, Any], List[Tuple[str, Any]]]


class NeurosnapAPI:
  """Client wrapper for interacting with the Neurosnap REST API.

  Provides authenticated helpers for:
  - listing services and jobs
  - submitting jobs and monitoring status
  - retrieving job metadata and output files
  - managing job notes and sharing settings
  - organizing projects and managing team roles and credit limits
  - validating, submitting, and inspecting pipelines

  Authentication is performed with a Neurosnap API key, and connectivity is
  validated during initialization.
  """

  BASE_URL = "https://neurosnap.ai/api"

  def __init__(self, api_key: str):
    logger.info("⚠️ NEVER SHARE YOUR API KEY WITH ANY UNAUTHORIZED PARTIES ⚠️")
    self.api_key = api_key
    self.headers = {"X-API-KEY": self.api_key, "User-Agent": USER_AGENT}
    # test to ensure authenticated
    r = requests.head(f"{self.BASE_URL}/jobs", headers=self.headers)
    if r.status_code != 200:
      logger.error(
        f"Invalid API key provided (status code: {r.status_code}). Failed to setup the NeurosnapAPI object, please ensure your API key is correct."
      )
      raise ValueError("Invalid API key provided")
    else:
      logger.info(
        "Successfully connected to the Neurosnap API.\n - For information visit https://neurosnap.ai/blog/post/66b00dacec3f2aa9b4be703a\n - For support visit https://neurosnap.ai/support\n - For bug reports visit https://github.com/NeurosnapInc/neurosnap"
      )

  def _check_request(self, response: requests.Response):
    """Utility function for checking whether a request was sent and responded to correctly.
    Basically just a cleaner way of doing r.raise_for_status() with a more descriptive error.

    Parameters:
      response: The response object to check.

    Raises:
      HTTPError: If the API request fails.

    """
    if response.status_code != 200:
      msg = f"Expected status code 200, got {response.status_code}. Error: {response.text}"
      logger.error(msg)
      raise requests.HTTPError(msg, response=response)

  def _request(self, method: str, path: str, **kwargs) -> requests.Response:
    """Send one authenticated request without automatically retrying mutations."""
    headers = kwargs.pop("headers", self.headers)
    response = requests.request(method, f"{self.BASE_URL}{path}", headers=headers, **kwargs)
    self._check_request(response)
    return response

  def _post_multipart(self, path: str, fields: MultipartFields, params: Optional[Dict] = None) -> Any:
    """Preserve repeated file-field names and let the encoder supply the boundary."""
    multipart = fields if isinstance(fields, MultipartEncoder) else MultipartEncoder(fields=fields)
    headers = {**self.headers, "Content-Type": multipart.content_type}
    return self._request("POST", path, headers=headers, data=multipart, params=params).json()

  def get_services(self, format: Optional[str] = None) -> List[Dict]:
    """Fetches and returns a list of available Neurosnap services. Optionally prints the services.

    Parameters:
      format:
        - "table": Prints services in a tabular format with key fields.
        - "json": Prints services as formatted JSON.
        - None (default): No printing.

    Returns:
      A list of dictionaries representing available services.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/services", headers=self.headers)
    self._check_request(r)
    services = r.json()

    def format_service(service):
      """Helper to format individual service details."""
      title = service.get("title")
      link = urllib.parse.quote(title, safe="")
      link = f"https://neurosnap.ai/service/{link}"
      return {"Title": title, "Beta": service.get("beta"), "Link": link}

    formatted_services = [format_service(service) for service in services]
    if isinstance(format, str):
      format = format.lower()
      if format == "table":
        print(tabulate(formatted_services, headers="keys"))
      elif format == "json":
        print(json.dumps(services, indent=2))
    return services

  def get_jobs(self, format: Optional[str] = None) -> List[Dict]:
    """Fetches and returns a list of submitted jobs. Optionally prints the jobs.

    Parameters:
      format:
        - "table": Prints jobs in tabular format.
        - "json": Prints jobs as formatted JSON.
        - None (default): No printing.

    Returns:
      Submitted jobs as a list of dictionaries. ``Start`` and ``End`` are
      integer Unix timestamps in seconds, preserved as returned by the API.
      Zero means the job has not started or has not ended, respectively.
      The former formatted ``Runtime`` field is no longer returned; calculate
      elapsed runtime as ``End - Start``, using the current Unix time for
      running jobs with a zero ``End``. Pending jobs have no runtime yet.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/jobs", headers=self.headers)
    self._check_request(r)

    jobs = r.json()
    if isinstance(format, str):
      format = format.lower()
      if format == "table":
        print(tabulate(jobs, headers="keys"))
      elif format == "json":
        print(json.dumps(jobs, indent=2))
    return jobs

  def submit_job(self, service_name: str, fields: MultipartFields, note: Optional[str] = None, project_id: Optional[str] = None) -> str:
    """
    Submit a job to the Neurosnap platform.

    This function handles the submission of a job to a specified Neurosnap service using
    either a dictionary of input fields or a pre-constructed `MultipartEncoder` object.

    Parameters:
      - service_name: The name of the Neurosnap service (e.g., "TemStaPro", "NetSolP", "Boltz-1") to submit the job to.
      - fields: Either:
        - A dictionary mapping exact field titles to values, or
        - A list of (title, value) pairs, preserving repeated file fields, or
        - A ``MultipartEncoder`` for control over encoding.
        File values are (filename, binary file object, optional MIME type) tuples.
        Molecule lists accept CSV manifests and/or SDF/MOL/MOL2 uploads;
        structure lists accept repeated PDB/mmCIF uploads. Serialize sequence
        lists and grids with ``json.dumps`` as multipart text values. Scalar
        values, including numbers and booleans, should be strings. Follow the
        service's API example and ``/api/services`` form definitions.
      note (Optional[str]): An optional note or tag for the job.
      project_id (Optional[str]): Destination project ID; access checks apply.
        Submitting creates a billable run. No automatic submission retry is made.

    Returns:
      The job ID assigned to the submitted job.

    Raises:
      HTTPError: If the API request fails or returns a non-200 status code.
    """
    params = {}
    if note is not None:
      params["note"] = note
    if project_id is not None:
      params["project_id"] = project_id
    job_id = self._post_multipart(f"/job/submit/{urllib.parse.quote(service_name, safe='')}", fields, params=params)
    logger.info(f"{service_name} job successfully submitted with ID {job_id}.")
    return job_id

  def copy_job(self, job_id: str) -> str:
    """Duplicate a job and launch a new billable run. Return the new job ID."""
    return self._request("POST", f"/job/copy/{urllib.parse.quote(job_id, safe='')}").json()

  def cancel_job(self, job_id: str) -> None:
    """Request cancellation and settle incurred credits; check status afterwards."""
    self._request("POST", f"/job/cancel/{urllib.parse.quote(job_id, safe='')}")

  def get_job_status(self, job_id: str, share_id: Optional[str] = None) -> str:
    """Fetches the status of a specified job.

    Parameters:
      job_id: The ID of the job.
      share_id: Optional share token for a shared job.

    Returns:
      The status of the job.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(
      f"{self.BASE_URL}/job/status/{urllib.parse.quote(job_id, safe='')}",
      headers=self.headers, params={"share": share_id} if share_id else None,
    )
    self._check_request(r)
    return r.json()

  def wait_job_status(self, job_id: str, delay: float = 5, timeout: Optional[float] = None) -> str:
    """Poll an existing job until completed, failed, cancelled, or deleted.

    Parameters:
      job_id: The existing job ID; this method never submits or retries jobs.
      delay: Seconds between status requests; must be positive.
      timeout: Optional polling deadline in seconds. A timeout does not cancel
        the job. The deadline is checked between HTTP requests.

    Returns:
      The terminal status string.

    Raises:
      HTTPError: If a status request fails.
      TimeoutError: If the polling deadline expires.
      ValueError: If delay or timeout is not positive.
    """
    if not math.isfinite(delay) or delay <= 0 or (timeout is not None and (not math.isfinite(timeout) or timeout <= 0)):
      raise ValueError("delay and timeout must be finite and positive")
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"Monitoring timed out for job {job_id}; the job has not been cancelled.")
      status = self.get_job_status(job_id)
      logger.info(f"Waiting for Neurosnap job with ID {job_id} (current status: {status})")
      if status in {"completed", "failed", "cancelled", "deleted"}:
        return status
      remaining = delay if deadline is None else min(delay, max(0, deadline - time.monotonic()))
      time.sleep(remaining)

  def get_job_data(self, job_id: str, share_id: str = None) -> Dict[str, Any]:
    """
    Fetches the config and all files from a Neurosnap job.
    Output files are only included once the job has completed.

    Parameters:
      job_id: The ID of the job.
      share_id: The share ID, if any.

    Returns:
      Dictionary with keys ``config``, ``in``, and ``out`` corresponding to config used, input files, and output files.

    Raises:
      HTTPError: If the API request fails.
    """
    # Construct the URL for the request
    url = f"{self.BASE_URL}/job/data/{job_id}"
    params = {"share": share_id} if share_id else None

    # Make the request and check the status
    r = requests.get(url, headers=self.headers, params=params)
    self._check_request(r)

    # Parse the response
    files = r.json()
    return files

  def get_job_file(self, job_id: str, file_type: str, file_name: str, save_path: Union[str, None] = None, share_id: str = None) -> Union[str, None]:
    """Fetches a specific file from a completed Neurosnap job and saves it to the specified path or
    returns it as a string if no path is provided.

    Parameters:
      job_id: The ID of the job.
      file_type: The type of file to fetch (can be ``in`` or ``out``).
      file_name: The name of the specific file to fetch.
      save_path: The path where the file content will be saved.
      share_id: The share ID, if any.

    Returns:
      File contents as text when ``save_path`` is omitted, or ``None`` after
      writing the file's binary bytes to ``save_path``. Signed storage downloads
      do not receive your account API key.

    Raises:
      HTTPError: If the API request fails.

    """
    if file_type not in {"in", "out"}:
      raise ValueError("file_type must be 'in' or 'out'")
    filename = urllib.parse.quote(file_name, safe="")
    url = f"{self.BASE_URL}/job/file/{urllib.parse.quote(job_id, safe='')}/{file_type}/{filename}"
    params = {"share": share_id} if share_id else None
    # Account credentials belong only on the Neurosnap request. Storage gets
    # the signed URL, never the account's custom X-API-KEY header.
    r = requests.get(url, headers=self.headers, params=params, allow_redirects=False)
    if r.status_code == 307:
      location = r.headers.get("Location", "")
      parsed = urllib.parse.urlparse(location)
      if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise requests.HTTPError("Invalid signed download URL returned by Neurosnap", response=r)
      r = requests.get(location)
    self._check_request(r)
    if save_path is None:
      return r.text
    else:
      with open(save_path, "wb") as f:
        f.write(r.content)
      logger.info(f"File saved to {save_path}")

  def set_job_note(self, job_id: str, note: str) -> None:
    """Set a note for a submitted job.

    Parameters:
      job_id: The ID of the job for which the note will be set.
      note: The note to be associated with the job.

    """
    # Prepare the request data
    payload = {"job_id": job_id, "note": note}
    # Send the POST request
    r = requests.post(f"{self.BASE_URL}/job/note/set", headers=self.headers, json=payload)
    self._check_request(r)
    logger.info(f"Note set successfully for job ID {job_id}.")

  def set_job_share(self, job_id: str) -> str:
    """Enables the sharing feature of a job and makes it public.

    Parameters:
      job_id: The ID of the job to be made public.

    Returns:
      The JSON response containing the share ID.

    """
    # Send the request to set job share
    r = requests.get(f"{self.BASE_URL}/job/share/set/{job_id}", headers=self.headers)
    self._check_request(r)

    share_id = r.json()
    logger.info(f"Job with ID {job_id} is now public with share ID {share_id}.")
    return share_id

  def delete_job_share(self, job_id: str) -> None:
    """Disables the sharing feature of a job and makes the job private.

    Parameters:
      job_id: The ID of the job to be made private.

    """
    # Send the request to delete job share
    r = requests.get(f"{self.BASE_URL}/job/share/delete/{job_id}", headers=self.headers)
    self._check_request(r)
    logger.info(f"Job with ID {job_id} is now private.")

  def get_team_info(self, format_type: Optional[str] = None) -> Dict:
    """Fetches your team's information if you are part of a Neurosnap Team.

    Parameters:
      format_type: The format to print the response: 'table', 'json', or None for no output.

    Returns:
      Team information including members, roles, permission names, capability
      flags, ``computeCreditsRemaining``, and ``seatsMax``. Membership and
      permissions determine which actions and billing fields are accessible.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/teams/info", headers=self.headers)
    self._check_request(r)

    team_info = r.json()
    format_type = format_type.lower() if isinstance(format_type, str) else None
    if format_type == "table":
      # Prepare data for tabulation
      members_table = [
        [
          member.get("ID", ""),
          member.get("Name", ""),
          member.get("Email", ""),
          member.get("LastLogin", ""),
          member.get("EmailVerified", ""),
          member.get("Leader", ""),
        ]
        for member in team_info.get("members", [])
      ]
      # Print the team info in table format
      print(tabulate(members_table, headers=["ID", "Name", "Email", "Last Login", "Email Verified", "Leader"], tablefmt="grid"))
      print(f"\nTeam Name: {team_info.get('name', '')}")
      print(f"Compute Credits Remaining: {team_info.get('computeCreditsRemaining', 0)}")
      print(f"Max Seats: {team_info.get('seatsMax', 0)}")
      print(f"Is Leader: {team_info.get('is_leader', False)}")
    elif format_type == "json":
      # Print the team information in JSON format
      print(json.dumps(team_info, indent=2))

    return team_info

  def get_team_jobs(self, format_type: Optional[str] = None) -> List[Dict]:
    """Fetches all the jobs submitted by all members of your Neurosnap Team.

    Parameters:
      format_type: The format to print the response: 'table', 'JSON', or None for no output.

    Returns:
      A list of jobs submitted by the team members. ``Start`` and ``End``
      are integer Unix timestamps in seconds, preserved as returned by the API.
      Zero means the job has not started or has not ended, respectively.
      Calculate runtime from these fields as described in :meth:`get_jobs`;
      the former formatted ``Runtime`` field is no longer returned.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/teams/jobs", headers=self.headers)
    self._check_request(r)

    jobs = r.json()
    if isinstance(format_type, str):
      format_type = format_type.lower()
      if format_type == "table":
        print(tabulate(jobs, headers="keys"))
      elif format_type == "json":
        print(json.dumps(jobs, indent=2))
    return jobs

  def get_pipeline(self, pipeline_id: str) -> Dict:
    """Fetches the pipeline data associated with a submitted Neurosnap pipeline.

    Parameters:
      pipeline_id: The ID of the pipeline to fetch.

    Returns:
      A dictionary containing ``pipeline``, ``jobs``, and ``errors``. Job rows
      preserve the API's ``Start`` and ``End`` Unix timestamps in seconds;
      calculate runtime from these fields as described in :meth:`get_jobs`.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/pipelines/status/{pipeline_id}", headers=self.headers)
    self._check_request(r)
    return r.json()

  def get_projects(self, scope: Optional[str] = None) -> List[Dict]:
    """List accessible projects, including access flags and job counts.

    ``scope="account"`` limits platform admins to their account's projects.
    Project visibility is ``personal``, ``team``, or ``restricted``.
    """
    return self._request("GET", "/job-projects", params={"scope": scope} if scope else None).json()

  def create_project(
    self, name: str, description: str = "", visibility: str = "personal",
    members: Optional[Dict[str, str]] = None, roles: Optional[Dict[str, str]] = None,
    allow_public_sharing: bool = False,
  ) -> Dict:
    """Create a project with the current visibility/access schema.

    ``members`` and ``roles`` map user/role IDs to ``viewer``, ``contributor``,
    or ``manager`` grants. Team permissions apply to non-personal projects.
    """
    payload = {"name": name, "description": description, "visibility": visibility,
               "allow_public_sharing": allow_public_sharing}
    if members is not None:
      payload["members"] = members
    if roles is not None:
      payload["roles"] = roles
    return self._request("POST", "/job-project/create", json=payload).json()

  def update_project(
    self, project_id: str, name: str, description: str = "", *,
    visibility: Optional[str] = None, members: Optional[Dict[str, str]] = None,
    roles: Optional[Dict[str, str]] = None, allow_public_sharing: Optional[bool] = None,
  ) -> Dict:
    """Update project metadata and optional access policy.

    Omitted policy fields are preserved; pass an empty grant map to clear it.
    ``description`` defaults to empty, so pass the existing value to retain it.
    The server rejects access changes that would expose restricted research.
    """
    payload = {"project_id": project_id, "name": name, "description": description}
    for key, value in {"visibility": visibility, "members": members, "roles": roles,
                       "allow_public_sharing": allow_public_sharing}.items():
      if value is not None:
        payload[key] = value
    return self._request("POST", "/job-project/update", json=payload).json()

  def delete_project(self, project_id: str) -> None:
    """Delete an empty project. The server rejects projects containing research."""
    self._request("POST", "/job-project/delete", json={"project_id": project_id})

  def assign_jobs_to_project(self, job_ids: List[str], project_id: Optional[str] = None) -> Dict:
    """Move jobs to a project, or unassign them when project_id is None.

    Ownership, current permissions, and destination access checks still apply.
    Unassigning a job does not delete or rerun it.
    """
    return self._request("POST", "/job-project/assign", json={
      "job_ids": job_ids, "project_id": project_id if project_id is not None else "000000000000000000000000",
    }).json()

  def invite_team_member(self, email: str) -> str:
    """Invite an existing verified account; requires members.invite permission."""
    return self._request("POST", "/teams/user/invite/", data=email).json()

  def remove_team_member(self, user_id: str) -> str:
    """Remove a member with POST; requires members.remove permission."""
    return self._request("POST", f"/teams/user/remove/{urllib.parse.quote(user_id, safe='')}").json()

  def get_team_credit_usage(self, user_id: str) -> Dict:
    """Read a member's billing-period credit limit, usage, and reservation totals."""
    return self._request("GET", f"/teams/user/credit-limit/{urllib.parse.quote(user_id, safe='')}").json()

  def set_team_credit_limit(self, user_id: str, limit: Optional[float]) -> Dict:
    """Set a billing-period limit and return usage. Zero blocks spending; None is unlimited.

    Requires billing.manage permission. Negative/nonfinite values are rejected
    by the server. This sets an allowance, not a prepaid credit transfer.
    """
    return self._request("POST", f"/teams/user/credit-limit/{urllib.parse.quote(user_id, safe='')}", json={"limit": limit}).json()

  def set_team_roles(self, roles: List[Dict], default_role_id: str) -> Dict:
    """Replace team roles/default role; requires roles.manage permission.

    Each role contains id, name, permissions, and unprojected_access. Read
    get_team_info() for current definitions and supported permission names.
    Roles still assigned to members or projects cannot be removed.
    """
    return self._request("POST", "/teams/roles", json={"roles": roles, "default_role_id": default_role_id}).json()

  def set_team_member_role(self, user_id: str, role_id: str) -> None:
    """Assign an existing role to a non-owner member; requires roles.manage."""
    self._request("POST", "/teams/member-role", json={"user_id": user_id, "role_id": role_id})

  def get_pipelines(self) -> List[Dict]:
    """List accessible pipeline history, newest first."""
    return self._request("GET", "/pipelines/list").json()

  def validate_pipeline(self, fields: MultipartFields) -> Dict:
    """Validate a builder-compatible multipart draft without creating jobs.

    Supply ``pipeline_graph`` as JSON text and manifest/file parts named
    ``<node_id>_<filename>``. Validation can return HTTP 200 with ``ok=false``;
    inspect ``ok``, ``errors``, ``estimated_cost``, and ``node_sizes`` before submitting.
    """
    return self._post_multipart("/pipelines/validate", fields)

  def submit_pipeline(self, fields: MultipartFields) -> str:
    """Submit a billable pipeline; return its ID without automatically retrying.

    Supply builder-compatible ``pipeline_graph`` JSON text, ``template``,
    ``descriptor``, and required ``<node_id>_<filename>`` uploads. Optional
    ``project_id`` assigns the project; ``submission_token`` supports the
    backend's duplicate-submission reconciliation. Use the exported builder
    configuration; pipeline graphs differ from ordinary service fields.
    """
    return self._post_multipart("/pipelines/submit", fields)

  def set_pipeline_descriptor(self, pipeline_id: str, descriptor: str) -> str:
    """Rename an accessible pipeline without rerunning its jobs."""
    return self._request("POST", "/pipeline/descriptor/set", json={"pipeline_id": pipeline_id, "descriptor": descriptor}).json()
