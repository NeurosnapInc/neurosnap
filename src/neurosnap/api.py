import json
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Union

import requests
from tabulate import tabulate

from neurosnap.log import logger

# try to get the version of the user agent
version = "X"
try:
  from importlib.metadata import version

  version("neurosnap")
except:
  pass
# User agent to use throughout the application
USER_AGENT = f"Neurosnap-OSS-Tools/v{version}"


class NeurosnapAPI:
  BASE_URL = "https://neurosnap.ai/api"

  def __init__(self, api_key: str):
    logger.warning("⚠️ NEVER SHARE YOUR API KEY WITH ANY UNAUTHORIZED PARTIES ⚠️")
    self.api_key = api_key
    self.headers = {"X-API-KEY": self.api_key, "User-Agent": USER_AGENT}
    # test to ensure authenticated
    r = requests.head(f"{self.BASE_URL}/jobs", headers=self.headers)
    if r.status_code != 200:
      logger.error(
        f"Invalid API key provided (status code: {r.status_code}). Failed to setup the NeurosnapAPI object, please ensure your API key is correct."
      )
      raise ValueError("Invalid token provided")
    else:
      logger.info(
        "Successfully connected to the Neurosnap API.\n - For information visit https://neurosnap.ai/blog/post/66b00dacec3f2aa9b4be703a\n - For support visit https://neurosnap.ai/support\n - For bug reports visit https://github.com/NeurosnapInc/neurosnap"
      )

  def get_services(self, format_type: Optional[str] = None) -> List[Dict]:
    """Fetches and returns a list of available Neurosnap services. Optionally prints the services.

    Parameters:
      format_type:
        - "table": Prints services in a tabular format with key fields.
        - "json": Prints services as formatted JSON.
        - None (default): No printing.

    Returns:
      A list of dictionaries representing available services.

    Raises:
      HTTPError: If the API request fails.

    """
    response = requests.get(f"{self.BASE_URL}/services", headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    services = response.json()

    def format_service(service):
      """Helper to format individual service details."""
      title = service.get("title")
      link = urllib.parse.quote(title)
      link = f"https://neurosnap.ai/service/{link}"
      return {"Title": title, "Beta": service.get("beta"), "Link": link}

    formatted_services = [format_service(service) for service in services]
    format_type = format_type.lower()
    if format_type == "table":
      print(tabulate(formatted_services, headers="keys"))
    elif format_type == "json":
      print(json.dumps(services, indent=2))
    return services

  def get_jobs(self, format_type: Optional[str] = None) -> List[Dict]:
    """Fetches and returns a list of submitted jobs. Optionally prints the jobs.

    Parameters:
      format_type:
        - "table": Prints jobs in tabular format.
        - "json": Prints jobs as formatted JSON.
        - None (default): No printing.

    Returns:
      Submitted jobs as a list of dictionaries.

    Raises:
      HTTPError: If the API request fails.

    """
    response = requests.get(f"{self.BASE_URL}/jobs", headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    jobs = response.json()
    for job in jobs:
      if "Submitted" in job:
        timestamp = job["Submitted"] // 1000  # Convert milliseconds to seconds
        job["Submitted"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    format_type = format_type.lower()
    if format_type == "table":
      print(tabulate(jobs, headers="keys"))
    elif format_type == "json":
      print(json.dumps(jobs, indent=2))
    return jobs

  def submit_job(self, service_name: str, files: Dict[str, str], data: Dict[str, str], note: Optional[str] = None) -> Dict:
    """Submit a Neurosnap job.

    Parameters:
      service_name: The name of the service to run.
      files: A dictionary mapping file names to file paths.
      data: A dictionary of additional data to be passed to the service.

    Returns:
      The job ID of the submitted job.

    Raises:
      HTTPError: If the API request fails.

    """
    # Filter out data fields with value 'false'
    filtered_data = {k: v for k, v in data.items() if v != "false"}

    # Open the files in binary mode
    files_dict = {k: open(v, "rb") for k, v in files.items()}

    # Make the POST request
    url = f"{self.BASE_URL}/job/submit/{service_name}"
    if note is not None:
      url += f"?note={note}"
    response = requests.post(url, headers=self.headers, files=files_dict, data=filtered_data)

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"

    # Return the job ID
    return response.json()

  def get_job_status(self, job_id: str) -> str:
    """Fetches the status of a specified job.

    Parameters:
      job_id: The ID of the job.

    Returns:
      The status of the job.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/job/status/{job_id}", headers=self.headers)
    assert r.status_code == 200, f"Expected status code 200, got {r.status_code}. Error: {r.text}"
    return r.json()

  def wait_job_status(self, job_id: str, delay: int = 5) -> str:
    """Waits for the status of a job to either change to ``completed`` or ``failed``.

    Parameters:
      job_id: The ID of the job.
      delay: The delay between requests in seconds.

    Returns:
      The status of the job.

    Raises:
      HTTPError: If the API request fails.
      Exception: If job fails.

    """
    status = ""
    while status != "completed" and status != "failed":
      time.sleep(5)
      r = requests.get(f"{self.BASE_URL}/job/status/{job_id}", headers=self.headers)
      r.raise_for_status()
      status = r.json()
      logger.info(f"Waiting for Neurosnap job with ID {job_id} (current status: {status})")
    return status

  def get_job_files(self, job_id: str, file_type: str, share_id: str = None, format_type: Optional[str] = None) -> List[str]:
    """Fetches all files from a completed Neurosnap job and optionally prints them.

    Parameters:
      job_id: The ID of the job.
      file_type: The type of files to fetch.
      share_id: The share ID, if any.
      format_type:
        - "table": Prints the files in a tabular format.
        - "json": Prints the files in formatted JSON.
        - None (default): No printing.

    Returns:
      A list of file names from the job.

    Raises:
      HTTPError: If the API request fails.

    """
    # Construct the URL for the request
    url = f"{self.BASE_URL}/job/files/{job_id}/{file_type}"
    if share_id:
      url += f"?share={share_id}"
    # Make the request and check the status
    response = requests.get(url, headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    # Parse the response
    files = response.json()
    # Optionally format the output based on the format_type
    format_type = format_type.lower()
    if format_type == "table":
      # Print the file details in a table format
      print(tabulate(files, headers=["File Name", "File Size"]))
    elif format_type == "json":
      # Print the file details in JSON format
      print(json.dumps(files, indent=2))
    return files

  def get_job_file(self, job_id: str, file_type: str, file_name: str, save_path: Union[str, None] = None, share_id: str = None) -> Union[str, None]:
    """Fetches a specific file from a completed Neurosnap job and saves it to the specified path or
    returns it as a string if no path is provided.

    Parameters:
      job_id: The ID of the job.
      file_type: The type of file to fetch.
      file_name: The name of the specific file to fetch (can be ``in`` or ``out``).
      save_path: The path where the file content will be saved.
      share_id: The share ID, if any.

    Returns:
      Contents of the file fetched as a string or ``None`` if save_path is not provided.

    Raises:
      HTTPError: If the API request fails.

    """
    # Construct the URL for the request
    url = f"{self.BASE_URL}/job/file/{job_id}/{file_type}/{file_name}"
    if share_id:
      url += f"?share={share_id}"

    # Make the request and check the status
    r = requests.get(url, headers=self.headers)
    r.raise_for_status()
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
    response = requests.post(f"{self.BASE_URL}/job/note/set", headers=self.headers, json=payload)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    print(f"Note set successfully for job ID {job_id}.")

  def set_job_share(self, job_id: str) -> Dict:
    """Enables the sharing feature of a job and makes it public.

    Parameters:
      job_id: The ID of the job to be made public.

    Returns:
      The JSON response containing the share ID.

    """
    # Send the request to set job share
    response = requests.get(f"{self.BASE_URL}/job/share/set/{job_id}", headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    return response.json()  # Return the JSON output containing the share ID

  def delete_job_share(self, job_id: str) -> None:
    """Disables the sharing feature of a job and makes the job private.

    Parameters:
      job_id: The ID of the job to be made private.

    """
    # Send the request to delete job share
    response = requests.get(f"{self.BASE_URL}/job/share/delete/{job_id}", headers=self.headers)

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"
    print(f"Job ID {job_id} is now private.")

  def get_team_info(self, format_type: Optional[str] = None) -> Dict:
    """Fetches your team's information if you are part of a Neurosnap Team.

    Parameters:
      format_type: The format to print the response: 'table', 'json', or None for no output.

    Returns:
      The team information.

    Raises:
      HTTPError: If the API request fails.

    """
    response = requests.get(f"{self.BASE_URL}/teams/info", headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"

    team_info = response.json()
    format_type = format_type.lower()
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
      print(f"Jobs Count: {team_info.get('jobsCount', 0)}")
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
      A list of jobs submitted by the team members.

    Raises:
      HTTPError: If the API request fails.

    """
    response = requests.get(f"{self.BASE_URL}/teams/jobs", headers=self.headers)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Error: {response.text}"

    jobs = response.json()
    for job in jobs:
      if "Submitted" in job:
        timestamp = job["Submitted"] // 1000  # Convert milliseconds to seconds
        job["Submitted"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
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
      Pipeline object.

    Raises:
      HTTPError: If the API request fails.

    """
    r = requests.get(f"{self.BASE_URL}/pipelines/status/{pipeline_id}", headers=self.headers)
    r.raise_for_status()
    return r.json()
