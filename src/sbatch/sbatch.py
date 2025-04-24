import json
import os
import pathlib
import re
import subprocess
import time

import numpy as np

USER_NAME = os.environ["USER"]


class SqueueJob:
    def __init__(self, info_dict: dict):
        self.info_dict = info_dict
        self.job_id = info_dict["job_id"]
        try:
            self.nodes = info_dict["job_resources"]["nodes"]
        except KeyError:
            # job pending and has no nodes info
            self.nodes = ""
        self.user_name = info_dict["user_name"]
        self.job_state = info_dict["job_state"]

    def mine(self):
        """My job"""
        return self.user_name == USER_NAME

    def running(self):
        """Job is running"""
        return "RUNNING" in self.job_state

    def __getitem__(self, key):
        return self.info_dict[key]


class SlurmManager:
    def __init__(
        self,
        job_name: str,
        command: str,
        time: str = "12:00:00",
        partition: str = "gpu",
        cpus_per_task: int = 32,
        mem_per_cpu: str = "4G",
        gpus: int = 0,
        output: str = "auto",
        error: str = "auto",
        exclude: str = "",
        nodelist: str = "",
        chdir: str = ".",
        max_jobs: int = 16,
    ):
        job_type = "cpu"
        if ("gpu" in partition.lower()) or (gpus > 0):
            job_type = "gpu"
            if gpus == 0:
                gpus = 1
        self.job_type = job_type
        self.command = command
        self.config = {
            "job-name": job_name,
            "time": time,
            "partition": partition,
            "cpus-per-task": cpus_per_task,
            "mem-per-cpu": mem_per_cpu,
            "gpus": gpus,
            "output": output,
            "error": error,
            "exclude": exclude,
            "nodelist": nodelist,
            "chdir": chdir,
        }
        self.max_jobs = max_jobs
        if self.job_type == "cpu":
            self.config["gpus"] = ""

    @property
    def _chdir(self):
        return pathlib.Path(self.config["chdir"])

    @property
    def _sbatch_dir(self):
        _path = self._chdir / "sbatch"
        _path.mkdir(exist_ok=True, parents=True)
        return _path

    @property
    def _savename(self):
        return f"{self._sbatch_dir}/{self.config['job-name']}"

    @property
    def _script_path(self):
        return pathlib.Path(f"{self._savename}.sh")

    @property
    def _job_id_path(
        self,
    ):
        return pathlib.Path(f"{self._savename}.job_id.txt")

    @property
    def _finish_flag(self):
        return pathlib.Path(f"{self._savename}.finish")

    def _create_script(self):
        script = "#!/bin/bash\n"
        for key, value in self.config.items():
            if value == "" or value is None:
                continue

            if key in ("output", "error") and value == "auto":
                value = f"{self._savename}.%j.{key}.log"

            script += f"#SBATCH --{key}={value}\n"

        script += f"cd {self.config['chdir']}\n"
        # execute cmd with timer
        script += "date\n"
        script += f"{self.command}\n"
        script += "date\n"
        # save a sbatch level success flag
        script += f"echo $? > {self._finish_flag}\n"

        if self._script_path.exists():
            # compare the old and new script
            with open(self._script_path) as f:
                old_script = f.read()
            assert old_script == script, (
                f"Script already exists and is different from the new one. "
                f"Please use a different job name or remove it manually. {self._script_path}"
            )
        else:
            with open(self._script_path, "w") as f:
                f.write(script)
        return

    def _reach_max_jobs(self):
        partition = self.config["partition"]

        jobs = self.get_running_jobs(running_only=False, mine_only=True)
        count = 0
        for job in jobs:
            if job["partition"] == partition:
                count += 1
        print(f"Current running jobs in partition {partition}: {count}")
        if count >= self.max_jobs:
            return True
        return False

    def submit(self, rerun: bool = False, block=True):
        """Submit the job to slurm."""
        self._create_script()

        # check max jobs
        if block:
            while self._reach_max_jobs():
                print(
                    f"Max jobs {self.max_jobs} reached for partition {self.config['partition']}, waiting for jobs to finish..."
                )
                time.sleep(60)
        else:
            if self._reach_max_jobs():
                print(
                    f"Max jobs {self.max_jobs} reached for partition {self.config['partition']}, skip."
                )
                return None

        _run = True
        # submitted before
        if self._job_id_path.exists():
            with open(self._job_id_path) as f:
                running_job_id = f.read()
            job_id = running_job_id

            if (running_job_id != "") and (
                int(running_job_id) in self.get_running_job_ids()
            ):
                # still running
                print(
                    f"Job {self.config['job-name']} is already running in job id {running_job_id}"
                )
                _run = False
            else:
                # job finished
                if rerun:
                    self._job_id_path.unlink()
                else:
                    print(
                        f"Job {self.config['job-name']} submitted with job id {running_job_id} is finished, skip."
                    )
                    _run = False

        if _run:
            with open(self._job_id_path, "w") as f:
                f.write("")

            # sbatch and get the job id
            # sbatch output is like "Submitted batch job 12345678"
            try:
                time.sleep(np.random.randint(1, 5))
                process = subprocess.run(
                    ["sbatch", self._script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit job: {e}")
                print(e.stderr)
                raise e

            job_id = re.search(r"\d+", process.stdout).group()

            with open(self._job_id_path, "w") as f:
                f.write(job_id)
        return job_id

    @staticmethod
    def get_running_jobs(running_only: bool = True, mine_only: bool = True):
        """Get my running jobs from squeue."""
        process = subprocess.run(
            ["squeue", "--json"],
            capture_output=True,
            text=True,
        )
        output = process.stdout
        # load json
        squeue_data = json.loads(output)
        jobs = [SqueueJob(info_dict) for info_dict in squeue_data["jobs"]]

        if running_only:
            jobs = [job for job in jobs if job.running()]
        if mine_only:
            jobs = [job for job in jobs if job.mine()]
        return jobs

    @staticmethod
    def get_running_job_ids():
        """Get my running job ids from squeue."""
        jobs = SlurmManager.get_running_jobs()
        return [job.job_id for job in jobs]


def cancel_job(job_id):
    """Cancel a job by job id."""
    subprocess.run(["scancel", str(job_id)], capture_output=True, text=True)
    return


class PreemptibleManager:
    def __init__(
        self,
        command_list,
        job_name="job",
        time: str = "12:00:00",
        cpus_per_task: int = 16,
        mem_per_cpu: str = "4G",
        gpus: int = 0,
        output: str = "auto",
        error: str = "auto",
        exclude: str = "",
        nodelist: str = "",
        chdir: str = ".",
        max_jobs: int = 16,
    ):
        if isinstance(command_list, str):
            with open(command_list) as f:
                command_list = [l.strip() for l in f.readlines()]
        self.command_dict = dict(enumerate(command_list))
        print(f"Total {len(self.command_dict)} commands to submit.")
        self.job_name = job_name

        self.job_config = {
            "time": time,
            "partition": "preemptible",
            "cpus_per_task": cpus_per_task,
            "mem_per_cpu": mem_per_cpu,
            "gpus": gpus,
            "output": output,
            "error": error,
            "exclude": exclude,
            "nodelist": nodelist,
            "chdir": chdir,
            "max_jobs": max_jobs,
        }

    def submit(self):
        """Submit commands with preemptible jobs."""
        submitted_jobs = {}
        cum_job = 0

        while len(self.command_dict) > 0:
            print(f"Submitting {len(self.command_dict)} jobs in this round...")
            # go through all commands
            submitted_idx = set(submitted_jobs.values())
            for idx, command in self.command_dict.items():
                if idx in submitted_idx:
                    # already submitted
                    continue
                job_name = f"{self.job_name}_{cum_job}"
                cum_job += 1
                manager = SlurmManager(
                    job_name=job_name, command=command, **self.job_config
                )
                slurm_id = manager.submit(block=True)
                if slurm_id is not None:
                    submitted_jobs[str(slurm_id)] = idx
                else:
                    # reaches max jobs
                    break

            time.sleep(1200)

            print("Checking running jobs...")
            my_jobs = SlurmManager.get_running_jobs(running_only=False, mine_only=True)
            running_job_ids = {str(job.job_id) for job in my_jobs if job.running()}
            my_job_ids = {str(job.job_id) for job in my_jobs}
            print(f"Total {len(running_job_ids)} running jobs:", running_job_ids)
            for slurm_id, idx in submitted_jobs.items():
                if slurm_id not in my_job_ids:
                    # job disappeared, treat as finished
                    idx = submitted_jobs[str(slurm_id)]
                    self.command_dict.pop(idx, None)
