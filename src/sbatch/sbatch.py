import json
import os
import pathlib
import re
import subprocess

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


class SlurmManager:
    def __init__(
        self,
        job_name: str,
        command: str,
        time: str = "12:00:00",
        partition: str = "gpu",
        nodes: int = 1,
        ntasks: int = 1,
        cpus_per_task: int = 32,
        mem_per_cpu: str = "4G",
        gres: str = "gpu:1",
        output: str = "auto",
        error: str = "auto",
        exclude: str = "",
        nodelist: str = "",
        chdir: str = ".",
    ):
        self.job_type = "gpu" if "gpu" in partition.lower() else "cpu"
        self.command = command
        self.config = {
            "job-name": job_name,
            "time": time,
            "partition": partition,
            "nodes": nodes,
            "ntasks": ntasks,
            "cpus-per-task": cpus_per_task,
            "mem-per-cpu": mem_per_cpu,
            "gres": gres,
            "output": output,
            "error": error,
            "exclude": exclude,
            "nodelist": nodelist,
            "chdir": chdir,
        }
        if self.job_type == "cpu":
            self.config["gres"] = ""

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

    def submit(self, rerun: bool = False):
        """Submit the job to slurm."""
        self._create_script()

        _run = True
        # submitted before
        if self._job_id_path.exists():
            with open(self._job_id_path) as f:
                running_job_id = f.read()
            job_id = running_job_id

            if int(running_job_id) in self.get_running_job_ids():
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
            process = subprocess.run(
                ["sbatch", self._script_path],
                capture_output=True,
                text=True,
            )
            print(process.stdout)
            job_id = re.search(r"\d+", process.stdout).group()

            with open(self._job_id_path, "w") as f:
                f.write(job_id)
        return job_id

    @staticmethod
    def get_running_jobs():
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

        my_running_job = [job for job in jobs if job.mine() and job.running()]
        return my_running_job

    @staticmethod
    def get_running_job_ids():
        """Get my running job ids from squeue."""
        jobs = SlurmManager.get_running_jobs()
        return [job.job_id for job in jobs]
