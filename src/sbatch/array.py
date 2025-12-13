import pathlib

ARRAY_SCRIPT = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{n_cmd}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --gpus={gpus}
#SBATCH --output={script_dir}/%A_%a.out
#SBATCH --error={script_dir}/%A_%a.err
#SBATCH --chdir={chdir}

# Get the command for this array index
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {command_path})

echo "Running task $SLURM_ARRAY_TASK_ID:"
echo "$CMD"

# Execute the command
eval "$CMD"
"""


def prepare_array(
    commands: list[str],
    job_name: str = "array",
    command_chunk_size: int | None = None,
    script_dir: str = "sbatch",
    partition: str = "preemptible",
    cpus_per_task: int = 8,
    mem_per_cpu: str = "8G",
    time: str = "1-00:00:00",
    gpus: int = 0,
    chdir: str = ".",
    force: bool = False,
    max_jobs: int = 32,
) -> str:
    """
    Prepare an array script for slurm.
    The command file should be a list of commands, one per line.

    Parameters
    ----------
    script_path : str
        The path to the script to write.
    command_path : str
        The path to the command file to read.
    job_name : str, optional
        The name of the job (default is "array").
    partition : str, optional
        The partition to run the job on (default is "preemptible").
    cpus_per_task : int, optional
        The number of CPUs per task (default is 8).
    mem_per_cpu : str, optional
        The memory per CPU (default is "8G").
    time : str, optional
        The time limit for the job (default is "1-00:00:00").
    gpus : int, optional
        The number of GPUs to use (default is 0).
    chdir : str, optional
        The directory to change to (default is ".").
    force : bool, optional
        Whether to force overwrite the script and command file if they already exist.
    max_jobs : int, optional
        The maximum concurrent jobs to submit via sbatch --array.

    Returns
    -------
    str
        The path to the script.
    """
    if len(commands) == 0:
        print(f"No commands to submit for job {job_name}, skipping.")
        return

    script_dir = pathlib.Path(script_dir).absolute().resolve()
    script_dir.mkdir(exist_ok=True, parents=True)

    command_path = script_dir / f"{job_name}.commands.txt"
    script_path = script_dir / f"{job_name}.array.sh"

    if not force:
        assert not script_path.exists(), f"Script already exists: {script_path}, please set force=True or use a different job name."
        assert not command_path.exists(), f"Command file already exists: {command_path}, please set force=True or use a different job name."

    if command_chunk_size is not None:
        before_len = len(commands)
        commands = [
            "; ".join(commands[cs : cs + command_chunk_size])
            for cs in range(0, len(commands), command_chunk_size)
        ]
        after_len = len(commands)
        print(f"Split {before_len} commands into {after_len} chunks.")

    with open(command_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
    n_cmd = len(commands) - 1
    if max_jobs is not None:
        n_cmd_str = f"{n_cmd}%{max_jobs}"
    else:
        n_cmd_str = str(n_cmd)

    script = ARRAY_SCRIPT.format(
        job_name=job_name,
        n_cmd=n_cmd_str,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_per_cpu=mem_per_cpu,
        time=time,
        gpus=gpus,
        chdir=chdir,
        script_dir=script_dir,
        command_path=command_path,
    )
    with open(script_path, "w") as f:
        f.write(script.lstrip())

    print(f"Creating sbatch array job {job_name} with {n_cmd} sub-jobs.")
    print("=" * 50)
    print(f"Command file: {command_path}")
    print(f"Script: {script_path}")
    print(f"partition: {partition}")
    print(f"cpus_per_task: {cpus_per_task}")
    print(f"mem_per_cpu: {mem_per_cpu}")
    print(f"time: {time}")
    print(f"gpus: {gpus}")
    print(f"chdir: {chdir}")
    print("=" * 50)

    print("To submit the job, run:")
    print(f"sbatch {pathlib.Path(script_path).absolute()}")
    return
