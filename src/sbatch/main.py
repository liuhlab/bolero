import click

from .sbatch import SlurmManager


@click.command()
@click.option("-n", "--job-name", required=True, help="Job name")
@click.option("--time", default="12:00:00", help="Time limit", show_default=True)
@click.option("--partition", default="cpu", help="Partition", show_default=True)
@click.option("--cpus-per-task", default=32, help="CPUs per task", show_default=True)
@click.option("--mem-per-cpu", default="4G", help="Memory per CPU", show_default=True)
@click.option("--gpus", default=1, help="gpus", show_default=True)
@click.option("--output", default="auto", help="Output file", show_default=True)
@click.option("--error", default="auto", help="Error file", show_default=True)
@click.option("--exclude", default="", help="Exclude nodes", show_default=True)
@click.option("--nodelist", default="", help="Node list", show_default=True)
@click.option("--chdir", default=".", help="Change directory", show_default=True)
@click.option("--rerun", is_flag=True, help="Rerun job", show_default=True)
@click.option(
    "--max_jobs",
    default=16,
    help="Max number of jobs to submit to current partition",
    show_default=True,
)
@click.argument("command", nargs=-1)
def submitter(
    job_name,
    time,
    partition,
    cpus_per_task,
    mem_per_cpu,
    gpus,
    output,
    error,
    exclude,
    nodelist,
    chdir,
    rerun,
    command,
    max_jobs=16,
):
    """Submit a job to slurm"""
    manager = SlurmManager(
        job_name=job_name,
        command=" ".join(command),
        time=time,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_per_cpu=mem_per_cpu,
        gpus=gpus,
        output=output,
        error=error,
        exclude=exclude,
        nodelist=nodelist,
        chdir=chdir,
        max_jobs=max_jobs,
    )
    _job_id = manager.submit(rerun=rerun)
    return
