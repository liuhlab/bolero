import inspect
import os
import pathlib
import re
import shutil

import nbformat as nbf
import pandas as pd
import yaml


def papermill_notebook(
    input_path, output_path, cwd, config_path, log_path, success_flag
):
    """Run papermill and ignore error if flag set."""
    import pathlib
    import subprocess
    import time

    ignore_path = pathlib.Path(cwd) / "log/ignore_error"
    if ignore_path.exists():
        print(f"Skip notebook {output_path} due to ignore flag.")
        subprocess.run(f"touch {success_flag}", shell=True, check=True)
        return

    try:
        subprocess.run(
            "papermill "
            f"--cwd {cwd} "
            f"-f {config_path} "
            f"{input_path} "
            f"{output_path} "
            f"> {log_path} 2>&1 "
            f"&& touch {success_flag}",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        time.sleep(5)
        if ignore_path.exists():
            print(f"Error ignored for {output_path}.")
            subprocess.run(f"touch {success_flag}", shell=True, check=True)
            return
        else:
            print(f"Ignore file: {log_path} not exist, raise error.")
            raise e
    return


def _extract_nb_parameters(nb_path, **default_kwargs):
    """
    Extrackt parameters from a notebook.

    Parameter cells are identified by the tag "parameters", the same as papermill convention.

    Parameters
    ----------
    nb_path : str
        Path to the notebook.
    default_kwargs : dict
        Default parameters, any parameters in the notebook will override the default parameters.

    Returns
    -------
    dict
        Parameters extracted from the notebook.
    """
    ntbk = nbf.read(nb_path, nbf.NO_CONVERT)
    parameters = default_kwargs
    for cell in ntbk.cells:
        tags = cell["metadata"].get("tags", [])
        if "parameters" in tags:
            # execute parameters cell code, and put the variables into parameters
            exec(cell.source, None, parameters)
            break
    return parameters


def _parse_workdir_notebooks(notebook_dir, default_cpu=1, default_mem_gb=1):
    """
    Parse notebook directory and extract parameters from each notebook.

    Notebook name pattern example:
    - 01.step1.ipynb
    - 02a.step2a.ipynb
    - 02b.any_name.ipynb
    - 03.ipynb  # or no name

    This example will be unserstand as a three step notebook template,
    01.step1.ipynb is the first step,
    02a.step2a.ipynb and 02b.any_name.ipynb are the second step, which will be executed in parallel,
    03.ipynb is the third step.

    Parameters
    ----------
    notebook_dir : str
        Directory containing a series of template notebooks with name suffix indicating the execution order.
        Notebook
    """
    # notebook name pattern
    nb_name_pattern = re.compile(r"(?P<num_step>\d+)(?P<sub_step>\w*).*ipynb")
    # example names:
    # 01.step1.ipynb
    # 02a.step2a.ipynb
    # 02b.any_name.ipynb
    # 03.ipynb  # or no name

    nb_records = {}
    for path in pathlib.Path(notebook_dir).glob("*.ipynb"):
        try:
            match = nb_name_pattern.match(path.name).groupdict()
            nb_records[(int(match["num_step"]), match["sub_step"])] = {
                "path": path.resolve().absolute(),
                "parameters": _extract_nb_parameters(
                    path, cpu=default_cpu, mem_gb=default_mem_gb
                ),
            }
        except AttributeError:
            # not match, ignore this notebook
            continue

    nb_records = pd.DataFrame(nb_records).T.reset_index()
    nb_records.columns = ["num_step", "sub_step"] + nb_records.columns[2:].tolist()
    nb_records = nb_records.loc[
        :, ["num_step", "sub_step", "path", "parameters"]
    ].copy()
    return nb_records


def notebook_snakemake(
    work_dir,
    notebook_dir,
    groups,
    group_files=None,
    default_cpu=1,
    default_mem_gb=1,
    redo_prepare=False,
):
    """
    Prepare snakemake file for running a series of notebooks in multiple groups.

    Parameters
    ----------
    work_dir :
        Working directory for snakemake.
    notebook_dir :
        Directory containing a series of template notebooks with name suffix indicating the execution order.
    groups :
        A list of groups, each group will have its own sub-dir in the work_dir, and run through the notebook series.
        If a single file path is given, each row in the file will be treated as a group.
    group_files :
        A dict of file paths, file paths as key, group name as value. This parameter allows providing
        group-specific file. Each file will be copied to the corresponding group sub-dir.
    default_cpu :
        Default cpu for each notebook.
    default_mem_gb :
        Default memory for each notebook.
    redo_prepare :
        If True, will re-generate the snakemake file even if it already exists.
    """
    # TODO: how to allow group specific resources?
    #     group_resources :
    #         A dict of dicts, group name as key, and a dict of resources as value. This parameter allows group-specific
    #         resource allocation, will override the default resource allocation or the resource allocation specified in
    #         the notebook.
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    work_dir = work_dir.resolve().absolute()
    snakefile_path = work_dir / "Snakefile"
    _local_notebook_dir = work_dir / "notebooks"
    _local_notebook_dir.mkdir(exist_ok=True)

    # copy all notebooks into _local_notebook_dir
    for path in pathlib.Path(notebook_dir).glob("*.ipynb"):
        shutil.copy(path, _local_notebook_dir)
    notebook_dir = _local_notebook_dir

    # chdir to work_dir
    previous_cwd = os.getcwd()
    os.chdir(work_dir)

    if snakefile_path.exists() and not redo_prepare:
        print(f"Snakefile already exists at {snakefile_path}, skip.")
        return

    if isinstance(groups, (str, pathlib.Path)):
        with open(groups) as f:
            groups = [line.strip() for line in f]

    nb_records = _parse_workdir_notebooks(
        notebook_dir=notebook_dir,
        default_cpu=default_cpu,
        default_mem_gb=default_mem_gb,
    )

    rule_template = """

rule {rule_name}:
    input:
        {input_pattern},
    params:
        input_nb="{nb_path}"
    output:
        "{{group_name}}/log/{rule_name}.success"
    log:
        output_nb="{{group_name}}/{nb_name}",
        log="{{group_name}}/log/{nb_name}.log"
    threads:
        {cpu}
    resources:
        mem_gb={mem_gb}
    run:
        papermill_notebook(
            input_path=params.input_nb,
            output_path=log.output_nb,
            cwd=wildcards.group_name,
            config_path=str(wildcards.group_name) + "/log/{rule_name}.config.yaml",
            log_path=log.log,
            success_flag=output
        )

"""

    snakemake_str = ""
    # notebook rules
    previous_rule_names = None
    rule_name = None
    for num_step, sub_df in nb_records.groupby("num_step"):
        step_rule_names = []
        for _, row in sub_df.iterrows():
            _, sub_step, nb_path, parameters = row

            rule_name = f"step_{num_step}{sub_step}"
            nb_name = pathlib.Path(nb_path).name
            if previous_rule_names is None:
                input_pattern = f'"{snakefile_path}"'
            else:
                input_files = []
                for _rn in previous_rule_names:
                    input_files.append(f'"{{group_name}}/log/{_rn}.success"')
                input_pattern = ", ".join(input_files)
            step_rule_names.append(rule_name)

            # resource parameters
            cpu = parameters.get("cpu", default_cpu)
            mem_gb = parameters.get("mem_gb", default_mem_gb)

            rule_str = rule_template.format(
                rule_name=rule_name,
                input_pattern=input_pattern,
                nb_path=nb_path,
                nb_name=nb_name,
                cpu=cpu,
                mem_gb=mem_gb,
            )
            snakemake_str += rule_str
        previous_rule_names = step_rule_names

    # summary rule
    final_rule_str = f"""
groups = {groups}

# the notebook execution function
{inspect.getsource(papermill_notebook)}

rule final:
    input:
        # target the final rule success flag of each group
        expand('{{group_name}}/log/{rule_name}.success', group_name=groups)
    output:
        "Snakemake.success"
    shell:
        # create a final success flag
        "touch {{output}}"

"""
    snakemake_str = final_rule_str + snakemake_str

    with open(snakefile_path, "w") as f:
        f.write(snakemake_str)

    # write parameters
    for group in groups:
        group_log_dir = pathlib.Path(f"{group}/log")
        group_log_dir.mkdir(exist_ok=True, parents=True)

        for _, (num_step, sub_step, _, parameters) in nb_records.iterrows():
            parameters["group_name"] = group
            config_path = group_log_dir / f"step_{num_step}{sub_step}.config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(parameters, f)

        # copy group files
        if group_files is not None:
            for file_path, g in group_files.items():
                file_path = pathlib.Path(file_path)
                shutil.copy(file_path, f"{g}/")

    os.chdir(previous_cwd)
    return
