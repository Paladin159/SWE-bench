"""
This script implements run_evaluation using Morph Cloud instead of Modal.
It follows the same steps as run_evaluation_modal.py: building a snapshot
via a chain of .setup() commands (including repository cloning and checkout
using the environment_setup_commit), starting an instance, applying a patch,
running tests and generating a report.
"""

import os
import time
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
from typing import cast

from morphcloud.api import MorphCloudClient
from swebench.harness.docker_build import setup_logger
from swebench.harness.reporting import make_run_report
from swebench.harness.utils import EvaluationError
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS, RUN_EVALUATION_LOG_DIR

@dataclass
class TestOutput:
    instance_id: str
    test_output: str
    report_json_str: str
    run_instance_log: str
    patch_diff: str
    log_dir: Path
    errored: bool

@dataclass
class TestOutput:
    instance_id: str
    test_output: str
    report_json_str: str
    run_instance_log: str
    patch_diff: str
    log_dir: Path
    errored: bool
    
client = MorphCloudClient()

@asynccontextmanager
async def base_snapshot_context(test_spec: TestSpec):
    """
    Build and yield a base snapshot that contains all common installation steps.
    These steps run once and are cached.
    """
    snapshot = client.snapshots.create(
        vcpus=4,
        memory=8192,
        disk_size=20000,
        digest="swebench-base"
    )
    # Common steps executed once
    snapshot = ( 
        snapshot.setup("apt update -y")
        .setup("export DEBIAN_FRONTEND=noninteractive && export TZ='Etc/UTC'")
        .setup("apt install -y wget git build-essential libffi-dev libtiff-dev jq curl locales locales-all tzdata patch")
    # Install Miniconda
        .setup("wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh")
        .setup("bash miniconda.sh -b -p /opt/miniconda3")
        .setup("echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc")
        .setup("/opt/miniconda3/bin/conda init --all")
        .setup("/opt/miniconda3/bin/conda config --append channels conda-forge")
        .setup("adduser --disabled-password --gecos 'dog' nonroot")
        .setup("mkdir -p /testbed")
    )
    env_script = test_spec.setup_env_script
    if env_script:
        snapshot = (
            snapshot.setup(f"cat <<'EOF' > /root/setup_env.sh\n{env_script}\nEOF")
            .setup("chmod +x /root/setup_env.sh")
            .setup("bash -c 'source ~/.bashrc && /root/setup_env.sh'")
            .setup("echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc")
        )
        # Inline the repository installation script from TestSpec.
    repo_script = test_spec.install_repo_script
    if repo_script:
        snapshot = (
            snapshot.setup(f"cat <<'EOF' > /root/setup_repo.sh\n{repo_script}\nEOF")
            .setup("chmod +x /root/setup_repo.sh")
            .setup("bash /root/setup_repo.sh")
        )
    with client.instances.start(snapshot.id, ttl_seconds=3600) as instance:
         try:
            yield instance
         finally:
            pass

async def get_log_dir(pred: dict, run_id: str, instance_id: str) -> Path:
    model_name_or_path = cast(
        str, pred.get("model_name_or_path", "None").replace("/", "__")
    )
    return RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

async def process_instance(test_spec: TestSpec, pred: dict, run_id: str, timeout: int) -> TestOutput:
    """
    Do the remaining work (patch application, running eval, logging, reporting)
    on the Morph Cloud instance yielded by base_snapshot_context.
    """
    instance_id = test_spec.instance_id
    # Setup logging directory:
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / test_spec.repo.replace("/", "__") / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_instance.log"
    # Retrieve any patch diff from the prediction:
    patch_diff = pred.get("model_patch", "")
    try:
         # Use the common instance from base_snapshot_context.
        async with base_snapshot_context(test_spec) as morphvm:
            # If there's a patch diff, write it to /tmp/patch.diff and try to apply it.
            if patch_diff:
                # Write patch file
                await morphvm.aexec(command=f"bash -c 'echo \"{patch_diff}\" > /tmp/patch.diff'")
                # Attempt to apply patch via git
                apply_patch_resp = await morphvm.aexec(command="cd /testbed && git apply -v /tmp/patch.diff")
                if apply_patch_resp.exit_code != 0:
                    # Fallback to using patch command.
                    apply_patch_resp = await morphvm.aexec(command="cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff")
                    if apply_patch_resp.exit_code != 0:
                        raise Exception( f"Patch failed:\n{apply_patch_resp.stdout}\n{apply_patch_resp.stderr}")
            # Get git diff before running the evaluation script
            git_diff_before_resp = await morphvm.aexec(command="cd /testbed && git diff")
            git_diff_before = git_diff_before_resp.stdout
            # Write the evaluation script to /root/eval.sh and make it executable.
            await morphvm.aexec(command=f"bash -c 'echo \"{test_spec.eval_script}\" > /root/eval.sh && chmod +x /root/eval.sh'")
            # Run evaluation: increase recursion limit and execute the eval script.
            start_time = time.time()
            run_command = (
                "cd /testbed && python3 -c 'import sys; sys.setrecursionlimit(10000)' "
                "&& /bin/bash /root/eval.sh"
            )
            eval_resp = await morphvm.aexec(command=run_command)
            test_output = eval_resp.stdout
            total_runtime = time.time() - start_time
            # Get git diff after running the evaluation.
            git_diff_after_resp = await morphvm.aexec(command="cd /testbed && git diff")
            # Write test output to log file.
            test_output_path = log_dir / "test_output.txt"
            with open(test_output_path, "w", encoding="utf-8") as f:
                f.write(test_output)
            # Get and log the git diff after running the eval script.
            git_diff_after_resp = await morphvm.aexec(command="cd /testbed && git diff")
            git_diff_after = git_diff_after_resp.stdout
            report = get_eval_report(
                test_spec=test_spec,
                prediction=pred,
                test_log_path=test_output_path,
                include_tests_status=True,
            )
            return TestOutput(
                instance_id=test_spec.instance_id,
                test_output=test_output,
                report_json_str=json.dumps(report, indent=4),
                run_instance_log=log_file.read_text(),
                patch_diff=patch_diff,
                log_dir=log_dir,
                errored=False,
            )
    except Exception:
        error_msg = traceback.format_exc()
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(error_msg)
        return TestOutput(
            instance_id=test_spec.instance_id,
            test_output="",
            report_json_str="",
            run_instance_log=log_file.read_text(),
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=True,
        )

async def process_instances_distributed(predictions: dict, instances: list, full_dataset: list, run_id: str, timeout: int):
    """
    Create an async queue over the test specifications and run each instance on Morph Cloud.
    """
    test_specs = [make_test_spec(instance) for instance in instances if instance["instance_id"] in predictions]
    tasks = []
    for test_spec in test_specs:
        pred = predictions[test_spec.instance_id]
        tasks.append(process_instance(test_spec, pred, run_id, timeout))
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results