import asyncio
import os
import time
from collections.abc import Callable
from threading import get_ident
from typing import Any, Dict, Optional

import pydantic
from starlette.background import BackgroundTasks
from starlette.responses import JSONResponse

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.inline.post_training.ilab.config import (
    ilabPostTrainingConfig,
)

# from llama_stack.providers.inline.post_training.ilab.recipes.lora_finetuning_single_device import (
#     LoraFinetuningSingleDevice,
# )
from llama_stack.schema_utils import webmethod


class TuningJob(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    task_uuid: str
    status: JobStatus
    subprocess: asyncio.subprocess.Process | None = None


def data_process(progress_callback: Callable[[JobStatus], None]):
    # If I make this async it'll go into the loop and block the critical event loop.
    # if non-async it'll go into another thread and we can run asyncio operations inside of a synchronous function
    progress_callback(JobStatus.in_progress)
    print(f"Running data processing from thread: ({get_ident()}) in pid: ({os.getpid()})")

    seconds = 10
    for i in range(seconds):
        time.sleep(1)
        print(f"waiting: {i}/{seconds - 1} from thread: ({get_ident()}) in pid: ({os.getpid()})")


async def subprocess_tuning(
    progress_callback: Callable[[JobStatus], None],
    tuning_process_callback: Callable[[asyncio.subprocess.Process], None],
):
    # this can be synchronous because we're awaiting everything that's interesting.

    proc_id = await asyncio.create_subprocess_shell("echo 'in a subprocess'; sleep 10; echo 'exiting subprocess'")
    tuning_process_callback(proc_id)

    await proc_id.wait()

    progress_callback(JobStatus.completed)


class ilabPostTrainingImpl:
    def __init__(
        self,
        config: ilabPostTrainingConfig,
        datasetio_api: DatasetIO,
        datasets: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets

        self.running_job: TuningJob | None = None
        self.checkpoints_dict = {}

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> JSONResponse:
        def callback_set_status(new_status: JobStatus):
            # closure for updating status of running job
            if self.running_job is None:
                return

            self.running_job.status = new_status

        def callback_set_process_ref(process_ref: asyncio.subprocess.Process):
            # closure for setting a reference to a subprocess
            if self.running_job is None:
                return

            self.running_job.subprocess = process_ref

        print(f"Starting background task from PID: ({os.getpid()}) in THREAD: ({get_ident()})")
        tasks = BackgroundTasks()
        tasks.add_task(func=data_process, progress_callback=callback_set_status)
        tasks.add_task(
            func=subprocess_tuning,
            progress_callback=callback_set_status,
            tuning_process_callback=callback_set_process_ref,
        )

        self.running_job = TuningJob(task_uuid=job_uuid, status=JobStatus.scheduled)

        response = PostTrainingJobStatusResponse(
            job_uuid="1234", status=JobStatus.scheduled.value, subprocess_running=False
        )

        return JSONResponse(content=response.model_dump(), background=tasks)

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        if self.running_job is not None:
            return ListPostTrainingJobsResponse(
                data=[
                    PostTrainingJob(
                        job_uuid=self.running_job.task_uuid,
                        current_status=self.running_job.status,
                    )
                ]
            )
        else:
            return ListPostTrainingJobsResponse(data=[])

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        # if self.running_job is not None:
        #     if self.running_job.proc.returncode == 0:
        #         self.running_job.status = JobStatus.completed

        #     if self.running_job.proc.returncode is not None and self.running_job.proc.returncode != 0:
        #         self.running_job.status = JobStatus.failed
        # return PostTrainingJobStatusResponse(job_uuid=self.running_job.job_uuid, status=self.running_job.status)

        print("Getting job status")

        if self.running_job:
            if self.running_job.subprocess is not None:
                print("Subprocess is not None")

                if self.running_job.subprocess.returncode is None:
                    print("returncode is None")
                    return PostTrainingJobStatusResponse(
                        job_uuid=self.running_job.task_uuid,
                        status=self.running_job.status.value,
                        subprocess_running=True,
                    )
                else:
                    print("returncode is not None")
                    return PostTrainingJobStatusResponse(
                        job_uuid=self.running_job.task_uuid,
                        status=self.running_job.status.value,
                        subprocess_running=False,
                        subrocess_return_code=self.running_job.subprocess.returncode,
                    )

            print("subprocess is None")
            return PostTrainingJobStatusResponse(
                job_uuid=self.running_job.task_uuid, status=self.running_job.status.value, subprocess_running=False
            )

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]: ...

    def _job_running(self):
        if self.running_job is not None:
            return_code = self.running_job.proc.returncode

            if return_code is None:
                return True

        # job is none OR returncode is not 0
        return False


if __name__ == "__main__":
    print("HELLO WORLD")
