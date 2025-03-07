import pathlib
import time
import typing
from datetime import datetime
from typing import Any, Dict, Optional

import pydantic
from fastapi.concurrency import run_in_threadpool

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


class ilabFinetuningMultipleDevice:
    def __init__(self, config, model, dataset_id, datasets_api, datasetsio_api):
        self.config = config
        self.model = model
        self.dataset_id = dataset_id
        self.datasets_api = datasets_api
        self.datasetsio_api = datasetsio_api

    def _get_model_tokenizer(self):
        """
        NOTE:
        These processes can be pretty sensitive to problems.
        There ought to be a centralized log storage for all jobs w/ the log levels.
        Users should be able to inquire about the logs from a job that they ran w/ more details than "failed" or "completed".
        """
        return transformers.AutoTokenizerForCausalLM(self.model)

    async def _get_dataset_rows(self):
        rows = await self.datasetsio_api.get_rows_paginated(dataset_id=self.dataset_id, rows_in_page=-1)
        all_rows = rows.rows
        return all_rows

    @staticmethod
    def _create_hf_dataset_from_rows(sample_rows) -> "dataset":
        """Takes rows of data and stores them in a map-able hf 'dataset' object"""
        ...

    def _tokenize_all_rows(self, sample_rows):
        rows_dataset = self._create_hf_dataset_from_rows(sample_rows)
        tokenizer = self._get_model_tokenizer()
        all_rows_tokenized = rows_dataset.map(lambda x: tokenizer.tokenize(x))
        return all_rows_tokenized

    async def prepare_tuning_data(self, target_folder: pathlib.Path | None) -> pathlib.Path:
        # load all rows from API
        all_rows = await self._get_dataset_rows()

        # do any conversions necessary

        # render each row w/ template
        # tokenize each row
        all_rows_tokenized = self._tokenize_all_rows(sample_rows=all_rows)

        # write all rows to .jsonl in target_folder in tempfile
        ...


class TuningJob(pydantic.BaseModel):
    job_uuid: str
    proc: typing.Any
    status: JobStatus
    scheduled_at: datetime


def data_process():
    for i in range(30):
        time.sleep(1)
        print(f"waiting: {i}")


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
    ) -> PostTrainingJob:
        # new_proc = await asyncio.create_subprocess_shell(cmd=f"{sys.executable} {__file__}")

        # from datasetio, read all rows, tokenize data, write to disc as data-tokens.jsonl
        # in templfile that can be given to distributed processes.
        # this doesn't break the abstraction of the the datasetio class.

        # create class that marshalls the input configuration into a
        # torchrun command that should run neighbor script:  f"torchrun ... {pathlib.Path(__file__).parent / distributed_ilab_tune.py}"
        threadpool_val = await run_in_threadpool(data_process)

        self.running_job = TuningJob(
            job_uuid=job_uuid, proc=None, status=JobStatus.in_progress, scheduled_at=datetime.now()
        )

        post_training_job = PostTrainingJob(job_uuid=job_uuid)
        return post_training_job

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
            return ListPostTrainingJobsResponse(data=[PostTrainingJob(job_uuid=self.running_job.job_uuid)])
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

        return PostTrainingJobStatusResponse(job_uuid="1234", status=JobStatus.in_progress)

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        if self.running_job is not None:
            self.running_job.status = JobStatus.failed
            self.running_job.proc.terminate()

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
