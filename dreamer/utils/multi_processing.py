from concurrent.futures import ProcessPoolExecutor
from dreamer.configs import config
from multiprocessing import Pool as Pool
import multiprocessing as mp
import queue
import json
from dataclasses import asdict
from logger import Logger


def _init_worker(config_overrides):
    from dreamer.configs import config
    config.configure(**config_overrides)


def create_process_pool_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(initializer=_init_worker, initargs=(config.export_configurations(),))

def create_pool():
    return Pool(initializer=_init_worker, initargs=(config.export_configurations(),))


# You will import your DTOs and the TrajectoryAttributesHandler here
# from dreamer.utils.storage.dtos import TrajectoryDTO
# from trajectory_attributes import TrajectoryAttributesHandler

def background_attribute_worker(
        worker_id: int,
        task_queue: mp.Queue,
        results_queue: mp.Queue
):
    """
    Consumer Process: Pulls lightweight DTOs, reconstructs the math,
    computes heavy attributes, and pushes the enriched DTO.
    """
    Logger(f"Worker {worker_id} started.", Logger.Levels.debug).log()

    while True:
        try:
            # Block until a DTO is available.
            # Timeout allows graceful shutdown checks if needed.
            dto = task_queue.get(timeout=3)

            # The Sentinel pattern: If we receive None, it means the run is over.
            if dto is None:
                Logger(f"Worker {worker_id} received shutdown signal.", Logger.Levels.debug).log()
                break

            # 1. Reconstruct the heavy math using only the parameters in the DTO
            # handler = TrajectoryAttributesHandler.from_parameters(...)

            # 2. Compute Stage 2 metrics (e.g., sorted_eigenvalues, kamidelta)
            # stage_2_results = handler.compute_heavy_metrics()

            # 3. For now, just simulate adding data to the DTO's extended_metrics
            dto.extended_metrics["worker_id"] = worker_id
            dto.extended_metrics["processed"] = True

            # 4. Push the enriched DTO to the dedicated writer
            results_queue.put(dto)

        except queue.Empty:
            continue
        except Exception as e:
            # Ensure a single crashing trajectory doesn't kill the worker
            Logger(f"Worker {worker_id} encountered an error: {e}", Logger.Levels.debug).log()


def dedicated_file_writer(results_queue: mp.Queue, output_file_path: str):
    """
    Sink Process: The ONLY process allowed to write to the disk.
    Prevents file lock collisions.
    """
    Logger("Dedicated Writer started.", Logger.Levels.debug).log()

    # We use JSON Lines (.jsonl) as it allows safe, fast appending
    # of flat dictionaries without loading the whole file into memory.
    with open(output_file_path, 'a') as f:
        while True:
            dto = results_queue.get()

            if dto is None:
                Logger("Writer received shutdown signal.", Logger.Levels.debug).log()
                break

            # Convert the frozen dataclass to a standard dictionary
            dto_dict = asdict(dto)

            # Write a single line of JSON
            f.write(json.dumps(dto_dict) + '\n')
            f.flush()  # Ensure it hits the disk immediately
