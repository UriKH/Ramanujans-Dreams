from concurrent.futures import ProcessPoolExecutor
from dreamer.configs import config
from multiprocessing import Pool as Pool


def _init_worker(config_overrides):
    from dreamer.configs import config
    config.configure(**config_overrides)


def create_process_pool_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(initializer=_init_worker, initargs=(config.export_configurations(),))

def create_pool():
    return Pool(initializer=_init_worker, initargs=(config.export_configurations(),))
