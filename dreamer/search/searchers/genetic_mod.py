import os
from typing import List, Optional

from ramanujantools.cmf import CMF

from dreamer.configs.system import sys_config
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM


class GeneticSearchMod(SearcherModScheme):
    """Searcher module that runs a GA-based trajectory search per searchable."""

    def __init__(
        self,
        searchables: List[Shard],
        use_LIReC: Optional[bool] = True,
        generations: int = 25,
        pop_size: int = 40,
        max_coord_init: int = 10,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.3,
        mutation_step: int = 1,
        crossover_prob: float = 0.5,
        max_retries: int = 3,
        refine_prob: float = 0.5,
        refine_coord_prob: float = 0.5,
        parallel_eval: bool = True,
    ):
        use_lirec_flag = bool(use_LIReC)
        super().__init__(
            searchables,
            use_lirec_flag,
            name="GeneticSearch",
            description="Genetic algorithm optimization module for trajectory discovery",
            version="1.0.0",
        )
        self.generations = generations
        self.pop_size = pop_size
        self.max_coord_init = max_coord_init
        self.elite_fraction = elite_fraction
        self.mutation_prob = mutation_prob
        self.mutation_step = mutation_step
        self.crossover_prob = crossover_prob
        self.max_retries = max_retries
        self.refine_prob = refine_prob
        self.refine_coord_prob = refine_coord_prob
        self.parallel_eval = parallel_eval

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        if not self.searchables:
            return

        export_root = sys_config.EXPORT_SEARCH_RESULTS or ""
        os.makedirs(
            dir_path := os.path.join(export_root, self.searchables[0].const.name),
            exist_ok=True,
        )

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in SmartTQDM(
                self.searchables,
                desc="Optimizing trajectories via GA: ",
                **sys_config.TQDM_CONFIG,
            ):
                searcher = GeneticSearchMethod(
                    space,
                    space.const,
                    generations=self.generations,
                    pop_size=self.pop_size,
                    max_coord_init=self.max_coord_init,
                    elite_fraction=self.elite_fraction,
                    mutation_prob=self.mutation_prob,
                    mutation_step=self.mutation_step,
                    crossover_prob=self.crossover_prob,
                    max_retries=self.max_retries,
                    refine_prob=self.refine_prob,
                    refine_coord_prob=self.refine_coord_prob,
                    parallel_eval=self.parallel_eval,
                    use_LIReC=self.use_LIReC,
                )
                res = searcher.search()

                if space.cmf.__class__ == CMF:
                    filename = f"generated_cmf_hashed_{hash(space.cmf)}"
                else:
                    filename = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in repr(space.cmf)).strip("_")
                write_chunk(res, filename)
