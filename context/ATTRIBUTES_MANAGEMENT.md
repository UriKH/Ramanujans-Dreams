## Attributes Management Plan

> **Doc type:** Design intent (planning snapshot). Captures the *why* behind
> the attributes-management subsystem; the implementation has since moved on.
> For *what the code does today* see
> [`dreamer/utils/storage/attribute_registry.py`](../dreamer/utils/storage/attribute_registry.py)
> and [`dreamer/utils/storage/trajectory_attributes.py`](../dreamer/utils/storage/trajectory_attributes.py).

This document outlines the current and planned strategies for managing trajectory and recurrence relation attributes in `dreamer.utils.storage` and their eventual integration with the CMF Atlas database.

### Current State

- `dreamer.utils.storage.trajectory_attributes.py`: A collection of functions to compute various attributes for trajectories.
- `dreamer.utils.storage.dtos.py`: Data Transfer Objects (DTOs) for storing and transferring trajectory data planning an hierarchy of the DTOS.

### Goal

- Compute attributes for each trajectory or recurrence relation gradually during the pipeline pass.
- Store computed attributes gradually during the pipeline pass. In a format that could be easily later be converted into DB data insertion operation (highly connected to the idea of a CMF Atlas database).
- We would like to be able to refrain from recomputation and check for a trajectory if it was already computed.
- We would like to append data and update attributes in the files containing the data about CMFs, Shards, trajectories etc. and not override it. This will be even simpler and easier later when we move to a DB and not rely on file storage (jsonl or similar formats).

### Ideas

**These are suggestions and impelementation details that could be changed!**

- In the Loading stage we will store data relevant to the the CMF families.
- In the Analysis stage we will know which constant we found in each CMF and could store CMF data along with full shard information.
- In the Search stage we will know which trajectories are of interest and compute minimal information regrading them (initial point, direction, $delta$, recurrence relation, trajectory matrix). The data could be loaded in a queue which will be accessed in parallel and compute more relevant attributes regarding the trajectory (e.g. regarding the recurrence relation).
- Implementing a multiple producers multiple consumers pattern to be utilized in the attribute computation process in search. The producers will provide the basic data about trajectory or recurrence relation, and the consumers will compute more attributes for it. Meanwhile a sink process could preform the write to memory (jsonl files per say) - writing the extra data created by the consumers.

### Consideratiosns

- Keep in mind that we cannot hold full gigabaytes of data in RAM mid run.
- Use the fact that each Shard is proccessed separately.
- Later we will evolve this implemenation into a DB based one (not files).
