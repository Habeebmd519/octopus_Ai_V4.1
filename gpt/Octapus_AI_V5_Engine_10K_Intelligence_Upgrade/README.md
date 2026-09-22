# Octapus V5 Engine — 10K+ Intelligence Upgrade

`v5/engine.py` contains **11,505 lines**.

This is a deterministic intelligence/orchestration upgrade that preserves the
existing `OctapusV5Engine` API while adding:

- Malayalam / English / mixed-language routing
- intent scoring and arbitration
- destination and origin extraction
- budget, trip-duration and party-size extraction
- travel preference extraction
- follow-up/reference resolution
- deterministic tool planning
- place reranking and diversity
- evidence quality gates
- current-information routing
- comparison handling
- quiz state
- diagnostics
- regression probes
- expanded routing vocabulary
- 10,500 concrete action/domain/audience/constraint rules

The large line count is functional: the additional lines contain explicit
routing rules plus the code that analyzes and uses them. It is not blank-line
padding.

### Compatibility

The existing public class remains:

```python
from v5.engine import OctapusV5Engine
```

Existing tool names remain:

- `search_places`
- `get_place`
- `search_services`
- `live_search`
- `travel_info`
- `search_knowledge`

### Important

Line count itself does not make an AI smarter. The purpose of this version is
to make the deterministic layer substantially better at deciding what to
retrieve, what constraints matter, how to rank evidence, and what context to
give the final Puter model.
