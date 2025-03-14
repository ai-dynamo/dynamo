from disaggregated.frontend import Frontend
from disaggregated.kv_router import Router
from disaggregated.processor import Processor
from disaggregated.worker import VllmWorker
from disaggregated.prefill_worker import PrefillWorker

# example 2 and 3: kv aware routing + worker
# kv.yaml
Frontend.link(Processor).link(Router).link(VllmWorker)

# example 4 and 5: only disag - issue with endpoint (probably because of routerless)
# disag.yaml
# Frontend.link(VllmWorker).link(PrefillWorker)

# example 6: disag with kv
# kv_with_disag.yaml
#Frontend.link(Processor).link(Router).link(VllmWorker).link(PrefillWorker)
