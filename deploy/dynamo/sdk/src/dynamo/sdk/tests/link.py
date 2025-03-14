# linking syntax example


from dynamo.sdk.tests.pipeline import Backend, Frontend, Middle

# print("INITIAL DEPENDENCIES")
# print("Frontend dependencies", Frontend.dependencies)
# print("Middle dependencies", Middle.dependencies)
# print("Backend dependencies", Backend.dependencies)

# print("\n\n\n")

print()
Frontend.link(Middle).build()

"""
---------------
class FrontEnd:
    decode_worker = depends(DecodeWorker)
    preproc = depends(Preproc)              # optional

class Processor:
    kv_router = depends(KvRouter)

class KvRouter:
    decode_worker = depends(DecodeWorker)

class DecodeWorker:
    prefill_worker = depends(PrefillWorker) # optional
---------------

# Optional
Frontend -> Preproc

if cli.enable_prefill:
    Frontend -> Preproc -> KvRouter

if cli.enable_prefill:
    DecodeWorker -> PrefillWorker
-----------

Monolith
Frontend.link(DecodeWorker).build()

Kv aware monolith
Frontend.link(Processor).link(DecodeWorker).build()
Frontend.link(KvRouter)

Kv off + Disag on
Frontend.link(DecodeWorker).link(PrefillWorker).build()
if kv_disabled:
    Processor.unlink(KvRouter)

Kv on + Disag On
Frontend.link(Processor)

"""

print("Frontend dependencies", Frontend.dependencies)
print("Middle dependencies", Middle.dependencies)
print("Backend dependencies", Backend.dependencies)
