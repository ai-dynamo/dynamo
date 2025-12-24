# Decode disagg working plan

NOTE: We will need api changes in the frameworks. This plan is _just_ for the core dynamo implementation such that it'll work on top of kv routing. Since long/short seqlen workers only take requests of their size, it doesn't make sense to share kv routing and each should get a distinct instance.

1. We need multiple workers, each with different sequence lengths
2. When a request exceeds the sequence length of a given worker, we want to migrate it to a new worker
3. When a request with a large sequence length comes in, we must route it to a decode worker that can handle it.


The structure should be like this:
1. instantiate 8k worker <- gets it's own kvrouter
2. instantiate 32k worker <- gets it's own kvrouter
3. Put a decode-disagger in front the kvrouter (here `servicebackend`)
```dynamo/lib/llm/src/entrypoint/input/common.rs:275
    let engine = frontend
        .link(preprocessor_op.forward_edge())?
        .link(migration.forward_edge())?
        .link(backend.forward_edge())?
        .link(prefill_op.forward_edge())?
        .link(decode_disagger.forward_edge())?
        .link(service_backend)?
        .link(decode_disagger.backward_edge())?
        .link(prefill_op.backward_edge())?
        .link(backend.backward_edge())?
        .link(migration.backward_edge())?
        .link(preprocessor_op.backward_edge())?
        .link(frontend)?;
```
4. annotate discovery model info with the correct metadata such that we can add smaller workers to a given deployment
    - will need to pass into workers --this-seqlen 8192 --max-seqlen 32768 and then wait for max-seqlen worker to actually finally publish the model deployment card, like how prefill/disagg don't publish on their own
5. for a first effort, we should wait until we see a sequence get to the max seqlen, then just send it back to prefill. Since we normally call decode first and that gets routed to prefill, this should be as simple as sending it to decode for the higher sequence length. Later we'll implement kv cache migration.


Note:
- no changes to prefill, just assume it's configured for the correct max seqlen