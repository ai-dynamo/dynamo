# Service Metrics

This example extends the hello_world example by calling the `scrape_service` method
with the service name for the request response the client just issued a request.

```bash
DYN_LOG=debug cargo run --bin service_server
```

The client can now observe some basic statistics about each instance of the service
begin hosted.

```bash
DYN_LOG=info cargo run --bin service_client
```

## Example Output
```
Annotated { data: Some("h"), id: None, event: None, comment: None }
Annotated { data: Some("e"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some(" "), id: None, event: None, comment: None }
Annotated { data: Some("w"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some("r"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("d"), id: None, event: None, comment: None }
ServiceSet { services: [ServiceInfo { name: "dynamo_init_backend_720278f8", id: "eOHMc4ndRw8s5flv4WOZx7", version: "0.0.1", started: "2025-02-26T18:54:04.917294605Z", endpoints: [EndpointInfo { name: "dynamo_init_backend_720278f8-generate-694d951a80e06abf", subject: "dynamo_init_backend_720278f8.generate-694d951a80e06abf", data: Some(Metrics(Object {"average_processing_time": Number(53662), "data": Object {"val": Number(10)}, "last_error": String(""), "num_errors": Number(0), "num_requests": Number(2), "processing_time": Number(107325), "queue_group": String("q")})) }] }] }
```

Note the following stats in the output demonstrate the custom
`stats_handler` attached to the service in `server.rs` is being invoked:
```
data: Some(Metrics(Object {..., "data": Object {"val": Number(10)}, ...)
```

If you start two copies of the server, you will see two entries being emitted.
