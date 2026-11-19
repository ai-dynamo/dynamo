# lib/router/session-affinity

`dynamo-session-affinity` is the session affinity stage: it binds a session to a worker and turns that binding into a routing constraint. Hosts outside `lib/router` import through `host`.
