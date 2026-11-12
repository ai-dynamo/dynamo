# lib/router/slot-tracker

`dynamo-slot-tracker` is the slot tracker stage: it records the requests booked on each worker rank and reports their load and fullness. Hosts outside `lib/router` import through `host`.
