nats-server -js&
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 &

#dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
