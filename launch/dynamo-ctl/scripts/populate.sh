#!/bin/bash

# initialize some kv 
etcdctl get /dynamo/ --prefix
etcdctl put /dynamo/foo bar
etcdctl get /dynamo/ --prefix
sleep 2

# change it
etcdctl put /dynamo/foo somethingelse
sleep 2

# change it again
etcdctl put /dynamo/foo anotherthing
sleep 2

# add another thing
etcdctl put /dynamo/namespace/worker "running"
etcdctl get /dynamo/ --prefix
sleep 2

# cleanup
etcdctl del /dynamo/namespace/worker
etcdctl del /dynamo/foo
etcdctl get /dynamo/ --prefix

