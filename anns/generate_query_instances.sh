#!/usr/bin/env bash

dim=$1
dis=$2

DIR="query_instance"

if [ -d "$DIR" ]; then
    rm -rf ${DIR}
fi

mkdir query_instance || true

cp *.h query_instance
cp *.cu query_instance
cp -R macro query_instance
cp -R graph_index query_instance
cp -R functions query_instance
cp -R graph_kernel_operation query_instance
cp -R hybrid query_instance
cp -R macro query_instance
cp -R previous query_instance
cp -R RVQ query_instance
cp -R utils query_instance
cp Makefile query_instance

cd query_instance

sed -i "s/PLACE_HOLDER_DIM/${dim}/g" graph_kernel_operation/kernel_local_graph_construction.h

if [ "${dis}" = "l2" ]; then
	make query mode=${mode}
elif [ "${dis}" = "cos" ]; then
	make query DISTTYPE=USE_COS_DIST_
elif [ "${dis}" = "ip" ]; then
	make query DISTTYPE=USE_IP_DIST_
fi

instance_name="query_${dim}_${dis}"

mv query ${instance_name}
cp ${instance_name} ..
rm ${instance_name}