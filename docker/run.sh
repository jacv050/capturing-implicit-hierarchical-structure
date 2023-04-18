#!/bin/bash
#docker run -d -it --rm \
#	--memory-swap 16384m \
#	--memory 16384m \

docker run -d -it --gpus '"device=0"' --rm \
	--volume=/mnt/md1/datasets/kinetics:/kinetics:rw \
	--volume=/home/jacastro/behavior/:/workspace:rw \
	--volume=/home/jacastro/virtualhome-to-behaviour/:/workspace/virtualhome-to-behaviour:rw \
	--volume=/mnt/md1/datasets/hyperfuture/weights/:/workspace/weights:rw \
	--volume=/home/jacastro/behavior/tmp_video_dir:/workspace/dataset/p1/test:rw \
	--volume=/mnt/md1/datasets/synthetic_herarchical:/workspace/dataset/bio_synthetic:rw \
	--name tontoelquelolea pytorch:behavior /bin/bash

#	--volume=/mnt/md1/datasets/kinetics/kinetics-downloader/dataset/prueba_behaviour:/workspace/dataset/p1/test:rw \
#	--volume=/home/jacastro/.ssh:/root/.ssh:ro \
