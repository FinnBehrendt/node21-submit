# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# FROM pytorch/pytorch:latest
FROM continuumio/anaconda3
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm/checkpoints/ /input /output /opt/algorithm/src/ /opt/conda  \
    && chown algorithm:algorithm /opt/algorithm/checkpoints /input /output /opt/algorithm/src /opt/conda 

RUN mkdir -p /opt/algorithm/checkpoints/retrain/yolo_s \
    /opt/algorithm/checkpoints/retrain/yolo_l \
    /opt/algorithm/checkpoints/retrain/effdet_s \
    /opt/algorithm/checkpoints/retrain/effdet_l \
    /opt/algorithm/checkpoints/retrain/fcrnn_s \
    /opt/algorithm/checkpoints/retrain/fcrnn_l \
    /opt/algorithm/checkpoints/retrain/retina_s \
    /opt/algorithm/checkpoints/retrain/retina_l \
    /opt/algorithm/wandb/ \
    && chown algorithm:algorithm  /opt/algorithm/checkpoints/retrain/yolo_l \
    /opt/algorithm/checkpoints/retrain/effdet_s \
    /opt/algorithm/checkpoints/retrain/effdet_l \
    /opt/algorithm/checkpoints/retrain/fcrnn_s \
    /opt/algorithm/checkpoints/retrain/fcrnn_l \
    /opt/algorithm/checkpoints/retrain/retina_s \
    /opt/algorithm/checkpoints/retrain/retina_l \
    /opt/algorithm/wandb/
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"
RUN python -m pip install --user -U pip
RUN pip install --upgrade pip
USER root
RUN apt-get update 
RUN apt-get install build-essential -y
RUN apt-get install libtinfo5
USER algorithm



# Copy all required files so that they are available within the docker image 
# All the codes, weights, anything you need to run the algorithm!
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
RUN python -m pip install --user -rrequirements.txt
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm checkpoints/ /opt/algorithm/checkpoints/
# COPY --chown=algorithm:algorithm resnet50-19c8e357.pth  /home/algorithm/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
# COPY --chown=algorithm:algorithm fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /home/algorithm/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
# COPY --chown=algorithm:algorithm retinanet_resnet50_fpn_coco-eeacb38b.pth /home/algorithm/.cache/torch/hub/checkpoints/retinanet_resnet50_fpn_coco-eeacb38b.pth
COPY --chown=algorithm:algorithm resnet50-0676ba61.pth /home/algorithm/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
# COPY --chown=algorithm:algorithm tf_efficientdet_d3_ap-e4a2feab.pth /home/algorithm/.cache/torch/hub/checkpoints/tf_efficientdet_d3_ap-e4a2feab.pth
# COPY --chown=algorithm:algorithm tf_efficientdet_d4_ap-f601a5fc.pth /home/algorithm/.cache/torch/hub/checkpoints/tf_efficientdet_d4_ap-f601a5fc.pth 
# COPY --chown=algorithm:algorithm detr-r50-e632da11.pth /home/algorithm/.cache/torch/hub/checkpoints/detr-r50-e632da11.pth
# COPY --chown=algorithm:algorithm /ultralytics_yolov5_master/ /opt/algorithm/ultralytics_yolov5_master/
COPY --chown=algorithm:algorithm /.config /home/algorithm/.config
COPY --chown=algorithm:algorithm src /opt/algorithm/src
COPY --chown=algorithm:algorithm plt-nodule.yml /opt/algorithm/plt-nodule.yml
COPY --chown=algorithm:algorithm training_utils /opt/algorithm/training_utils
COPY --chown=algorithm:algorithm fastercnn50.pth /opt/algorithm/fastercnn50.pth
COPY --chown=algorithm:algorithm F1_E79_ModelX_v4_T0.325_V0.410.ckpt /opt/algorithm/F1_E79_ModelX_v4_T0.325_V0.410.ckpt
COPY --chown=algorithm:algorithm yolo5x_vindr.pt /opt/algorithm/yolo5x_vindr.pt


# Install required python packages via pip - please see the requirements.txt and adapt it to your needs .cache/torch/hub/checkpoints/resnet50-0676ba61.pth

COPY --chown=algorithm:algorithm config_retina_l.yaml /opt/algorithm/config_retina_l.yaml
COPY --chown=algorithm:algorithm config_effdet2_l.yaml /opt/algorithm/config_effdet2_l.yaml
COPY --chown=algorithm:algorithm config_fcrnn_l.yaml /opt/algorithm/config_fcrnn_l.yaml
# RUN chown -R algorithm:algorithm *
# RUN conda env create -f plt-nodule.yml
# RUN echo "source activate env" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH

COPY --chown=algorithm:algorithm process.py postprocessing.py /opt/algorithm/

# Entrypoint to run, entypoint.sh files executes process.py as a script
ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS: these labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=noduledetection
# These labels are required and describe what kind of hardware your algorithm requires to run for grand-challenge.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=12G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G


