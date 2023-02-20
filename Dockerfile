FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
COPY data_process /data_process
COPY dataset /dataset
COPY modelling /modelling
RUN mkdir /models
RUN mkdir /models/deberta
RUN mkdir /models/inference
# COPY models /models
COPY models/deberta /models/deberta
COPY models/inference/config.json /models/inference/config.json
COPY models/inference/merges.txt /models/inference/merges.txt
COPY models/inference/tokenizer_config.json /models/inference/tokenizer_config.json
COPY models/inference/118-1810-phrase-4-0.050446-0.600858-model.bin /models/inference/118-1810-phrase-4-0.050446-0.600858-model.bin
COPY models/inference/118-1847-3-0.7637499570846558-type_pred_model.bin /models/inference/118-1847-3-0.7637499570846558-type_pred_model.bin
COPY models/inference/118-2032-passage-5-0.352349-0.487054-model.bin /models/inference/118-2032-passage-5-0.352349-0.487054-model.bin
COPY models/inference/118-2150-multi-7-0.168945-0.390739-model.bin /models/inference/118-2150-multi-7-0.168945-0.390739-model.bin
COPY models/inference/119-028-mix-5-0.158897-0.454500-model.bin /models/inference/119-028-mix-5-0.158897-0.454500-model.bin
COPY models/inference/vocab.json /models/inference/vocab.json
# COPY x00 /x00
# COPY x01 /x01
# COPY x02 /x02
# COPY x03 /x03
# COPY x04 /x04
# COPY x05 /x05
# COPY x06 /x06
# COPY x07 /x07
# COPY x08 /x08
# COPY x09 /x09
# COPY x10 /x10
# COPY x11 /x11
# COPY x12 /x12
# COPY x13 /x13
# COPY x14 /x14
# COPY x15 /x15
# COPY x16 /x16
# COPY x17 /x17
# COPY unzip_and_run.sh /unzip_and_run.sh
COPY output /output
COPY tmp /tmp
COPY utils /utils
COPY clickbait_spoiling_task_1.py /
COPY clickbait_spoiling_task_2.py /
COPY hyperparams.py /
COPY pipeline.py /
COPY requirements.txt /
COPY run_classification.py /
COPY run_training.py /
COPY training_utils.py /

#RUN apt-get update \
#	&& apt-get install -y git-lfs wget \
#    && git clone 'https://huggingface.co/ttssrr423/semeval2023-clickbait-spoiler-st491' /models/inference \
#	&& cd /models/inference \
#    && git fetch \
#	&& git pull \
#    && rm -Rf .git \
#    && cd /
# https://huggingface.co/ttssrr423/semeval2023-clickbait-spoiler-st491/blob/main/118-1810-phrase-4-0.050446-0.600858-model.bin
RUN pip3 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  pandas jupyterlab docker datasets transformers
RUN pip3 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /requirements.txt

# COPY transformer-baseline-task-2.py run_qa.py trainer_qa.py utils_qa.py /

ENTRYPOINT [ "/clickbait_spoiling_task_2.py" ]