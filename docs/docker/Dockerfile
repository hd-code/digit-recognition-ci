FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y texlive-full
RUN apt-get install -y pandoc pandoc-citeproc
RUN apt-get clean

WORKDIR /data

CMD bash