FROM python:2.7-stretch

RUN mkdir /install
WORKDIR /install

RUN apt-get update && apt-get -y install libopenblas-base libopenblas-dev

RUN git clone https://github.com/martijnvanbeers/processing_arithmetics.git \
    && git clone https://github.com/dieuwkehupkes/keras.git

COPY models/ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10.h5 models/diagnoses/ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10_dc8.h5 processing_arithmetics/pa_demo/models/

RUN pip install ./keras \
    && pip install ./processing_arithmetics

#COPY --from=builder /install /usr/local

RUN mkdir /app
WORKDIR /app

COPY keras.json /root/.keras/

CMD ["/usr/local/bin/gunicorn", "-w 2",  "-e SCRIPT_NAME=/arithmetics", "--bind=0.0.0.0", "--timeout=600", "pa_demo.main:app"]
