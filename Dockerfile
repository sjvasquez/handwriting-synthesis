FROM tensorflow/tensorflow:1.6.0
WORKDIR /app
COPY . .

RUN apt update
RUN apt-get install python-tk python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libgtk-3-dev -y

RUN pip install svgwrite
RUN pip install matplotlib
RUN pip install scipy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install awscli
RUN pip install cairosvg==1.0.22

