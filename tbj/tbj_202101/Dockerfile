FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN conda install clang_osx-64 clangxx_osx-64 -c anaconda
RUN export CC=gcc_linux-64 \ 
    && export CXX=g++_linux-64

CMD [ "python", "./your-daemon-or-script.py" ]