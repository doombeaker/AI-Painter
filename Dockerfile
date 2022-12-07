FROM oneflowinc/oneflow-sd:cu112

COPY requirements.txt /tmp/

RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && python3 -m pip install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt
