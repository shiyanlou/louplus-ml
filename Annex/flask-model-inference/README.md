### 本地启动 Flask 服务完成模型推理

1. 首先在终端执行 `predict.py` 启动 Flask app

```bash
$ python predict.py

* Serving Flask app "predict" (lazy loading)
* Environment: production
WARNING: Do not use the development server in a production environment.
Use a production WSGI server instead.
* Debug mode: off
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

```

2. 开启新终端向目的地址发送请求。

```python
In [1]: import requests                                                                             

In [2]: sample = [{"pclass": 1, "sex": "male", "embarked": "C"}, {"pclass": 2, "sex": "female", "embarked": "S"}]                                  

In [3]: requests.post(url='http://127.0.0.1:5000', json=sample).content                             
Out[3]: b'{"prediction":["no","yes"]}\n'
```
