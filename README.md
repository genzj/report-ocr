# 识别检验报告

## 安装

1. 安装[Python 3.12](https://www.python.org/downloads/release/python-3124/)
1. 安装[pipx](https://pipx.pypa.io/stable/installation/#installing-pipx)
1. 安装[poetry](https://python-poetry.org/docs/#installing-with-pipx)
1. (Windows) 安装[VS builder tool](https://visualstudio.microsoft.com/downloads/)，
   安装时勾选C++和Windows 10 SDK(或Windows 11 SDK，取决于系统版本)
1. 下载代码，进入代码文件夹后，执行`poetry install`

## 运行

### OCR

1. 将pdf文件放入`data`文件夹下
1. 在代码目录内执行 `poetry run python report_ocr.py`
1. 输出结果在output文件夹内

### 结果合并

1. 将OCR得到的csv放在`output`文件夹下
1. 在代码目录内运行 `poetry run python extract_report.py`
1. 输出结果为`output/merge.csv`

## 常见故障

### 执行`poetry install`失败，提示找不到`rc.exe`文件

参考 https://stackoverflow.com/a/14373113 找到rc.exe和rcdll.dll，复制到任何PATH环境
变量中存在的路径下。

### 下载模型时间太长或下载失败

按[此说明](https://cnocr.readthedocs.io/zh-cn/stable/usage/#_2)从百度盘下载
`densenet_lite_136-gru`模型并放到指定文件夹下。
