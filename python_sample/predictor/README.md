# predictor
* 已将各部分模块封装在predictor文件夹中<br>
* predictor - 文本处理及预测模块，接受列表，列表中为文本<br>
* model - 包含词库模块tokenizer及预测模型<br>
* data_processing.py 包含分词及文本转序列功能<br>
``` python
from predictor import Predictor

content = ['公诉机关起诉指控，被告人张某某秘密窃取他人财物',
           '锡林浩特市人民检察院指控，被告人杨某某以非法占有为目的，秘密窃取他人财物']
model = Predictor()
p = model.predict(content)
print(p)
``` 