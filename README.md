9.19
colab配置好后，出现以下报错OOM：
![OOM](https://github.com/maxqiqq/WSRD-myversion-v1/assets/116487542/c4d2ad2c-79d4-49d7-9250-d4b18ad20533)

解决方法：
删除不必要的def, loss, 具体删除变量见wandb 9.18/19_logbook


9.21
删除修改之后OOM报错更提前了。。：
![image](https://github.com/maxqiqq/WSRD-myversion-v1/assets/116487542/4afa75ff-49f6-497f-b09d-88251d9af11b)
所以，在报错的perceptua_loss=之前检查并删除
