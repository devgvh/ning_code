import numpy as np
import torch
import 两数相加

# 定义预测函数
def predict(x):
    model.eval()
    mask_pad_x = 两数相加.mask_pad(x)
    target = [两数相加.vocab_y['<SOS>']] + [两数相加.vocab_y['<PAD>']] * 9 # 初始化输出，这个是固定值  #49->9
    target = torch.LongTensor(target).unsqueeze(0) # 增加一个维度，shape变为[1,10]
    target = target.to(两数相加.device)
    x = model.embed_x(x)
    # 编码层计算，维度不变
    x = model.encoder(x, mask_pad_x)
    # 遍历生成第1个词到第9个词
    for i in range(9):  #49->9
        y = target
        mask_tril_y = 两数相加.mask_tril(y) # 上三角遮盖
        # y编码，添加位置信息
        y = model.embed_y(y)
        # 解码层计算，维度不变
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        out = model.fc_out(y)
        # 取出当前词的输出
        out = out[:,i,:]
        out = out.argmax(dim=1).detach()
        # 以当前词预测下一个词，填到结果中
        target[:,i + 1] = out
    return target

# 测试
def model_test(x, y):
    x, y = x.to(两数相加.device), y.to(两数相加.device)
    #直接求答案
    answer = ''.join([两数相加.vocab_yr[i] for i in predict(x.unsqueeze(0))[0].tolist()])

    #这里推理已经结束，下面代码主要用于比较正确性
    question = ''.join([两数相加.vocab_xr[i] for i in x.tolist()])
    correct_answer = ''.join([两数相加.vocab_yr[i] for i in y.tolist()])

    #把问题和答案中数字无关的字符都去掉
    question_s = question.strip('<SOS>PADE')
    answer_s = answer[:13].strip('<SOS>PADE')  # 偶尔答案中会在结束符后面还生成一些数字，忽略，增加成功率
    correct_answer_s = correct_answer.strip('<SOS>PADE')

    if answer_s == correct_answer_s:
        is_correct = '预测正确'
    else:
        is_correct = '错误，正确答案是:' + correct_answer_s

    print("问题:", question_s,'预测答案:', answer_s, is_correct)

# 两数相加测试
def get_data(a, opt, b):
    #为代码简单，不是最优写法
    x = list(a) + [opt] + list(b)
    y = list(str(eval(a+opt+b)))
    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补PAD，直到固定长度
    x = x + ['<PAD>'] * 10  #50->10
    y = y + ['<PAD>'] * (11)  #51->11
    x = x[:10]   #50->10
    y = y[:11]   #51->11

    # 编码成数据
    x = [两数相加.vocab_x[i] for i in x]
    y = [两数相加.vocab_y[i] for i in y]
    # 转Tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


if __name__ == '__main__':
    # 用Transformer类定义一个模型model
    model = 两数相加.Transformer()

    #加载已经训练好的模型
    model.load_state_dict(torch.load("./model_plus_final.pth"))
    model = model.to(两数相加.device)
    x1, y1 = get_data('123', '+', '456')
    x2, y2 = get_data('111', '*', '111')
    x4, y4 = get_data('987', '-', '321')

    model_test(x1, y1)
    model_test(x2, y2)
    model_test(x4, y4)