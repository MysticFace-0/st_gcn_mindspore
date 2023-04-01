import mindspore
from mindspore import ops
total_acc_num = 0.
label = mindspore.Tensor([22])
pred = mindspore.Tensor([22])
total_acc_num += (pred == label).sum()
accuracy = total_acc_num/ (1* 10)
print(accuracy)

