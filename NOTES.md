About Quantization
==================

In C++ it's doable via this function, but AFAIK there is no "just quantize this model for me" thing.

```cpp
torch::quantize_per_tensor(param, 0.1, 0, torch::kQInt8)
```

In Python it can be done like this.

```python
from collections import OrderedDict

net = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 3, 3)),
    ('conv2', nn.Conv2d(3, 1, 3)),
]))
net.eval()
net.qconfig = torch.quantization.get_default_qconfig('fbgemm')

quant = torch.quantization.quantize(net, nn.Sequential.forward, (torch.rand(64, 1, 6, 6), ))
```

Dump weights
============
```cpp
void Net::dump_parameters() {
    float data = *(conv1->weight[0][0][0][0].data_ptr<float>());
    fmt::print("conv1 weight sizes = {}\n", conv1->weight.sizes());
    fmt::print("conv1 weight first entry = {}\n", conv1->weight[0][0][0][0]);
    fmt::print("conv1 weight first entry = {}\n", data);
}
```
