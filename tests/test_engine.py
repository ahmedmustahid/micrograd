import torch
from micrograd.engine import Value

class TestAdd:
    a = Value(2.0)
    b = Value(3.0)


    at = torch.Tensor([2.0]).double()
    bt = torch.Tensor([3.0]).double()


    def test_add(self):
        c = self.a + self.b
        ct = self.at + self.bt

        assert c.data == ct.data.item()

    def test_radd(self):
        d = 3.0 + self.a
        dt = 3.0 + self.at

        assert d.data == dt.data.item()



