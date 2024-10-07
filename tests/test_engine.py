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


class TestMul:
    a = Value(2.0)
    b = Value(3.0)


    at = torch.Tensor([2.0]).double()
    bt = torch.Tensor([3.0]).double()


    def test_mul(self):
        c = self.a * self.b
        ct = self.at * self.bt

        assert c.data == ct.data.item()

    def test_radd(self):
        d = 3.0 * self.a
        dt = 3.0 * self.at

        assert d.data == dt.data.item()


class TestDiv:
    a = Value(6.0)
    b = Value(3.0)


    at = torch.Tensor([6.0]).double()
    bt = torch.Tensor([3.0]).double()


    def test_pow(self):
        c = self.a **2
        ct = self.at **2

        assert c.data == ct.data.item()

    def test_div(self):
        d = self.a / self.b
        dt = self.at / self.bt

        assert d.data == dt.data.item()

class TestSub:
    a = Value(6.0)
    b = Value(3.0)


    at = torch.Tensor([6.0]).double()
    bt = torch.Tensor([3.0]).double()

    def test_sub(self):
        assert (self.a-self.b).data == (self.at - self.bt).data.item()

def test_grad():
    x = Value(-4.0)
    z = 2 * x + 2 + x

    z.backward()
    xgrad = x.grad


    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    z.backward()
    xtgrad = x.grad.item()

    assert xgrad==xtgrad

def test_tanh():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    q.backward()
    xgrad = x.grad


    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    q.backward()
    xtgrad = x.grad.item()

    assert (xgrad - xtgrad) < 1e-6

def test_tanh_exp():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = ((2*z).exp()-1)/((2*z).exp()+1) + z * x
    q.backward()
    xgrad = x.grad


    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    # q = z.tanh() + z * x
    q = ((2*z).exp()-1)/((2*z).exp()+1) + z * x
    q.backward()
    xtgrad = x.grad.item()

    assert (xgrad - xtgrad) < 1e-6