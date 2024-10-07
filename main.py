from micrograd.nn import MLP
from micrograd.train import train

if __name__=="__main__":
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] 

    epochs = 200

    net = MLP(nin=3, nouts=[4, 4, 1])

    y_pred, _ = train(net, xs, ys, epochs) 