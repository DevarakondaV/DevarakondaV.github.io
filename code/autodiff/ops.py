import numpy as np
import torch


class Add:
    def init(self):
        pass

    def forward(self, A, B):
        C = A + B
        self.jacobianA = np.ones(A.shape)
        self.jacobianB = np.ones(B.shape)
        return C

    def backward(self, jacobian):
        return jacobian * self.jacobianA, jacobian * self.jacobianB


def addExample():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    Op = Add()
    C = Op.forward(A, B)
    JacobianA, JacobianB = Op.backward(1)
    print(f"ADD: {C}\n{JacobianA}\n{JacobianB}")

    tA = torch.tensor(A, requires_grad=True)
    tB = torch.tensor(B, requires_grad=True)
    tC = tA + tB
    tC.backward(torch.ones(tC.shape))
    print(tA.grad, tB.grad)


class Sub:
    def init(self):
        pass

    def forward(self, A, B):
        C = A - B
        self.jacobianA = np.ones(A.shape)
        self.jacobianB = -np.ones(A.shape)
        return C

    def backward(self, jacobian):
        return jacobian * self.jacobianA, jacobian * self.jacobianB


def subExample():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    Op = Sub()
    C = Op.forward(A, B)
    JacobianA, JacobianB = Op.backward(1)
    print(f"ADD: \n{C}\n{JacobianA}\n{JacobianB}")

    tA = torch.tensor(A, requires_grad=True)
    tB = torch.tensor(B, requires_grad=True)
    tC = tA - tB
    tC.backward(torch.ones(tC.shape))
    print(f"Torch: \n{tC}\n{tA.grad}\n{tB.grad}")


class Multiply:  # Hadamard Product
    def init(self):
        pass

    def forward(self, A, B):
        C = np.multiply(A, B)
        self.jacobianA = B
        self.jacobianB = A
        return C

    def backward(self, jacobian):
        return jacobian * self.jacobianA, jacobian * self.jacobianB


def multiplyExample():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    Op = Multiply()
    C = Op.forward(A, B)
    JacobianA, JacobianB = Op.backward(1)
    print(f"Multiply: \n{C}\n{JacobianA}\n{JacobianB}")

    tA = torch.tensor(A, requires_grad=True)
    tB = torch.tensor(B, requires_grad=True)
    tC = tA*tB
    tC.backward(torch.ones(tC.shape))
    print(f"Torch: \n{tC}\n{tA.grad}\n{tB.grad}")


class MatMul:
    def init(self):
        pass

    def forward(self, A, B):
        C = A @ B
        self.jacobianA = np.transpose(B)
        self.jacobianB = np.transpose(A)
        return C

    def backward(self, jacobian):
        return [jacobian @ self.jacobianA, self.jacobianB @ jacobian]


def matMulExample():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    Op = MatMul()
    C = Op.forward(A, B)
    JacobianA, JacobianB = Op.backward(np.ones(A.shape))
    print(f"MatMul: \n{C}\n{JacobianA}\n{JacobianB}")

    tA = torch.tensor(A, requires_grad=True)
    tB = torch.tensor(B, requires_grad=True)
    tC = tA @ tB
    tC.backward(torch.ones(tC.shape))
    print(f"Torch: \n{tC}\n{tA.grad}\n{tB.grad}")


def chainedOp():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    C = np.array([[9, 10], [11, 12]], dtype=float)
    addOp = Add()
    multiplyOp = Multiply()
    D = addOp.forward(A, B)
    E = multiplyOp.forward(D, C)
    jacobianD, jacobianC = multiplyOp.backward(np.ones(E.shape))
    jacobianA, jacobianB = addOp.backward(jacobianD)
    print(f"MatMul: \n{E}\n{jacobianA}\n{jacobianB}")

    tA = torch.tensor(A, requires_grad=True)
    tB = torch.tensor(B, requires_grad=True)
    tC = torch.tensor(C, requires_grad=True)
    tD = tA + tB
    tE = tD * tC
    tE.backward(torch.ones(tE.shape))
    print(f"Torch: \n{tE}\n{tA.grad}\n{tB.grad}")


chainedOp()
