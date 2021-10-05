import torch
import torch.nn as nn
from mpi4py import MPI
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Person:
    def __init__(self):
        self.base = 10
    def greeting(self):
        print('안녕하세요.')

class Gauss:
    def __init__(self, rms_dict):
        print('Gauss', rms_dict)
        self.rms_dict = rms_dict
    
    def check(self):
        print('Check Gauss', self.rms_dict)
    
    def update(self):
        print('Update Gauss')
        self.rms_dict['sum'] = self.rms_dict['sum'] + torch.tensor([1000.])
        self.rms_dict['count'] = self.rms_dict['count'] + torch.tensor([1000.])
 
class Student(Person):
    def __init__(self):
        super().__init__()
        self.rms_sum = torch.tensor([1.])
        self.rms_count = torch.tensor([2.])
        self.rms_dict = {'sum': self.rms_sum, 'count': self.rms_count}
        self.pi = Gauss(self.rms_dict)

    def check(self):
        print('Check Student', self.rms_dict)
        print('id', id(self.rms_dict['sum']))
        print('id', id(self.rms_sum))

    def update(self):
        print('Update Student')
        self.rms_sum = self.rms_sum+ torch.tensor([100.])
        self.rms_count = self.rms_count + torch.tensor([100.])
        

def main():
    james = Student()
    james.check()
    james.pi.check()
    james.update()
    james.check()
    james.pi.check()
    james.pi.update()
    james.check()
    james.pi.check()

if __name__ == '__main__':
    main()
