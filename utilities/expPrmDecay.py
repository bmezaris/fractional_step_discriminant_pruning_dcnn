"""
Implementation in PyTorch of the asymptotic schedule described in the paper:
N. Gkalelis, V. Mezaris, "Fractional Step Discriminant Pruning:
A Filter Pruning Framework for Deep Convolutional Neural Networks",
Proc. 7th IEEE Int. Workshop on Mobile Multimedia Computing (MMC2020)
at the IEEE Int. Conf. on Multimedia and Expo (ICME), London, UK, July 2020.
History
-------
DATE       | DESCRIPTION                                         | NAME              | ORGANIZATION |
16/01/2020 | first creation asymptotic pruning schedule function | Nikolaos Gkalelis | CERTH-ITI    |
"""

import matplotlib.pyplot as plt
import numpy as np
import time


def cmpAsymptoticSchedule(theta3=.5, e3=199, tau=8., theta_cs_final = .1, scaling_attn=1.):
       """Computes an asymptotic pruning schedule for the number of filters and their weights.

       Args:
              theta3: 3d point for pruning rate
              e3: 3d point for epoch (last epoch starting from 0)
              tau: determines the epoch (i.e. 1/tau of the total epochs) where the pruning rate reaches 75% of the final pruning rate
              theta_cs_final: final pruning rate associated with the CS criterion
              scaling_attn: attenuation parameter for scaling factors 

       Returns:
              thetas: total pruning rate per epoch
              zetas: weight scaling factor per epoch
              thetas_cs: pruning rate per epoch with respect to CS criterion
              thetas_fpgm: pruning rate per epoch with respect to FPGM criterion
              e2: epoch equal to 1/tau of the total epochs

       Raises:

       """


       theta3 = 100 * theta3 # convert to percentage, e.g. from 0.5 to 50%
       theta_cs_final = 100 * theta_cs_final

       start = time.time()

       # pruning parameters
       theta2 = (3. / 4.) * theta3 # 2nd point

       # epoch parameters
       e1 = 0.
       e2 = (1. / tau) * e3

       # polynomial coefficients
       coeff = np.zeros(int(tau)+1)
       coeff[-1] = theta3 - theta2
       coeff[-2] = -theta3
       coeff[0] = theta2
       #coeff = [theta2, 0, 0, 0, 0, 0, 0, -theta3, theta3 - theta2]

       # compute nontrivial real root of the polynomial
       outroots = np.roots(coeff)
       ksi = np.real(outroots[int(tau)-1])
       print('The nontrivial real root of the polynomial is: {}'.format( ksi ) )

       # estimate the parameters of the exponential function
       betta = - np.log(ksi) / e2
       alpha = theta2 / (ksi - 1)
       gamma = - alpha

       # timing
       end = time.time()
       print('Time needed: {}'.format(end - start))

       # plotting
       epochs = np.arange(e1, e3+1)
       thetas = alpha * np.exp( - betta * epochs) + gamma
       thetas[-1] = theta3 # ensure that the last element is the target pruning rate

       thetas_cs = np.minimum(theta_cs_final, thetas)
       thetas_fpgm = thetas - thetas_cs
       zetas = 1 -  thetas / theta3
       zetas = zetas * scaling_attn

       figT, ax = plt.subplots()
       ax.plot(epochs, thetas)
       ax.set(xlabel='epoch', ylabel='pruning rate', title='Total Pruning rate along different epochs')
       ax.grid()
       figT.savefig("Fractional_pruning_rate_total.png")
       #plt.show()

       figCS, ax = plt.subplots()
       ax.plot(epochs, thetas_cs)
       ax.set(xlabel='epoch', ylabel='pruning rate', title='CS pruning rate along different epochs')
       ax.grid()
       figCS.savefig("Fractional_pruning_rate_cs.png")
       #plt.show()

       figFpgm, ax = plt.subplots()
       ax.plot(epochs, thetas_fpgm)
       ax.set(xlabel='epoch', ylabel='pruning rate', title='FPGM pruning rate along different epochs')
       ax.grid()
       figFpgm.savefig("Fractional_pruning_rate_fpgm.png")
       #plt.show()

       figZ, ax = plt.subplots()
       ax.plot(epochs, zetas)
       ax.set(xlabel='epoch', ylabel='pruning rate', title='Scaling rate along different epochs')
       ax.grid()
       figZ.savefig("Fractional_scaling_rate.png")
       #plt.show()

       thetas = thetas / 100.
       thetas_cs  = thetas_cs / 100.
       thetas_fpgm = thetas_fpgm / 100.

       return thetas, zetas, thetas_cs, thetas_fpgm, int(e2)


if __name__ == '__main__':
    cmpAsymptoticSchedule()
