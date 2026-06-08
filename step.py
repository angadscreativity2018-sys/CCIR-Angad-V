import numpy as np

G = 1.321e8

def step(dt, mass1, velocity1, position1, mass2, velocity2, position2):
    d = position2 - position1
    r3 = np.dot(d, d) ** 1.5
    dt2h = 0.5 * dt * dt
    dth  = 0.5 * dt
    acc1 =  (G * mass2 / r3) * d
    acc2 = -(G * mass1 / r3) * d

    position1 = position1 + velocity1 * dt + acc1 * dt2h
    position2 = position2 + velocity2 * dt + acc2 * dt2h

    d = position2 - position1
    r3 = np.dot(d, d) ** 1.5
    nacc1 =  (G * mass2 / r3) * d
    nacc2 = -(G * mass1 / r3) * d

    velocity1 = velocity1 + (acc1 + nacc1) * dth
    velocity2 = velocity2 + (acc2 + nacc2) * dth

    return position1, velocity1, position2, velocity2
