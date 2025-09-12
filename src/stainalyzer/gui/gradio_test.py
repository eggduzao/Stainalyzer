
import numpy as np

np.random.seed(1987)

def add_glitch(value, n, maxv=1.0, minv=0.0, mult=1):

    possibility_vector = np.array(["mili", "micro", "nano", "pico", "femto", "atto", "zepto", "yocto", "ronto", "quecto"])
    sign_vector = np.array([-1, 1])

    for i in range(0, n):

        item = np.random.choice(possibility_vector)
        sign = np.random.choice(sign_vector)
        value += (_glitch(0.0, type=item) * sign * mult)

    return max(min(value, maxv), minv)

def _glitch(value, type="pico", function=None, *args, **kwargs):

    random_digit = None
    if function:
        random_digit = function(*args, **kwargs)
        random_digit = np.clip(random_digit, 0, 10)
    else:
        random_digit = np.random.randint(0, 10)  # Random number from 0 to 9

    if type=="mili":
        new_n = int(value * 10) / 10 + random_digit / 1e1
    elif type=="micro":
        new_n = int(value * 10) / 10 + random_digit / 1e2
    elif type=="nano":
        new_n = int(value * 10) / 10 + random_digit / 1e3
    elif type=="pico":
        new_n = int(value * 10) / 10 + random_digit / 1e4
    elif type=="femto":
        new_n = int(value * 10) / 10 + random_digit / 1e5
    elif type=="atto":
        new_n = int(value * 10) / 10 + random_digit / 1e6
    elif type=="zepto":
        new_n = int(value * 10) / 10 + random_digit / 1e7
    elif type=="yocto":
        new_n = int(value * 10) / 10 + random_digit / 1e8
    elif type=="ronto":
        new_n = int(value * 10) / 10 + random_digit / 1e9
    elif type=="quecto":
        new_n = int(value * 10) / 10 + random_digit / 1e10
    else:
        new_n = value

    return new_n

vec = ["mili", "micro", "nano", "pico", "femto", "atto", "zepto", "yocto", "ronto", "quecto"]

# for k in range(0,len(vec)):
#     for i in range(0, 10):
#         value = 0
#         v = _glitch(value, type=vec[k])
#         print(f"{vec[k]} = {v}")

for i in range(0, 10):
    value = 96.572
    new_value = add_glitch(value, n=10, maxv=100.0, minv=0.0)
    print(new_value)




