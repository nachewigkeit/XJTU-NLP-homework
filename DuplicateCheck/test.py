import numpy as np

import config
import utils

with open(config.newTextPath, "r", encoding="utf-8") as f:
    lines = f.readlines()

tar = list(range(1, 17))
answers = utils.duplicateCheck(lines, tar, 0.8, 3)

for answer in answers:
    print(answer)
