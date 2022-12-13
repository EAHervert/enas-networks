import matplotlib.pyplot as plt
from Image_Processor import visualize_mat
import numpy as np

mat_class = visualize_mat.Validation()

N = mat_class.np_NOISY
GT = mat_class.np_GT

torch_N = mat_class.tensor_NOISY
torch_GT = mat_class.tensor_GT

print(N.shape)
print(torch_N.size())

plt.imshow(np.concatenate([N[10][5], GT[10][5]], axis=1))
plt.show()
