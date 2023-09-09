from ENAS_DHDN import SHARED_DHDN

model_graph = SHARED_DHDN.SharedDHDN()
print(model_graph.layers)
print('\n')

encoder = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bottleneck = [0, 0]
decoder = [0, 0, 0, 0, 0, 0, 0, 0, 0]
model_instance_1 = SHARED_DHDN.SharedDHDN(architecture=encoder + bottleneck + decoder)
print(model_instance_1.layers)
print('\n')

encoder = [0, 0, 2, 0, 0, 2, 0, 0, 2]
model_instance_2 = SHARED_DHDN.SharedDHDN(architecture=encoder + bottleneck + decoder)
print(model_instance_2.layers)
