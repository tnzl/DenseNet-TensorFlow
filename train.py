from model import get_model

model = get_model(growth_rate=12, layers_per_block=[16,16,16], compression_ratio=0.5, bottleneck_ratio=4)
print('Number of params : ', model.count_params())