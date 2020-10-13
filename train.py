from model import get_model
from data_loader import get_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

tr_ds, te_ds = get_data('cifar100')

print(tr_ds.element_spec)
model = get_model(growth_rate=12, layers_per_block=[16,16,16], compression_ratio=0.5, bottleneck_ratio=4)
optimizer = SGD(momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
model.fit(tr_ds, validation_data=te_ds, epochs=1)
