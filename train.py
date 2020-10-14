from model import get_model
from data_loader import get_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
import os
os.system("pip install wandb -q")
import wandb
from wandb.keras import WandbCallback


growth_rate = int(input("Growth rate: "))
d = int(input("Number of dense blocks: "))

layers = []
for i in range(d):
    layers.append(int(input(f"Number of layers in dense block {i+1}: ")))

B = input("Bottleneck (y/n): ")
B = True if B == 'y'else False
C = input("Compression (y/n): ")
C = True if C == 'y'else False

depth = 2 * sum(layers) + len(layers) + 1

# Run name and model name are same
run_name = 'k'+str(growth_rate)+'d'+str(depth)
if B: 
    run_name+='B'
if C:
    run_name+='C'

depth = 2 * sum(layers) + len(layers) + 1

print(run_name)

wandb.init(id=run_name, project='dense-net-implementation', resume=True )

data = int(input('Dataset : 1. cifar100; 2. cifar10\nEnter(1/2): '))

classes = None
if data == 1:
    tr_ds, te_ds = get_data('cifar100')
    classes = 100
elif data == 2:
    tr_ds, te_ds = get_data('cifar10')
    classes = 10

model = get_model(growth_rate=growth_rate
                    , layers_per_block = layers
                    , classes=classes
                    , compression_ratio=0.5 if C else 1
                    , bottleneck_ratio=4 if B else 1)
optimizer = SGD(momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

model.fit(tr_ds, epochs=5, validation_data=te_ds12
, callbacks=[WandbCallback])
