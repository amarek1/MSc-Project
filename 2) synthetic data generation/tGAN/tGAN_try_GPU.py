
from tgan.data import load_demo_data

data, continuous_columns = load_demo_data('census')

from tgan.model import TGANModel

tgan = TGANModel(continuous_columns,gpu=0)

tgan.fit(data)

num_samples = 100

samples = tgan.sample(num_samples)