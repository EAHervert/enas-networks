from ENAS_DHDN import CONTROLLER

model = CONTROLLER.Controller(k_value=3,
                              kernel_bool=True,
                              down_bool=True,
                              up_bool=True,
                              LSTM_size=32,
                              LSTM_num_layers=1)
model()
print(model.sample_arc)
