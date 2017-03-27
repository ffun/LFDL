import ffun.util as Fut

nc = Fut.NetCalculator()
nc.set_dataLayer([9,33,3])
nc.append_layer('conv',ksize=[3,3,3,64])
nc.append_layer('pool',ksize=[1,1,2,1],strides=[1,1,2,1])
nc.append_layer('conv',ksize=[3, 3, 64, 128])
nc.append_layer('pool',ksize=[1,1,2,1],strides=[1,1,2,1])

nc.print_layers()

print nc.num_of_layers()