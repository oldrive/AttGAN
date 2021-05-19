个人练习（AttGAN的tf1转tf2）

官网的AttGAN之TF1版本https://github.com/LynnHo/AttGAN-Tensorflow.git

参考的DCGAN等的TF2版本https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2

笔记

1.数据准备（构建数据管道）

x_batch, label_batch = dataset.take(1)，保证取出来的一个个都是按照batch的形式


2.模型创建

用方法+函数式API构建模型时，函数参数一定要有所有需要传入模型的数据的input_shape，并且用这些input_shape挨个实例化layers.Input;
如果有多个输入或者输出，在函数返回的Model（inputs, outputs）中按照列表的形式传入inputs/outputs中，并且要在实例化模型中，将需要的输入数据同样按照列表的方式传入模型（如G_dec(z + [atts]),z和atts是具体的张量数据）


3.损失函数和优化器定义（有必要的话，比如对抗损失）

最终的损失值注意得是一个数值（要进行tf.reduce_mean）
优化器中的学习率参数可以传入一个学习率衰减器进去（可以自定义也可以使用内置的）


4.训练自定义（train_step）

这里是根据tape求梯度并更新参数；
求梯度时gradients = tape.gradient(loss, model.trainable_variables)，其中的loss是多个model的loss的加权和时，求导对象variables就按列表的方式传入这多个model的变量（即model_1.trainable_variables + model_2.trainable_variables）trainable_variables本身就是一个列表所以用+号；
更新参数时也一样传入列表形式的多个trainable_variables（如G_optimizer.apply_gradients(zip(G_gradients, [*G_enc.trainable_variables, *G_dec.trainable_variables]))）；


5.训练

如果时生成对抗式训练，先训练D，根据D.optimizer.iterations.numpy()判断D训练了几次后再训练一次G


6.配置文件config

所有的常量按照大写的方式写入这个文件（如BATCH_SIZE）

