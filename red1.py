from __future__ import print_function
import tensorflow as tf

# Importando los datos
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Hyperparametros
learning_rate = 0.001  # Tasa de apredizaje. Cuanto mas pequena, mas tarda en entrenarse la red, pero mas precisa sera
training_iters = 1000
batch_size = 128  # Tamano del lote/iteración de entrenamiento
display_step = 10  # Numero de lotes en que se mostrará en pantalla info (10 x 128 = cada 1280 iteraciones)

# Parametros de la red
n_input = 784  # 28 x 28
n_classes = 10  # Numero de clases, de etiquetas, que la red va a poder clasificar en cada imagen
dropout = 0.75  # Evita sobreajuste, apaga neuronas aleatoriamente durante el entrenamiento, forzando a buscar nuevos 'caminos' de una manera mas generalizada

# Metiendo data en la red
x = tf.placeholder(tf.float32, [None, n_input])  # De la imagen de entrada
y = tf.placeholder(tf.float32, [None, n_classes])  # De las etiquetas
keep_prob = tf.placeholder(tf.float32)  # Del dropout


# Metodo para crear la capa convolutiva: input, pesos, bias, stride
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='SAME')  # Filtros de la convolucion, usamos ZeroPadding para que la salida sea del mismo tamano
    x = tf.nn.bias_add(x, b)  # Anadimos la constante a sumar
    return tf.nn.relu(x)  # Funcion de activacion ReLu sobre los tensores


# Metodo para crear la capa de pooling
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Creamos el modelo
def conv_net(x, weights, biases, dropout):
    # Reajustamos el input primero
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Creamos primera capa convolucional, utilizando los pesos y las bias indicadas
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Creamos primera capa de pooling
    pool1 = maxpool2d(conv1, k=2)

    # Segunda capa conv
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Segunda capa pool
    pool2 = maxpool2d(conv2, k=2)

    # Capa Fully Connected
    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)  # Funcion de activacion ReLu
    fc1 = tf.nn.dropout(fc1, dropout)  # Dropout

    # Salida
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Pesos: Alto y ancho del filtro, profundidad de input y de output.
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),  # El input ocupa lo mismo que el output de wc1
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

# Biases
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construimos modelo
pred = conv_net(x, weights, biases, keep_prob)

# Definimos optimizador y la tasa de perdida
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluamos modelo
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,
                                                      1))  # Comparamos los resultados entre la prediccion de la red y los  verdaderos outputs de y
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Inicializamos todas las variables
init = tf.initialize_all_variables()

# Grafo
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Entrenar la red hasta que alcance el numero de iteraciones escogido anteriormente
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculamos la perdida del lote y la precision
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                             y: mnist.test.labels[:256],
                                                             keep_prob: 1.}))
