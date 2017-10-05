import tensorflow as tf

def train():
  tf.reset_default_graph()
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  return x

def main(_):
    for i in range(2):
        x = train()
        print(x)

if __name__ == '__main__':
    tf.app.run(main=main)