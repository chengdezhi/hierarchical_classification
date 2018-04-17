import tensorflow as tf

class Trainer(object):
  def __init__(self, config, model):
    self.config = config
    self.model = model
    self.opt = tf.train.AdamOptimizer(config.learning_rate)

    self.var_list = model.get_var_list()
    self.global_step = model.get_global_step()
    self.summary = model.summary

    with tf.name_scope("grads_{}".format(config.model_name)) as scope, tf.device("/{}:{}".format(config.device_type, config.gpu_idx)):
      loss = model.get_loss()
      grads = self.opt.compute_gradients(loss, var_list=self.var_list)
      self.bclip = grads
      self.grads = [(tf.clip_by_value(grad, -1., 1.), var) for (grad, var) in grads if not grad is None]

    # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

  def step(self, sess, batch, get_summary=False):
    batch_idx, batch_ds = batch
    model = self.model
    config = self.config
    if config.check:
      for key,value in batch_ds.data.items():
        if isinstance(value[0], list) and len(value[0])>10: print(key, value[:][:5])
        else: print(key, value[:])
    feed_dict = model.get_feed_dict(batch, self.config)
    if config.check and False:
      w2v, l2v = sess.run([model.word_embeddings, model.label_embeddings], feed_dict=feed_dict)
      print("check: w2v", w2v.shape, w2v)
      print("check: l2v", l2v.shape, l2v)
    if config.check :
      logits = sess.run(model.logits, feed_dict=feed_dict)
      print("logits:", logits[0:4], logits.shape)
    train_op = model.train_op    #  use orignal train_op , worked
    if get_summary:
      loss, summary, train_op = sess.run([model.loss, model.summary, train_op], feed_dict=feed_dict)
    else:
      loss, train_op = sess.run([model.loss, train_op], feed_dict=feed_dict)
      summary = None
    grads = sess.run(self.grads, feed_dict = feed_dict)
    # print("clip grad:",  type(grads[0]), type(grads[1]))
    print("loss:", loss)

    return loss, summary, train_op











