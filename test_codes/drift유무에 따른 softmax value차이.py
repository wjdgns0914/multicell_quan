from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from funcData import get_data_provider
import funcCustom
FLAGS = tf.app.flags.FLAGS
"""
질문1:sess를 인자로 왜 받아올까? 내가 더한거였다..삭제.
질문2:왜 세션을 한번 더 여는걸까?
    추측1#이 녀석 때문에 sess를 하나 더 생성해야한다. 같은 세션안에 조정자가 여러개 있으면 충돌하는거 같다
    추측2#저장 된 웨이트를 통해 evaluation을 할 수 있게하려고?같은 세션에서 저장한 변수를 restore하면 덮어씌워지니까 같다.
    ->이 이유는 아닐듯, 왜냐면 그럴거면 바로 evaluate쓸 때의 파라미터로 evaluation하면 되니까.
"""
def evaluate(model, dataset,
        batch_size=100,
        checkpoint_dir='./checkpoint',
        now_epoch=1):
    with tf.Graph().as_default() as g:
        data = get_data_provider(dataset, training=False)
        x, yt = data.next_batch(batch_size)
        # is_training = tf.placeholder(tf.bool,[],name='is_training')
        # Build the Graph that computes the logits predictions
        y = model(x, is_training=False)

        # Calculate predictions.
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
        # accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y,yt,1), tf.float32))
        yt_one=tf.one_hot(yt,10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yt_one, logits=y))
        # softmax = tf.nn.softmax(logits=y)
        # loss = tf.reduce_mean(-tf.reduce_sum(yt_one * tf.log(softmax + 0.00000001), axis=[1]))
        accuracy=tf.reduce_mean(tf.cast(tf.equal(yt, tf.cast(tf.argmax(y, dimension=1),dtype=tf.int32)),dtype=tf.float32))
        saver = tf.train.Saver()#variables_to_restore
        # Configure options for session
        sess = tf.Session(
                config=tf.ConfigProto(
                            log_device_placement=False,
                            allow_soft_placement=True,
                            )
                        )
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/')
        latestname=tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

         # Start the queue runners.
        coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
            tempsize = 10000 if dataset == 'MNIST' else data.size[0]
            num_batches = int(math.ceil(tempsize / batch_size))
            total_acc = 0  # Counts the number of correct predictions per batch.
            total_loss = 0 # Sum the loss of predictions per batch.
            step = 0
            count_confidence = np.zeros(shape=[10, 10])
            count_correct = np.zeros(shape=[10, 1])
            count_error = np.zeros(shape=[10, 1])

            while step < num_batches and not coord.should_stop():
              if now_epoch%10==0:
                  acc_val, loss_val,xval,yval,softmax_value,softmax_max_index = sess.run([accuracy, loss,x,yt]+[softmax]+[tf.arg_max(softmax,dimension=1)])
                  is_correct = (yval == softmax_max_index).reshape([batch_size, 1])
                  for number in range(10):
                      is_this_number = (yval==number).reshape([batch_size,1])
                      count_confidence[number,:]+=np.sum(((~is_correct)*is_this_number)*softmax_value,axis=0)
                      count_correct[number,0]+=np.sum((is_correct*is_this_number),axis=0)
                      count_error[number,0] +=np.sum(((~is_correct)*is_this_number),axis=0)
              else:
                  acc_val, loss_val, xval, yval= sess.run([accuracy, loss, x, yt])
              total_acc += acc_val
              total_loss += loss_val
              step += 1
            # Compute precision and loss
            total_acc /= num_batches
            total_loss /= num_batches
            if now_epoch % 5 == 0:
                file = open(FLAGS.checkpoint_dir + "/save_confidence.py", "a")
                funcCustom.magic_print("\nNow_epochs : ", now_epoch, file=file)
                funcCustom.magic_print("\ncount_error : ", count_error, "\ncount_correct : ", count_correct, "\ncount_not_confidence : ",
                                       count_confidence / count_error, file=file)
                file.close()


        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)
        return total_acc, total_loss

def main(argv=None):  # pylint: disable=unused-argument
  pass
  # a,c=evaluate(model=model,dataset=FLAGS.dataset,checkpoint_dir=FLAGS.checkpoint_dir)
  # print(a,c)

if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('checkpoint_dir', './results/2017-12-5-18-14-34',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'MNIST',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model_name', 'MNIST0_nodrop',
                             """Name of loaded model.""")
  tf.app.flags.DEFINE_string('Drift', False,
                             """Drift or Not.""")

  FLAGS.log_dir = FLAGS.checkpoint_dir+'/log/'
      # Build the summary operation based on the TF collection of Summaries.
      # summary_op = tf.merge_all_summaries()

      # summary_writer = tf.train.SummaryWriter(log_dir)
          # summary = tf.Summary()
          # summary.ParseFromString(sess.run(summary_op))
          # summary.value.add(tag='accuracy/test', simple_value=precision)
          # summary_writer.add_summary(summary, global_step)

  tf.app.run()
