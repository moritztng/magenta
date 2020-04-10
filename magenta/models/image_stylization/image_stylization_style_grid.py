import os
from PIL import Image

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
import tensorflow.compat.v1 as tf

flags = tf.app.flags
flags.DEFINE_boolean('style_crossover', False,
                     'Whether to do a style crossover in the style grid.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('num_styles', None, 'Number of styles.')
flags.DEFINE_float('alpha', 1.0, 'Width multiplier')
flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')
flags.DEFINE_string('train_dir', None,
                    'Directory for checkpoints and summaries')
flags.DEFINE_string('master', '',
                    'Name of the TensorFlow master to use.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
FLAGS = flags.FLAGS


def main(_):
  with tf.Graph().as_default():
    # Create inputs in [0, 1], as expected by vgg_16.
    inputs, _ = image_utils.imagenet_inputs(
        FLAGS.batch_size, FLAGS.image_size)
    evaluation_images = image_utils.load_evaluation_images(FLAGS.image_size)

    # Load style images.
    style_images, labels, style_gram_matrices = image_utils.style_image_inputs(
        os.path.expanduser(FLAGS.style_dataset_file),
        batch_size=FLAGS.num_styles, image_size=FLAGS.image_size,
        square_crop=True, shuffle=False)
    labels = tf.unstack(labels)

    def _create_normalizer_params(style_label):
      """Creates normalizer parameters from a style label."""
      return {'labels': tf.expand_dims(style_label, 0),
              'num_categories': FLAGS.num_styles,
              'center': True,
              'scale': True}

    # Dummy call to simplify the reuse logic
    model.transform(
        inputs,
        alpha=FLAGS.alpha,
        reuse=False,
        normalizer_params=_create_normalizer_params(labels[0]))

    def _style_sweep(inputs):
      """Transfers all styles onto the input one at a time."""
      inputs = tf.expand_dims(inputs, 0)
      stylized_inputs = []
      for _, style_label in enumerate(labels):
        stylized_input = model.transform(
            inputs,
            alpha=FLAGS.alpha,
            reuse=True,
            normalizer_params=_create_normalizer_params(style_label))
        stylized_inputs.append(stylized_input)
      return tf.concat([inputs] + stylized_inputs, 0)

    style_row = tf.concat(
        [tf.ones([1, FLAGS.image_size, FLAGS.image_size, 3]), style_images],
        0)
    stylized_training_example = _style_sweep(inputs[0])
    stylized_evaluation_images = [
        _style_sweep(image) for image in tf.unstack(evaluation_images)]
    stylized_noise = _style_sweep(
        tf.random_uniform([FLAGS.image_size, FLAGS.image_size, 3]))
    stylized_style_images = [
        _style_sweep(image) for image in tf.unstack(style_images)]
    if FLAGS.style_crossover:
        grid = tf.concat(
            [style_row, stylized_training_example, stylized_noise] +
            stylized_evaluation_images + stylized_style_images,
            0)
    else:
        grid = tf.concat(
            [style_row, stylized_training_example, stylized_noise] +
            stylized_evaluation_images,
            0)
    if FLAGS.style_crossover:
        grid_shape = [
            3 + evaluation_images.get_shape().as_list()[0] + FLAGS.num_styles,
            1 + FLAGS.num_styles]
    else:
        grid_shape = [
            3 + evaluation_images.get_shape().as_list()[0],
            1 + FLAGS.num_styles]

    style_grid = tf.cast(
        image_utils.form_image_grid(
            grid,
            grid_shape,
            [FLAGS.image_size, FLAGS.image_size],
            3) * 255.0,
        tf.uint8)

    sess = tf.Session()
    with sess.as_default():
        np_array = tf.squeeze(style_grid).eval()
        im = Image.fromarray(np_array)
        im.save('matrix.png')

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
