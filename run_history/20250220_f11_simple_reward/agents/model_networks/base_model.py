import tensorflow as tf
import datetime
from keras.api._v2.keras.optimizers import Adam, RMSprop, Adagrad, SGD


class BaseModel:
    """
    Base class for deep Q learning
    """
    def __init__(self, config, network_type):
        self.config = config
        self.model_dir = config.model_dir
        self.dir_output = self.model_dir + "/output/" + config.experiment + "/" + str(datetime.datetime.utcnow()) + "/"
        self.dir_model = self.model_dir + "/net/" + config.experiment + "/" + str(datetime.datetime.utcnow()) + "/"

        self.optimizer = None
        self.debug = not True
        self.sess = None
        self.saver = None

        self.add_optimizer(config.lr_method, config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def add_optimizer(self, lr_method, learning_rate, clip=-1):
        lr_method = lr_method.lower()

        if lr_method == 'adam':  # sgd method
            optimizer = Adam(learning_rate=learning_rate)
        elif lr_method == 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate)
        elif lr_method == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif lr_method == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate, momentum=0.95, epsilon=0.01)
        else:
            raise NotImplementedError("Unknown method {}".format(lr_method))

        self.optimizer = optimizer

        return optimizer

    def nn_model(self):
        raise NotImplementedError("Method not implemented")

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action_q_values(self, state):
        q_value_array = self.predict(state)
        return q_value_array[0]

    # def initialize_session(self):
    #     print("Initializing tf session")
    #     self.sess = tf.Session()
    #     self.sess.run(tf.global_variables_initializer())
    #     self.saver = tf.train.Saver()
    #
    # def close_session(self):
    #     self.sess.close()
    #
    # def add_summary(self, summary_tags, histogram_tags):
    #     self.summary_placeholders = {}
    #     self.summary_ops = {}
    #
    #     for tag in summary_tags:
    #         self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag)
    #         self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
    #
    #     for tag in histogram_tags:
    #         self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
    #         self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])
    #
    #     self.file_writer = tf.summary.FileWriter(self.dir_output + "/train", self.sess.graph)
    #
    # def inject_summary(self, tag_dict, step):
    #     summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict], {
    #         self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    #     })
    #     for summ in summary_str_lists:
    #         self.file_writer.add_summary(summ, step)
    #
    # def save_session(self):
    #     """Saves session = weights"""
    #     if not os.path.exists(self.dir_model):
    #         os.makedirs(self.dir_model)
    #     self.saver.save(self.sess, self.dir_model)
    #
    # def restore_session(self, path=None):
    #     if path is not None:
    #         self.saver.restore(self.sess, path)
    #     else:
    #         self.saver.restore(self.sess, self.dir_model)

