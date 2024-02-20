
import tensorflow as tf
import os


LAYER_SIZE = 400
PROB_WIN_LAYER_SIZE_1 = 100
PROB_WIN_LAYER_SIZE_2 = 50

TENSORFLOW_SAVE_FILE = 'agent'
TENSORFLOW_CHECKPOINT_FOLDER = 'tensorflow_checkpoint'



class Model:
    """ Neural network to implement deep Q-learning with memory
    """
    def __init__(self, num_states, num_actions, batch_size, restore, sess):

        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        
        # define the placeholders
        self.states = None
        self.actions = None
        
        # the output operations
        self.logits = None
        self.loss = None
        self.optimizer = None
        
        
        # now setup the model
        self.define_model()

        
        self.init_variables = tf.compat.v1.global_variables_initializer()


        self.sess = sess
        if restore:
            self.load()
        else:
            self.sess.run(self.init_variables)
    
    def save(self):
        self.saver = tf.compat.v1.train.Saver()
        """ save model parameters to file"""
        local = self.saver.save(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
        print("saved to ", local)
        
    def load(self):
        """ load model parameters from file"""
        self.saver.restore(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
        
        
    def define_model(self):
        """ builds a simple tensorflow dense neural network that accepts the state and computes the action."""
        
        self.states = tf.compat.v1.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.q_s_a = tf.compat.v1.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(LAYER_SIZE, activation=tf.nn.relu, input_shape =(self.num_states,)),
            tf.keras.layers.Dense(LAYER_SIZE, activation=tf.nn.relu),
            tf.keras.layers.Dense(LAYER_SIZE, activation=tf.nn.relu),
            tf.keras.layers.Dense(LAYER_SIZE, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.num_actions)
        ])

        
        self.logits = self.model(self.states)
        self.loss = tf.reduce_mean(self.logits)


        print("################:")
        self.logits = self.model(self.states)
        print("nn output:", self.logits)
        print("qsa:", self.q_s_a)
        
    def training_steps(self):    
        print("going thru 1 training step")
        with tf.GradientTape() as tape:
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels = self.q_s_a, logits = self.logits)
            self.loss = tf.reduce_mean(neg_log_prob*self.logits)
            gradients = tape.gradient(self.loss, self.model.trainable_variables)
        
        self.optimizer = tf.optimizers.Adam()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        
        
    def get_num_actions(self):
        """ Returns the number of possible actions """
        return self.num_actions

    def get_num_states(self):
        """ Returns the length of the input state """
        return self.num_states

    def get_batch_size(self):
        """ Returns the batch size """
        return self.batch_size
        
    def predict_one(self, state):
        """ Run the state ( which is state.asVector() ) through the model and return the predicted q values """
        # print("SELF.STATES:", self.states)
        # print("STATE:", state)
        # print("predict1")
        return self.sess.run(self.logits, feed_dict={self.states: state.reshape(1, self.num_states)})
            
    def predict_batch(self, states):
        """ Run a batch of states through the model and return a batch of q values. """
        # print("SELF.STATES:", self.states)
        # print("STATE:", states)
        # print("MODEL:", self.model.weights)
        # print("LOGITS:", self.logits)
        # print("predict_batch")
        return self.sess.run(self.logits, feed_dict={self.states: states})
    
    def train_batch(self, x_batch, y_batch):
        """ Trains the model with a  batch of X (state) -> Y (reward) examples """
        # print("SELF.STATES:", self.states)
        # print("MODEL:", self.model.weights)
        # print("optimizer:", self.optimizer)
        # print("loss:", self.loss)
        # print("X_batch:", x_batch)
        # print("Y_batch:", y_batch)
        # print("train_batch")
        return self.sess.run([self.loss], feed_dict={self.states: x_batch, self.q_s_a: y_batch})