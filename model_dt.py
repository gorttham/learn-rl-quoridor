
import tensorflow as tf
import os
from dt_models.decision_transformer import DecisionTransformer


LAYER_SIZE = 400
PROB_WIN_LAYER_SIZE_1 = 100
PROB_WIN_LAYER_SIZE_2 = 50

TENSORFLOW_SAVE_FILE = 'agent'
TENSORFLOW_CHECKPOINT_FOLDER = 'tensorflow_checkpoint'



class Model:
    """ Neural network to implement deep Q-learning with memory
    """
    def __init__(self, num_states, num_actions, batch_size, restore):

        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        
        self.time_step = 0

        # now setup the model
        self.define_model()


        
        if restore:
            self.load()
    
    # def save(self):
    #     self.saver = tf.compat.v1.train.Saver()
    #     """ save model parameters to file"""
    #     local = self.saver.save(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
    #     print("saved to ", local)
        
    # def load(self):
    #     """ load model parameters from file"""
    #     self.saver.restore(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
        
        
    def define_model(self):
        """ build decision transformer"""
        
        self.model = DecisionTransformer(state_dim = self.num_states, 
                                         act_dim = self.num_actions, 
                                         hidden_size = self.num_actions + self.num_states)

        print("Model and Optimiser initialised...")
        
        
        
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
        action_preds = self.model.forward( #, state_preds, reward_preds
            state.reshape(1, self.num_states)
        )

        return action_preds[0]
            
    def predict_batch(self, states):
        """ Run a batch of states through the model and return a batch of q values. """
        # print("SELF.STATES:", self.states)
        # print("STATE:", states)
        # print("MODEL:", self.model.weights)
        # print("LOGITS:", self.logits)
        # print("predict_batch")

        action_preds = self.model.forward(states) #, state_preds, reward_preds

        return action_preds
    
    def train_batch(self, x_batch, y_batch): #returns loss
        """ Trains the model with a  batch of X (state) -> Y (reward) examples """
        # print("SELF.STATES:", self.states)
        # print("MODEL:", self.model.weights)
        # print("optimizer:", self.optimizer)
        # print("loss:", self.loss)
        # print("X_batch:", x_batch)
        # print("Y_batch:", y_batch)
        # print("train_batch")

        # print("going thru 1 batch training step")
        loss = self.model.forward_train(x_batch, y_batch)

        return loss
    