import numpy as np
import torch
import tensorflow as tf
import sys

sys.path.append('./')
import constants

import transformers

from dt_models.trajectory_model import TrajectoryModel #dt_models.

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    # print("x.shape",x.shape)
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        #use_causal_mask = True
        )
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class Block(tf.keras.layers.Layer):
    """An unassuming Transformer block"""

    def __init__(self, hidden_dim, **kwargs):
        super(Block, self).__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = CausalSelfAttention(**kwargs)  # Pass kwargs to CausalSelfAttention
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * hidden_dim),  # Unpack kwargs
            tf.keras.layers.Activation(tf.nn.gelu),
            tf.keras.layers.Dense(hidden_dim),  # Unpack kwargs
            tf.keras.layers.Dropout(0.1),  
        ])

    def call(self, x, training=True):
        with tf.GradientTape() as tape:
            # print("x_block.shape", x.shape)
            # print("ln1:", self.ln1(x).shape)
            attn_output = self.attn(self.ln1(x), training=training)
            x = x + attn_output
            mlp_output = self.mlp(self.ln2(x), training=training)
            x = x + mlp_output
        return x

# test_block = Block(hidden_dim = 10, num_heads = 3, key_dim = 10)
# try:
#     print(test_block.trainable_variables)
# except:
#     print(test_block.trainable_weights)
# config = {"num_heads": 3, "key_dim":10}
# test = Block(num_heads = 3, key_dim = 10)
# print("test:", test)

class DecisionTransformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            # max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim)#, max_length=max_length)

        self.hidden_size = hidden_size

        # print("act_dim:", act_dim)
        # print("state_dim:", state_dim)
        # print("hidden_size:", hidden_size)

        self.optimizer = tf.optimizers.Adam()
       
        # self.embed_timestep = tf.keras.layers.Embedding(max_ep_len, hidden_size) #max_ep_len determines max number of timesteps
        # self.embed_return = tf.keras.layers.Dense(1, hidden_size)
        self.embed_state = tf.keras.layers.Conv2D(self.state_dim, (3,3), input_shape=(2*constants.BOARD_SIZE-1,2*constants.BOARD_SIZE-1,1), dtype=tf.float32)
        # self.compress_state = tf.keras.layers.Dense(self.state_dim) #too small to capture
        # self.embed_state_reshape = tf.keras.layers.Reshape((state_dim, 1))

        # self.embed_action = tf.keras.layers.Dense(self.act_dim, hidden_size) #only need when actions are continuous

        self.embed_ln = tf.keras.layers.LayerNormalization(axis = -1)

        self.transformer = Block(hidden_dim = state_dim, num_heads = 4, key_dim = state_dim)

        self.predict_action = tf.keras.Sequential(
            [*([tf.keras.layers.Dense(self.act_dim)] + 
               ([tf.keras.layers.Dense(self.act_dim, activation = 'tanh')] if action_tanh else []) +
               [tf.keras.layers.Reshape((-1, act_dim*11*11))] +
               [tf.keras.layers.Dense(self.act_dim),
                tf.keras.layers.Flatten()]

               )
            ]
        )

        self.model = tf.keras.Sequential([
            self.embed_state,
            # self.compress_state,
            # self.embed_state_reshape,
            self.embed_ln,
            self.transformer,
            self.predict_action 
            ])
        
       
        self.model.build((None,state_dim))
        self.optimizer.build(self.model.trainable_variables)
        print(self.model.summary())

        # self.predict_state = tf.keras.layers.Dense(hidden_size, self.state_dim)
        
        
        # self.predict_return = tf.keras.layers.Dense(hidden_size, 1)

    def forward(self, states): #actions, rewards, returns_to_go, timesteps,  attention_mask=None):
        # print("states shape:", states.shape)
        # # batch_size, seq_length = states.shape[0], states.shape[1]

        # # if attention_mask is None:
        # #     # attention mask for GPT: 1 if can be attended to, 0 if not
        # #     attention_mask = tf.ones((batch_size, seq_length), dtype=tf.double)

        # # embed each modality with a different head
        # print("pre embed states:", states)
        # print(self.embed_state)
        # state_embeddings = self.embed_state(states)
        # # action_embeddings = self.embed_action(actions)
        # # returns_embeddings = self.embed_return(returns_to_go)
        # # time_embeddings = self.embed_timestep(timesteps)
        
        
        # # time embeddings are treated similar to positional embeddings
        # # state_embeddings = state_embeddings + time_embeddings
        # # action_embeddings = action_embeddings + time_embeddings
        # # returns_embeddings = returns_embeddings + time_embeddings

        # # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # # which works nice in an autoregressive sense since states predict actions
        # # stacked_inputs = tf.stack( #only need when input action,state,reward as a sequence
        # #     [state_embeddings], axis=1 #returns_embeddings, action_embeddings
        # # )


        # # print(tf.shape(stacked_inputs))
        # # print(self.embed_ln.weights)
        # # stacked_inputs = tf.transpose(stacked_inputs, perm=[0, 2, 1, 3])
        # # stacked_inputs = tf.reshape(stacked_inputs, [batch_size, 3 * seq_length, self.hidden_size])
        # # stacked_inputs = self.embed_ln(stacked_inputs)
        # stacked_inputs = self.embed_ln(state_embeddings)

        # # to make the attention mask fit the stacked inputs, have to stack it as well
        # # stacked_attention_mask = tf.stack(
        # #     (attention_mask), axis=1 #, attention_mask, attention_mask
        # # )
        # # stacked_attention_mask = tf.transpose(stacked_attention_mask, perm=[0, 2, 1])
        # # stacked_attention_mask = tf.reshape(stacked_attention_mask, [batch_size, 3 * seq_length])

        # print("stacked_inputs:", stacked_inputs)
        # # we feed in the input embeddings (not word indices as in NLP) to the model
        # # transformer_outputs = self.transformer(stacked_inputs)           ##uncomment
        #     #attention_mask=stacked_attention_mask,
        
        

        # # reshape x so that the second dimension corresponds to the original
        # # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # # x = tf.reshape(transformer_outputs, [batch_size, seq_length, 3, self.hidden_size])
        # # x = tf.transpose(x, perm=[0, 2, 1, 3])


        # # get predictions
        # # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        # # action_preds = self.predict_action(x[:,1])  # predict next action given state
        # print("#####")

        # # print("transformer outputs:",transformer_outputs)
        # # print(self.predict_action)
        # action_preds = self.predict_action(stacked_inputs) #transformer_outputs) #uncomment
        grid_state = [state[0] for state in states]
        positions = [state[1] for state in states]
        print("gs:",grid_state)
        grid_state = np.expand_dims(grid_state, axis = -1)
        action_preds = self.model(grid_state)


        return action_preds #state_preds, return_preds
    
    def forward_train(self, states, y_actions): #actions, rewards, returns_to_go, timesteps,  attention_mask=None):
        # print("states shape:", states)
        grid_state = [state[0] for state in states]
        positions = [state[1] for state in states]
        grid_state = np.expand_dims(grid_state, axis = -1)


        with tf.GradientTape() as tape:
            action_preds = self.model(grid_state) #this must be within tape so that it will be under the scope
            loss = tf.math.reduce_mean((action_preds - y_actions)**2)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        # print("model vars:", self.model.trainable_variables)
        # print("model_first summary:", self.model.summary())
        
        # print("action pred final:", action_preds)


        return loss #state_preds, return_preds
        

    def get_action(self, states, **kwargs): #, actions, rewards, returns_to_go, timesteps
        # given multiple states, predict the next action
        grid_state = [state[0] for state in states]
        positions = [state[1] for state in states]
        states = tf.reshape(grid_state, [1, -1, self.state_dim])
        # actions = tf.reshape(actions, [1, -1, self.act_dim])
        # returns_to_go = tf.reshape(returns_to_go, [1, -1, 1])
        # timesteps = tf.reshape(timesteps, [1, -1])

        # if self.max_length is not None:
        #     states = states[:, -self.max_length:]
        #     actions = actions[:, -self.max_length:]
        #     returns_to_go = returns_to_go[:, -self.max_length:]
        #     timesteps = timesteps[:, -self.max_length:]

        #     # Pad all tokens to sequence length
        #     padding_length = tf.maximum(0, self.max_length - tf.shape(states)[1])
        #     padding_states = tf.zeros((tf.shape(states)[0], padding_length, self.state_dim), dtype=tf.float32)
        #     padding_actions = tf.zeros((tf.shape(actions)[0], padding_length, self.act_dim), dtype=tf.float32)
        #     padding_returns_to_go = tf.zeros((tf.shape(returns_to_go)[0], padding_length, 1), dtype=tf.float32)
        #     padding_timesteps = tf.zeros((tf.shape(timesteps)[0], padding_length), dtype=tf.int32)

        #     states = tf.concat([padding_states, states], axis=1)
        #     actions = tf.concat([padding_actions, actions], axis=1)
        #     returns_to_go = tf.concat([padding_returns_to_go, returns_to_go], axis=1)
        #     timesteps = tf.concat([padding_timesteps, timesteps], axis=1)

        #     # Create attention mask
        #     attention_mask = tf.concat([tf.zeros((1, padding_length), dtype=tf.int32), tf.ones((1, tf.shape(states)[1]), dtype=tf.int32)], axis=1)

        #     # Convert attention mask to appropriate dtype
        #     attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        # else:
        #     attention_mask = None

        action_preds = self.forward( #, return_preds, _, 
            states)#, attention_mask=attention_mask,actions, None, returns_to_go, timesteps, **kwargs)

        return action_preds[0,-1]
    


