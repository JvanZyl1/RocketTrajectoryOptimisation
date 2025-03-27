from src.agents.functions.networks import Actor, DoubleCritic

### TENSORFLOW COPIES FOR LOGGING ###
# Don't automatically update the activation functions btw, state and action names too btw
import tensorflow as tf

### ACTOR ###
class ActorTF(tf.keras.Model):
    def __init__(self, flax_actor: Actor):
        super(ActorTF, self).__init__()

        # Explicit state input names
        self.state_inputs = {
            "x": tf.keras.Input(shape=(1,), name="x"),
            "y": tf.keras.Input(shape=(1,), name="y"),
            "theta": tf.keras.Input(shape=(1,), name="theta"),
            "theta_dot": tf.keras.Input(shape=(1,), name="theta_dot"),
            "alpha": tf.keras.Input(shape=(1,), name="alpha")
        }

        # Hidden layers
        self.hidden_layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(flax_actor.hidden_dim, activation='relu', name=f'Actor_Hidden_Layer_{i+1}')
             for i in range(flax_actor.number_of_hidden_layers)]
        )

        # Output layers
        self.mean_layer = tf.keras.layers.Dense(flax_actor.action_dim, activation='tanh', name='Mean_Action')
        self.std_layer = tf.keras.layers.Dense(flax_actor.action_dim, activation='softplus', name='Std_Deviation')

    def call(self, inputs):
        # Concatenate states
        state_concat = tf.concat([inputs["x"], inputs["y"], inputs["theta"], inputs["theta_dot"], inputs["alpha"]],
                                 axis=-1, name="Concatenated_State")

        x = self.hidden_layers(state_concat)
        mean = self.mean_layer(x)
        std = self.std_layer(x)
        return mean, std

### CRITIC ###
class DoubleCriticTF(tf.keras.Model):
    def __init__(self, flax_critic: DoubleCritic):
        super(DoubleCriticTF, self).__init__()

        # Explicit state and action input names
        self.state_inputs = {
            "x": tf.keras.Input(shape=(1,), name="x"),
            "y": tf.keras.Input(shape=(1,), name="y"),
            "theta": tf.keras.Input(shape=(1,), name="theta"),
            "theta_dot": tf.keras.Input(shape=(1,), name="theta_dot"),
            "alpha": tf.keras.Input(shape=(1,), name="alpha")
        }
        
        self.action_inputs = {
            "u0": tf.keras.Input(shape=(1,), name="u0"),
            "u1": tf.keras.Input(shape=(1,), name="u1"),
            "u2": tf.keras.Input(shape=(1,), name="u2")
        }

        # Hidden layers
        self.hidden_layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(flax_critic.hidden_dim, activation='relu',
                                   name=f'Critic_Hidden_Layer_{i+1}')
             for i in range(flax_critic.number_of_hidden_layers)]
        )

        # Output layers
        self.q1_output = tf.keras.layers.Dense(1, name='Q1_Output')
        self.q2_output = tf.keras.layers.Dense(1, name='Q2_Output')

    def call(self, state_inputs, action_inputs):
        # Concatenate states and actions separately for clarity
        state_concat = tf.concat([state_inputs["x"], state_inputs["y"], state_inputs["theta"], 
                                  state_inputs["theta_dot"], state_inputs["alpha"]], 
                                  axis=-1, name="Concatenated_State")
        
        action_concat = tf.concat([action_inputs["u0"], action_inputs["u1"], action_inputs["u2"]],
                                  axis=-1, name="Concatenated_Action")

        # Final concatenation for full input
        x = tf.concat([state_concat, action_concat], axis=-1, name="State_Action_Concat")

        x = self.hidden_layers(x)
        q1 = self.q1_output(x)
        q2 = self.q2_output(x)
        return q1, q2

### GRAPH WRITER FUNCTION ###
def write_graph(writer, flax_actor: Actor, flax_critic: DoubleCritic):
    
    actor_tf = ActorTF(flax_actor)
    critic_tf = DoubleCriticTF(flax_critic)

    # Enable tracing
    tf.summary.trace_on(graph=True, profiler=False)

    # Generate dummy inputs with correct names and shapes
    dummy_state = {key: tf.random.normal((1, 1), name=key) for key in ["x", "y", "theta", "theta_dot", "alpha"]}
    dummy_action = {key: tf.random.normal((1, 1), name=key) for key in ["u0", "u1", "u2"]}

    # Perform a forward pass to capture the graph
    _ = actor_tf(dummy_state)
    _ = critic_tf(dummy_state, dummy_action)

    # Export Actor Graph
    with writer:
        tf.summary.trace_export(name="Actor_Graph", step=0)

    # Export Critic Graph
    tf.summary.trace_on(graph=True, profiler=False)
    with writer:
        tf.summary.trace_export(name="Critic_Graph", step=0)

    writer.flush()

    print(f"TensorBoard network graphs saved.")