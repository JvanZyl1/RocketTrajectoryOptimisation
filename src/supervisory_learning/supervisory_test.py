import matplotlib.pyplot as plt
from tqdm import tqdm

from src.agents.functions.networks import Actor

def endo_ascent_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase in ['subsonic', 'supersonic']
    batch_size_tester = 552
    u0_learnt = []
    u1_learnt = []
    u2_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean, _ = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())
        u1_learnt.extend(output_values[:, 1].tolist())
        u2_learnt.extend(output_values[:, 2].tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(reference_data['u0'][1:-2], label='PID')
    plt.plot(u0_learnt[1:-2], linestyle='--', label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('u0')
    plt.title('u0 Over Time (gimballing)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/u0Imitation.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(reference_data['u1'][1:-2], label='PID')
    plt.plot(u1_learnt[1:-2], linestyle='--', label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('u1')
    plt.title('u1 Over Time (throttle)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/u1Imitation.png')
    plt.close()
    

def flip_over_boostbackburn_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase == 'flip_over_boostbackburn'
    batch_size_tester = 552
    u0_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean, _ = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(reference_data['u0'][1:-2], label='PID')
    plt.plot(u0_learnt[1:-2], linestyle='--', label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('u0')
    plt.title('u0 Over Time (gimballing)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/u0Imitation.png')
    plt.close()
    