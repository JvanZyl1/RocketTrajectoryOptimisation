import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from src.agents.functions.networks import ClassicalActor as Actor

def endo_ascent_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase in ['subsonic', 'supersonic']
    batch_size_tester = 552
    u0_learnt = []
    u1_learnt = []
    u2_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())
        u1_learnt.extend(output_values[:, 1].tolist())
        u2_learnt.extend(output_values[:, 2].tolist())

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Action Imitation', fontsize=32)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.5, wspace=0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(reference_data['u0'][1:-2],linestyle='--', label='Controller', color='red', linewidth=4)
    ax1.plot(u0_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax1.set_xlabel('Step', fontsize=20)
    ax1.set_ylabel('Action [-1, 1]', fontsize=20)
    ax1.set_title('Gimballing Action', fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(reference_data['u1'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax2.plot(u1_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax2.set_xlabel('Step', fontsize=20)
    ax2.set_ylabel('Action [-1, 1]', fontsize=20)
    ax2.set_title('Throttle Action', fontsize=22)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/ActionImitation.png')
    plt.close()
    

def flip_over_boostbackburn_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase == 'flip_over_boostbackburn'
    batch_size_tester = 552
    u0_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Action Imitation', fontsize=32)
    gs = gridspec.GridSpec(1, 1, height_ratios=[1], hspace=0.5, wspace=0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(reference_data['u0'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax1.plot(u0_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax1.set_xlabel('Step', fontsize=20)
    ax1.set_ylabel('Action [-1, 1]', fontsize=20)
    ax1.set_title('Gimballing Action', fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/ActionImitation.png')
    plt.close()


def ballistic_arc_descent_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase == 'ballistic_arc_descent'
    batch_size_tester = 552
    u0_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Action Imitation', fontsize=32)
    gs = gridspec.GridSpec(1, 1, height_ratios=[1], hspace=0.5, wspace=0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(reference_data['u0'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax1.plot(u0_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax1.set_xlabel('Step', fontsize=20)
    ax1.set_ylabel('Action [-1, 1]', fontsize=20)
    ax1.set_title('RCS Throttle', fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/ActionImitation.png')
    plt.close()
    
def landing_burn_supervisory_test(inputs, flight_phase, state_network, targets, hidden_dim, number_of_hidden_layers, reference_data):
    assert flight_phase == 'landing_burn'
    batch_size_tester = 552
    u0_learnt = []
    u1_learnt = []
    u2_learnt = []
    u3_learnt = []
    num_batches = len(inputs) // batch_size_tester + (1 if len(inputs) % batch_size_tester != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size_tester:(i+1)*batch_size_tester]
        mean = Actor(action_dim=targets.shape[1],
                            hidden_dim=hidden_dim,
                            number_of_hidden_layers=number_of_hidden_layers).apply({'params': state_network.params}, batch_inputs)
        output_values = mean
        u0_learnt.extend(output_values[:, 0].tolist())
        u1_learnt.extend(output_values[:, 1].tolist())
        u2_learnt.extend(output_values[:, 2].tolist())
        u3_learnt.extend(output_values[:, 3].tolist())
        
    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Action Imitation', fontsize=32)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.45, wspace=0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(reference_data['u0'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax1.plot(u0_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax1.set_xlabel('Step', fontsize=20)
    ax1.set_ylabel('Action [-1, 1]', fontsize=20)
    ax1.set_title('Gimballing Action', fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(reference_data['u1'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax2.plot(u1_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax2.set_xlabel('Step', fontsize=20)
    ax2.set_ylabel('Action [-1, 1]', fontsize=20)
    ax2.set_title('Throttle Action', fontsize=22)
    ax2.grid(True)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(reference_data['u2'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax3.plot(u2_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax3.set_xlabel('Step', fontsize=20)
    ax3.set_ylabel('Action [-1, 1]', fontsize=20)
    ax3.set_title('Left grid fin deflection', fontsize=22)
    ax3.grid(True)
    ax3.legend(fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=18)

    ax4 = plt.subplot(gs[3, 0])
    ax4.plot(reference_data['u3'][1:-2], linestyle='--', label='Controller', color='red', linewidth=4)
    ax4.plot(u3_learnt[1:-2], label='Neural Network', color='blue', linewidth=1)
    ax4.set_xlabel('Step', fontsize=20)
    ax4.set_ylabel('Action [-1, 1]', fontsize=20)
    ax4.set_title('Right grid fin deflection', fontsize=22)
    ax4.grid(True)
    ax4.legend(fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f'results/SupervisoryLearning/{flight_phase}/ActionImitation.png')
    plt.close()