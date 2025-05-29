# Deploying Rocket Trajectory Optimization to Google Cloud

This guide explains how to deploy this project to Google Cloud for running larger simulations or sharing results.

## Prerequisites

1. A Google Cloud account with billing enabled
2. Google Cloud SDK installed locally
3. Basic familiarity with Google Cloud services

## Recommended Google Cloud Services

### Option 1: Google Compute Engine (VM)

For running simulations and optimization:

1. **Create a VM instance**:
   - Use a Compute Engine instance with sufficient CPU/RAM (e.g., n1-standard-4 or better)
   - Select a machine with GPU if using PyTorch for deep learning components
   - Use Ubuntu 20.04 LTS or later

2. **Setup steps**:
   ```bash
   # SSH into your VM
   gcloud compute ssh YOUR_INSTANCE_NAME
   
   # Clone repository
   git clone https://github.com/YOUR_USERNAME/RocketTrajectoryOptimisation.git
   cd RocketTrajectoryOptimisation
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run your simulations
   python main_particle_swarm_optimisation.py
   ```

3. **Access TensorBoard remotely**:
   ```bash
   # On the VM, start TensorBoard
   cd data/pso_saves/{flight_phase}
   tensorboard --logdir=runs --host 0.0.0.0
   
   # Alternatively, use ngrok as in your local setup
   ngrok http 6006
   ```

### Option 2: Google AI Platform/Vertex AI

For managed ML training:

1. Package your code for AI Platform training
2. Use a config file to specify resources
3. Submit training jobs via Google Cloud CLI

## Data Storage Considerations

- Use **Google Cloud Storage** for saving simulation results
- Example bucket setup:
  ```bash
  gsutil mb gs://rocket-trajectory-optimization
  ```

- Update your code to save to GCS:
  ```python
  # Example modification for cloud storage
  save_path = 'gs://rocket-trajectory-optimization/pso_saves/{flight_phase}'
  ```

## Cost Management

- Use preemptible VMs to reduce costs (but handle interruptions)
- Set budget alerts
- Shut down resources when not in use

## Security Considerations

- Use service accounts with minimal permissions
- Set up firewall rules to restrict access
- Consider using Cloud IAM for access control

## Next Steps

1. Create a small test VM first to validate your setup
2. Monitor resource usage to optimize instance sizing
3. Consider containerizing your application with Docker for more consistent deployment 