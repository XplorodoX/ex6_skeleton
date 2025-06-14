import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Das neuronale Netzwerk zur Approximation der Q-Funktion."""

    def __init__(self, num_obs, num_actions, hidden_size=4):
        """
        Parameter:
        ----------
        num_obs : int
            Dimensionalität des Beobachtungsraums.
        num_actions : int
            Anzahl möglicher Aktionen.
        hidden_size : int
            Größe der versteckten Schichten.
        """
        super(NeuralNetwork, self).__init__()

        # Sequential-Container für effizientere Implementierung
        self.network = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        """Forward-Pass durch das Netzwerk."""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Batch-Dimension hinzufügen, falls nötig

        return self.network(x)