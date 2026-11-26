"""Client-side code for local federated learning experiments.

Runs one Flower client: loads its configured local dataset, trains the
shared model for a round, and returns updated weights to the server.
"""
