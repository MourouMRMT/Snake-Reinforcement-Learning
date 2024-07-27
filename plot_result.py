import matplotlib.pyplot as plt
from IPython import display
from torchviz import make_dot
import torch

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Snake AI Training')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def show_CNN(agent,state):

    # Convertir l'état en tenseur
    state_tensor = torch.tensor(state, dtype=torch.float)

    # Obtenir la prédiction du modèle pour cet état
    prediction = agent.model(state_tensor)

    # Créer le graphe
    dot = make_dot(prediction, params=dict(agent.model.named_parameters()))

    # Afficher le graphe
    dot.render("network_structure", format="png")  # Enregistrer le graphe sous forme d'image
    dot.view()  # Afficher le graphe