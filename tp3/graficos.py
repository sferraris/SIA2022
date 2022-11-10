import numpy
from matplotlib import pyplot as plt
import tp2.main as funcs


from main import autoencoder_run

def graph_results_latent_space():
    inner_layers = 4
    nodes_count = 4
    output_nodes = 2
    b = 0.1
    n = 0.1
    cot = 100
    min_weights_encoder, min_weights_decoder, errors, epocas, accuracy_array, initial_points = autoencoder_run(cot, n, b, inner_layers, nodes_count, output_nodes)


    latent_points_x = []
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = funcs.p_forward(inner_layers, min_weights_encoder, nodes_count,
                                                                 output_nodes, p, b)
        #latent_points.append(o_dictionary_encoder[])




def main():
    graph_results_latent_space()

if __name__ == "__main__":
    main()
