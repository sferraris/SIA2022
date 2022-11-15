import matplotlib
import numpy
from matplotlib import pyplot as plt
import tp2.main as funcs


from main import autoencoder_run
from main import p_forward
from tp2.declarations import Point
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def graph_results_latent_space():
    inner_layers = 4
    nodes_count = 4
    output_nodes = 2
    b = 0.1
    n = 0.1
    cot = 100
    layers = [35, 20, 8, 2, 8, 20, 35]
    layers_aux = [35, 20, 8, 2]
    c = 0x60
    weights, errors, epocas, accuracy_array, initial_points = autoencoder_run(cot, n, b, False, False, False, 0.2, 0.5,
                    layers, False)



    latent_points_x = []
    latent_points_y = []
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers_aux, weights, p,
                                                                 b)
        #latent_points_x.append(o_dictionary_encoder[len(layers_aux)-1][0])
        #latent_points_y.append(o_dictionary_encoder[len(layers_aux) - 1][1])
        #latent_points.append(o_dictionary_encoder[])
        x = o_dictionary_encoder[len(layers_aux)-1][0]
        y = o_dictionary_encoder[len(layers_aux) - 1][1]
        plt.scatter([x], [y])
        plt.text(x, y, chr(c))
        c += 1
    print("hola2")
    #plt.scatter(latent_points_x, latent_points_y)
    plt.show()

def show_letters():
    inner_layers = 4
    nodes_count = 4
    output_nodes = 2
    b = 0.1
    n = 0.1
    cot = 1
    layers = [35, 20, 8, 2, 8, 20, 35]
    weights, errors, epocas, accuracy_array, initial_points = autoencoder_run(cot, n, b, False, False, False, 0.2, 0.5,
                    layers, False)



    latent_points_x = []
    latent_points_y = []
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p,
                                                                 b)
        #latent_points_x.append(o_dictionary_encoder[len(layers_aux)-1][0])
        #latent_points_y.append(o_dictionary_encoder[len(layers_aux) - 1][1])
        #latent_points.append(o_dictionary_encoder[])
        font_char = p.e[1:]
        s = ""
        for char in range(35):
            if char % 5 == 0 and char != 0:
                print(s)
                s = ""
            c = 'X' if font_char[char] > 0 else '.'
            s = s + c
        print(s)
        print()

        """
        for row in range(7):
            row_n = row + 1
            s = ""
            for col in range(5):
                col_n = col+1
                c = 'X' if font_char[row_n*col_n] > 0 else '.'
                s = c + s
            print(s)
        print(" ")
        """
        for row in range(7):
            s = ""
            for col in range(5):
                c = 'X' if o_dictionary_encoder[len(layers)-1][(row + 1) * col + 1] > 0 else ' '
                s = s + c
            print(s)
        print(" ")



def main():
    show_letters()

if __name__ == "__main__":
    main()
