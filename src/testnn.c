/**
 * @file
 * Toy neural networks to demonstrate the library.
 *
 * @author Nathan Geffen
 * @copyright The BSD 2-Clause License
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include "nn/nn.h"

void stress ()
{
	struct ann_net *ann;
	int layers[] = {2560, 2560, 2000, 1000, 1000,  2000}, i, j;
	double patterns[10][2560];
	clock_t start, end;

	start = clock();
	for (i = 0; i < 10; ++ i)
		for (j = 0; j < 256; ++j)
			patterns[i][j] = (i + j % 2 == 0) ? 1 : 0;
	end = clock();
	printf("D0 Time taken: %.2f\n",
	       (double)(end - start) / CLOCKS_PER_SEC);
	start = clock();
	ann = ann_create_feed_forward_net(layers, 6);
	end = clock();
	printf("D1 Time taken: %.2f\n",
	       (double)(end - start) / CLOCKS_PER_SEC);
	start = clock();
	ann_process_pattern(ann, patterns[0], 2560);
	end = clock();
	printf("D2 Time taken: %.2f\n",
	       (double)(end - start) / CLOCKS_PER_SEC);
	start = clock();
	ann_destroy_net(ann);
	end = clock();
	printf("D3 Time taken: %.2f\n",
	       (double)(end - start) / CLOCKS_PER_SEC);
}

int main(int argc, char *argv[])
{
	struct ann_net *ann;
	struct ann_layer *output_layer, *layer, *bias_layer;
	struct ann_neuron *neuron;
	FILE *f;
	int num_nets;

	int layers[] = {2, 2, 1};
	double patterns[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

	ann = ann_create_feed_forward_net(layers, 3);
	assert(ann_check_net(ann));

	// Set bias connections
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 1, 0),
					       ann_get_neuron_by_pos(ann, 2, 0)),
			       -2.82);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 1, 0),
					       ann_get_neuron_by_pos(ann, 2, 1)),
			       -2.74);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 1, 0),
					       ann_get_neuron_by_pos(ann, 3, 0)),
			       -2.86);
	// Input to hidden
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 0, 0),
					       ann_get_neuron_by_pos(ann, 2, 0)),
			       4.83);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 0, 0),
					       ann_get_neuron_by_pos(ann, 2, 1)),
			       -4.63);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 0, 1),
					       ann_get_neuron_by_pos(ann, 2, 0)),
			       -4.83);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 0, 1),
					       ann_get_neuron_by_pos(ann, 2, 1)),
			       4.6);
	// Hidden to output
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 2, 0),
					       ann_get_neuron_by_pos(ann, 3, 0)),
			       5.73);
	ann_set_synapse_weight(ann_get_synapse(ann_get_neuron_by_pos(ann, 2, 1),
					       ann_get_neuron_by_pos(ann, 3, 0)),
			       5.83);
	printf("***************\n");
	assert(ann_check_net(ann));
	printf("PRINTING FIRST NET.\n");
	ann_print_net(ann);
	printf("FINISHED PRINTING FIRST NET.\n");
	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		printf("*************************\n"
		       "Pattern\t%d:", i);
		for (int j = 0; j < 2; ++j)
			printf("\t%.2f", patterns[i][j]);
		printf("\n");
		ann_print_layer_outputs(output_layer);
	}
	ann_set_net_name(ann, "XOR");
	assert(f = fopen("ann.json", "w"));
	assert(ann_save(f, &ann, 1));
	fclose(f);
	assert(ann_save_nets_bin("output.bin", &ann, 1));
	ann_destroy_net(ann);
	assert(f = fopen("ann.json", "rb"));

	printf("********************\n");
	printf("Loading and reprinting net\n");
	struct ann_net **ann_arr;
	assert(ann_arr = ann_load(f, &num_nets));
	assert(num_nets == 1);
	fclose(f);
	ann_check_net(ann_arr[0]);
	ann_print_net(ann_arr[0]);
	ann = ann_arr[0];
	printf("********************\n");
	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		printf("*************************\n"
		       "Pattern\t%d:", i);
		for (int j = 0; j < 2; ++j)
			printf("\t%.2f", patterns[i][j]);
		printf("\n");
		ann_print_layer_outputs(output_layer);
	}
	printf("********************\n");
	ann_destroy_net(ann_arr[0]);
	free(ann_arr);

	printf("********************\n");
	printf("Loading and reprinting net from binary file\n");

	assert(ann_arr = ann_load_nets_bin("output.bin", &num_nets));
	assert(num_nets == 1);
	ann_check_net(ann_arr[0]);
	ann_print_net(ann_arr[0]);
	ann = ann_arr[0];
	printf("********************\n");
	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		printf("*************************\n"
		       "Pattern\t%d:", i);
		for (int j = 0; j < 2; ++j)
			printf("\t%.2f", patterns[i][j]);
		printf("\n");
		ann_print_layer_outputs(output_layer);
	}
	printf("********************\n");
	ann_destroy_net(ann_arr[0]);
	free(ann_arr);


	printf("***************\nInitialising second net\n");

	// Create net
	ann = ann_create_net();
	assert(ann_check_net(ann));

	// Create input layer
	layer = ann_add_layer(ann);
	assert(layer);
	assert(ann_check_net(ann));

	// Create two input layer neurons
	neuron = ann_add_neurons(layer, 2, ann_neuron_fire_input, NULL);
	assert(neuron);
	assert(ann_check_net(ann));

	// Create bias layer
	bias_layer = ann_add_layer(ann);
	assert(bias_layer);
	assert(ann_check_net(ann));

	// Create bias neuron
	neuron = ann_add_neurons(layer, 1, ann_neuron_fire_bias, NULL);
	assert(neuron);
	assert(ann_check_net(ann));

	// Create hidden layer
	layer = ann_add_layer(ann);
	assert(layer);
	assert(ann_check_net(ann));

	// Create two hidden layer neurons
	neuron = ann_add_neurons(layer, 2, ann_neuron_fire_sigmoid, NULL);
	assert(neuron);
	assert(ann_check_net(ann));

	// Connect bias to hidden layer neurons
	assert(ann_connect_layers(bias_layer, layer));
	assert(ann_check_net(ann));

	// Connect input layer (layer->prev->prev) to hidden layer
	assert(ann_connect_layers(layer->prev->prev, layer));
	assert(ann_check_net(ann));

	// Create output layer
	layer = ann_add_layer(ann);
	assert(layer);
	assert(ann_check_net(ann));

	// Create output layer neuron
	neuron = ann_add_neuron(layer, ann_neuron_fire_sigmoid, NULL);
	assert(neuron);
	assert(ann_check_net(ann));

	// Connect bias to output layer neuron
	assert(ann_connect_layers(bias_layer, layer));
	assert(ann_check_net(ann));

	// Connect hidden layer to output layer
	assert(ann_connect_layers(layer->prev, layer)); //layer->prev is hidden
	assert(ann_check_net(ann));

	printf("*****\n");
	printf("Printing second net.\n");
	ann_print_net(ann);
	printf("*****\n");

	ann_destroy_net(ann);

	stress();

	return 0;
}
