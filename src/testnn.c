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
#include <math.h>
#include "nn/nn.h"
#include "test.h"

void stress()
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
	struct ann_net *ann, **ann_arr;
	struct ann_layer *output_layer, *layer, *bias_layer;
	struct ann_neuron *neuron;
	struct test_series *t;
	FILE *f;
	int num_nets;
	int layers[] = {2, 2, 1};
	double patterns[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

	t = test_init("Test ann", true, NULL);
	if (t == NULL) {
		fprintf(stderr, "Cannot start testing.");
		return EXIT_FAILURE;
	}

	ann = ann_create_feed_forward_net(layers, 3);
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after create feed forward", "%d");

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
	ann_set_net_name(ann, "XOR");
	TESTEQ(t, ann_check_net(ann), true, "valid net", "%d");

	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		TESTEQ(t, output_layer->num_neurons, 1,
		       "Number neurons in output layer.", "%d")
		switch(i) {
		case 0:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01, "Xor 0 0 == 0.1", "%f");
			break;
		case 1:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.9),
			       0.01, "Xor 0 1 == 0.9", "%f");
			break;
		case 2:
			TESTLT(t, output_layer->neuron_head->value - 0.9,
			       0.01, "Xor 1 0 == 0.9", "%f");
			break;
		case 3:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01,  "Xor 0 0 == 0.1", "%f");
			break;
		}
	}
	f = fopen("ann.json", "w");
	if (f == NULL) {
		fprintf(stderr, "Cannot open json file for writing.\n");
		return EXIT_FAILURE;
	}

	/* Test JSON save and load. */
	TESTEQ(t, ann_save(f, &ann, 1), true,
	       "Save ann.", "%d");
	fclose(f);
	ann_destroy_net(ann);
	f = fopen("ann.json", "r");
	if (f == NULL) {
		fprintf(stderr, "Cannot open json file for reading.\n");
		return EXIT_FAILURE;
	}
	ann_arr = ann_load(f, &num_nets);
	TESTEQ(t, num_nets, 1, "Number of nets equals 1.", "%d");
	ann = ann_arr[0];
	TESTEQ(t, ann_check_net(ann), true, "valid net after json load", "%d");
	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		TESTEQ(t, output_layer->num_neurons, 1,
		       "Number neurons in output layer.", "%d")
		switch(i) {
		case 0:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01, "Xor 0 0 == 0.1", "%f");
			break;
		case 1:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.9),
			       0.01, "Xor 0 1 == 0.9", "%f");
			break;
		case 2:
			TESTLT(t, output_layer->neuron_head->value - 0.9,
			       0.01, "Xor 1 0 == 0.9", "%f");
			break;
		case 3:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01,  "Xor 0 0 == 0.1", "%f");
			break;
		}
	}

	/* Test save and load from binary file. */

	TESTEQ(t, ann_save_nets_bin("output.bin", ann_arr, 1), true,
	       "Saved net to binary file.", "%d");
	ann_destroy_net(ann_arr[0]);
	ann_arr = ann_load_nets_bin("output.bin", &num_nets);
	TESTEQ(t, num_nets, 1, "Number of nets equals 1.", "%d");
	ann = ann_arr[0];
	TESTEQ(t, ann_check_net(ann), true, "valid net after json load", "%d");
	for (int i = 0; i < 4; ++i) {
		output_layer = ann_process_pattern(ann, patterns[i], 2);
		TESTEQ(t, output_layer->num_neurons, 1,
		       "Number neurons in output layer.", "%d")
		switch(i) {
		case 0:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01, "Xor 0 0 == 0.1", "%f");
			break;
		case 1:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.9),
			       0.01, "Xor 0 1 == 0.9", "%f");
			break;
		case 2:
			TESTLT(t, output_layer->neuron_head->value - 0.9,
			       0.01, "Xor 1 0 == 0.9", "%f");
			break;
		case 3:
			TESTLT(t, fabs(output_layer->neuron_head->value - 0.1),
			       0.01,  "Xor 0 0 == 0.1", "%f");
			break;
		}
	}
	ann_destroy_net(ann_arr[0]);
	free(ann_arr);

	// Create net
	ann = ann_create_net();
	ann->name = "Multilayered feedforward neural network";
	ann->description = "Test network for libnn";
	TESTEQ(t, ann_check_net(ann), true, "Valid net after create.", "%d");

	// Create input layer
	layer = ann_add_layer(ann);
	if (layer == NULL) {
		fprintf(stderr, "Cannot create layer.x\n");
		return EXIT_FAILURE;
	}
	layer->label = "input";
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after input layer created.", "%d");

	// Create two input layer neurons
	neuron = ann_add_neurons(layer, 2, ann_neuron_fire_input, NULL);
	if (neuron == NULL) {
		fprintf(stderr, "Cannot create input neurons.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after input neurons created.", "%d");

	// Create bias layer
	bias_layer = ann_add_layer(ann);
	if (bias_layer == NULL) {
		fprintf(stderr, "Cannot create input neurons.\n");
		return EXIT_FAILURE;
	}
	bias_layer->label = "bias";
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after bias layer created.", "%d");


	// Create bias neuron
	neuron = ann_add_neurons(bias_layer, 1, ann_neuron_fire_bias, NULL);
	if (neuron == NULL) {
		fprintf(stderr, "Cannot create bias neuron.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after bias neuron created.", "%d");

	// Create first hidden layer
	layer = ann_add_layer(ann);
	if (layer == NULL) {
		fprintf(stderr, "Cannot create first hidden layer.\n");
		return EXIT_FAILURE;
	}
	layer->label = "first hidden layer";
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after first hidden layer created.", "%d");

	// Create two hidden layer neurons
	neuron = ann_add_neurons(layer, 2, ann_neuron_fire_sigmoid, NULL);
	if (neuron == NULL) {
		fprintf(stderr, "Cannot create first hidden layer neurons.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after first layer hidden neurons created.", "%d");

	// Connect bias to hidden layer neurons
	if (ann_connect_layers(bias_layer, layer) == false) {
		fprintf(stderr, "Cannot connect bias to first hidden layer");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after bias connected to first hidden layer.", "%d");

	// Connect input layer (layer->prev->prev) to hidden layer
	if (ann_connect_layers(layer->prev->prev, layer) == false) {
		fprintf(stderr, "Cannot connect input layer to hidden layer.");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after input connected to first hidden layer.", "%d");


	// Create second hidden layer
	layer = ann_add_layer(ann);
	if (layer == NULL) {
		fprintf(stderr, "Cannot create second hidden layer.\n");
		return EXIT_FAILURE;
	}
	layer->label = "second hidden layer";
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after second hidden layer created.", "%d");

	// Create two second layer hidden layer neurons
	neuron = ann_add_neurons(layer, 3, ann_neuron_fire_sigmoid, NULL);
	if (neuron == NULL) {
		fprintf(stderr, "Cannot create second hidden layer neurons.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after second layer hidden neurons created.", "%d");

	// Connect bias to second hidden layer neurons
	if (ann_connect_layers(bias_layer, layer) == false) {
		fprintf(stderr, "Cannot connect bias to second hidden layer.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after bias connected to second hidden layer.", "%d");

	// Connect first hidden layer to second hidden layer
	if (ann_connect_layers(layer->prev, layer) == false) {
		fprintf(stderr, "Cannot connect hidden layers.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after hidden layers connected.", "%d");

	// Create output layer
	layer = ann_add_layer(ann);
	if (layer == NULL) {
		fprintf(stderr, "Cannot create output layer.\n");
		return EXIT_FAILURE;
	}
	layer->label = "output";
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after output layer created.", "%d");

	// Create output neurons
	neuron = ann_add_neurons(layer, 2, ann_neuron_fire_sigmoid, NULL);
	if (neuron == NULL) {
		fprintf(stderr, "Cannot create output neurons.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after output neurons created.", "%d");

	// Connect bias to output neurons
	if (ann_connect_layers(bias_layer, layer) == false) {
		fprintf(stderr, "Cannot connect bias to output layer.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after bias connected to output layer.", "%d");

	// Connect second hidden layer to output neurons
	if (ann_connect_layers(layer->prev, layer) == false) {
		fprintf(stderr, "Cannot connect hidden to output layer.\n");
		return EXIT_FAILURE;
	}
	TESTEQ(t, ann_check_net(ann), true,
	       "Valid net after hidden layer connected to output layer.", "%d");


	ann_print_net(ann);


	ann_destroy_net(ann);

	/* stress(); */

	test_summary(t);
	return 0;
}
