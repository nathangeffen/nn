/**
 * @file
 * Functions for initializing, modifying, running, and destroying a neural
 * network, its layers, neurons and synapses.
 *
 * @author Nathan Geffen
 * @copyright The BSD 2-Clause License
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "nn.h"

/**
 * Creates and initialises a neural network.
 *
 * @return A properly initialised neural network on success, else NULL.
 */

struct ann_net *ann_create_net(void)
{
	struct ann_net *ann;

	ann = malloc(sizeof(*ann));
	ANN_CHECK(ann, "Cannot allocate neural network.", return NULL);
	ann->name = ann->description = NULL;
	ann->free_name = ann->free_description = false;
	ann->layer_head = ann->layer_last = NULL;
	ann->min_weight = MIN_WEIGHT;
	ann->max_weight = MAX_WEIGHT;
	ann->layer_ctr = ann->num_layers = 0;
	ann->rng = gsl_rng_alloc(gsl_rng_mt19937);
	ANN_CHECK(ann->rng, "Cannot allocate random number generator",
		  {
			  ann_destroy_net(ann);
			  return NULL;
		  });
	return ann;
}

char *ann_set_net_name(struct ann_net *ann, const char *name)
{
	char *c = malloc(sizeof(*c) * strlen(name) + 1);
	ANN_CHECK(c, "No memory to allocate net name.", return NULL);
	strcpy(c, name);
	if (ann->name) free(ann->name);
	ann->name = c;
	ann->free_name = true;
	return c;
}

char *ann_set_net_description(struct ann_net *ann, const char *desc)
{
	char *c = malloc(sizeof(*c) * strlen(desc) + 1);
	ANN_CHECK(c, "No memory to allocate net description.", return NULL);
	strcpy(c, desc);
	if (ann->description) free(ann->description);
	ann->description = c;
	ann->free_description = true;
	return c;
}


inline void ann_net_set_min_weight(struct ann_net *ann, const double weight)
{
	ann->min_weight = weight;
}

inline void ann_net_set_max_weight(struct ann_net *ann, const double weight)
{
	ann->max_weight = weight;
}

inline void ann_net_set_weights(struct ann_net *ann,
				const double min_weight,
				const double max_weight)
{
	ann_net_set_min_weight(ann, min_weight);
	ann_net_set_max_weight(ann, max_weight);
}

inline void ann_layer_set_min_weight(struct ann_layer *layer,
				     const double weight)
{
	layer->min_weight = weight;
}

inline void ann_layer_set_max_weight(struct ann_layer *layer,
				     const double weight)
{
	layer->max_weight = weight;
}

inline void ann_layer_set_weights(struct ann_layer *layer,
				  const double min_weight,
				  const double max_weight)
{
	ann_layer_set_min_weight(layer, min_weight);
	ann_layer_set_max_weight(layer, max_weight);
}


/**
 * Generates a random value that can be used to set a synapse weight.
 *
 * @param weight_min minimum possible weight.
 * @param weight_max maximum possible weight.
 *
 * @return A real number in the semi-open range [min, max).
 */

double ann_get_weight(const gsl_rng *rng,
		      const double weight_min,
		      const double weight_max)
{
	return gsl_rng_uniform(rng) * (weight_max - weight_min) + weight_min;
}

/**
 * Calculates standard neuron sigmoid function: 1/(1+exp(-x).
 *
 * @param x Usually the total sum of inputs to a neuron.
 *
 * @return A real number between 0 and 1.
 */

inline double ann_sigmoid(const double x)
{
	return 1.0 / (1.0 + exp(-x));
}

/**
 * Calculates derivative of the neuron sigmoid function: x*(1-x).
 *
 * @param x Usually the total sum of inputs to a neuron.
 *
 * @return The derivative of sigmoid function.
 */

inline double ann_sigmoid_deriv(const double x)
{
	return x * (1.0 - x);
}


struct ann_synapse *ann_add_synapse(struct ann_neuron *from,
				    struct ann_neuron *to,
				    const double weight)
{
	struct ann_synapse *synapse;
	struct ann_synapse_list *head_from, *head_to;

	// Create the synapse
	synapse = malloc(sizeof(*synapse));
	ANN_CHECK(synapse, "Cannot allocate synapse.", return NULL);
	synapse->label = NULL;
	synapse->from = from;
	synapse->to = to;
	synapse->id = from->synapse_ctr++;
	synapse->weight = weight;

	// Now we allocate both head_from and head_to before connecting the
	// synapse, else reverting to correct state is too complex.
	head_from = malloc(sizeof(*head_from));
	ANN_CHECK(head_from, "Cannot allocate synapse list node.",
		  {
			free(synapse);
			return NULL;
		  });
	head_to =  malloc(sizeof(*head_to));
	ANN_CHECK(head_to, "Cannot allocate synapse list node.",
		  {
			  free(head_from);
			  free(synapse);
			  return NULL;
		  });

	// Connect the synapse to the from and to neurons output and input
	// synapse lists respectively.
	head_from->synapse = synapse;
	head_from->next = from->outputs;
	from->outputs = head_from;
	head_to->synapse = synapse;
	head_to->next = to->inputs;
        to->inputs = head_to;

	return synapse;
}

inline void ann_set_synapse_weight(struct ann_synapse *synapse,
				   const double weight)
{
	synapse->weight = weight;
}

inline void ann_synapse_modify_weight(struct ann_synapse *synapse,
				      const double weight)
{
	synapse->weight += weight;
}

inline double synapse_get_weight(const struct ann_synapse *synapse)
{
	return synapse->weight;
}

struct ann_synapse *ann_get_synapse(const struct ann_neuron *from,
				    const struct ann_neuron *to)
{
	struct ann_synapse_list *it = from->outputs;

	assert(it && it->synapse);

	while (it->synapse->to != to) {
		it = it->next;
		assert(it && it->synapse);
	}
	return it->synapse;
}

struct ann_synapse *ann_get_synapse_or_null(const struct ann_neuron *from,
					    const struct ann_neuron *to)
{
	struct ann_synapse_list *it = from->outputs;

	while (it && it->synapse->to != to)
		it = it->next;

	return it ? it->synapse : NULL;
}

/**
 * Calculates the inputs to a neuron and returns its firing value as
 * determined by the sigmoid function.
 *
 * @param neuron Neuron for which to calculate firing value.
 * @param data Unused user data for compatibility with other firing functions.
 */

void
ann_neuron_fire_sigmoid(struct ann_neuron *neuron,
			void *data)
{
	double total = 0.0;
	struct ann_synapse_list *it;
	for (it = neuron->inputs; it; it = it->next)
		total += it->synapse->weight * it->synapse->from->value;
	neuron->value = ann_sigmoid(total);
}

/**
 * Simple firing function that returns 1. Used by bias neurons.
 *
 * @param neuron Neuron for which to calculate firing value.
 * @param data Unused user data for compatibility with other firing functions.
 */

void
ann_neuron_fire_bias(struct ann_neuron *neuron,
		     void *data)
{
	neuron->value = 1.0;
}

/**
 * Simple firing function that returns the current value of the neuron
 * unchanged. Typically used by input neurons.
 *
 * @param neuron Neuron for which to calculate firing value.
 * @param data Unused user data for compatibility with other firing functions.
 */

inline void
ann_neuron_fire_input(struct ann_neuron *neuron, void *data)
{
	return;
}


/**
 * Standard feed forward neural network processing algorithm.
 *
 * Takes an input pattern, presents it to neural network and calculates the
 * outputs for all neurons in the network.
 *
 * @param ann Artificial neural network to process pattern.
 * @param inputs Array
 *
 * @return Derivative of sigmoid function
 */

struct ann_layer *
ann_process_pattern(struct ann_net *ann,
		    const double inputs[],
		    const int num_inputs)
{
	const double *f;
	struct ann_layer *layer, *last = NULL;
	struct ann_neuron *neuron;

	// Set up inputs
	for (f = inputs, neuron = ann->layer_head->neuron_head;
	     f < inputs + num_inputs && neuron;
	     ++f, neuron = neuron->next)
		neuron->value = *f;

	// Process subsequent layers
	for (layer = ann->layer_head->next; layer;
	     last = layer, layer = layer->next)
		for (neuron = layer->neuron_head; neuron; neuron = neuron->next)
			neuron->fire_func(neuron, NULL);

	return last;
}

/**
 * Allocates a new neuron on the heap and returns it.
 *
 * @return Newly created neuron upon success, else NULL.
 */

struct ann_neuron *ann_create_neuron()
{
	struct ann_neuron *neuron;

	neuron = malloc(sizeof(*neuron));
	ANN_CHECK(neuron, "Could not allocate neuron.", return NULL);
	neuron->layer = NULL;
	neuron->inputs = NULL;
	neuron->outputs = NULL;
	neuron->label = NULL;
	neuron->id = 0;
	neuron->value = 0.0;
	return neuron;
}

/**
 * Removes a synapse from a neuron's input or output list.
 *
 * @param synapse_list Neuron's synapse list from which the synapse must be
 * removed.
 * @param synapse Synapse to be removed.
 *
 */

void ann_remove_synapse_from_neuron(struct ann_synapse_list **synapse_list,
				    const struct ann_synapse *synapse)
{
	struct ann_synapse_list *it, *prev = NULL, *head = *synapse_list;
	for (it = head; it->synapse != synapse; it = it->next) {
		prev = it;
	}
	if (prev) {
		prev->next = it->next;
	} else {
		*synapse_list = (*synapse_list)->next;
	}
	free(it);
}

/**
 * Removes a synapse from its neurons and returns it to the heap.
 *
 * @param synapse Synapse to be destroyed.
 *
 */

void ann_destroy_synapse(struct ann_synapse *synapse)
{
	// First remove from input neuron list
	ann_remove_synapse_from_neuron(&synapse->from->outputs, synapse);
	// Then remove from output neuron list
	ann_remove_synapse_from_neuron(&synapse->to->inputs, synapse);
	free(synapse);
}

/**
 * Removes all the synapses in a synapse list, and returns all the
 * synapses and the synapse list to the heap.
 *
 * @param synapse_list Synapse list to be destroyed.
 *
 */

void
ann_destroy_synapse_list(struct ann_synapse_list *synapse_list,
			 bool free_synapse)
{
	struct ann_synapse_list *it, *next;
	for (it = synapse_list; it; it = next) {
		next = it->next;
		if (free_synapse)
			free(it->synapse);
		free(it);
		it = next;
	}
}

/**
 * Removes both the input and output synapses from a neuron, returning
 * them to the heap.
 *
 * @param neuron Neuron from which to remove the synapse lists.
 *
 */

void ann_destroy_synapse_lists(struct ann_neuron *neuron)
{
	ann_destroy_synapse_list(neuron->inputs, true);
	ann_destroy_synapse_list(neuron->outputs, false);
}

/**
 * Returns a neuron to the heap.
 *
 * @param neuron Neuron to destroy.
 *
 */

void ann_destroy_neuron(struct ann_neuron *neuron)
{
	ann_destroy_synapse_lists(neuron);
	free(neuron);
}


/**
 * Connects two neurons by creating a weighted synapse between them.
 *
 * @param from Neuron to connect synapse from.
 * @param to Neuron to connect synapse to.
 * @param weight Initial weight of the synapse.
 *
 * @return Snapse pointer upon success else NULL.
 */




/**
 * Finds a neuron, by id, in a neural network.
 *
 * @sa ann_find_neuron_by_pos
 *
 * @param ann Neural net to search.
 * @param layer_id Id of layer of neuron being searched for.
 * @param neuron_id Id of neuron being searched for.
 *
 * @return The found neuron, else NULL if none found.
 */

struct ann_neuron *ann_get_neuron_by_id(const struct ann_net *ann,
					const int layer_id,
					const int neuron_id)
{
	struct ann_layer *layer;
	struct ann_neuron *neuron;

	for (layer = ann->layer_head; layer; layer = layer->next)
		if (layer->id == layer_id)
			for (neuron = layer->neuron_head; neuron;
			     neuron = neuron->next)
				if (neuron->id == neuron_id)
					return neuron;

	return NULL;
}

/**
 * Finds a neuron, by position, in a neural network.
 *
 * The network is searched from the first layer (the input layer in a feed
 * forward network) to the last (the output layer in a feed forward
 * network). The first layer is considered to be layer position zero. The first
 * neuron in each layer is considered to be at neuron position zero.
 *
 * @sa ann_find_neuron_by_id
 *
 * @param ann Neural net to search.
 * @param layer_pos Position of layer of neuron being searched for,
 * relative to first layer, with first layer being zero.
 * @param neuron_pos Position of neuron being searched for, relative to first
 * neuron in layer, with first neuron being zero.
 *
 * @return The found neuron, else NULL if none found.
 */

struct ann_neuron *ann_get_neuron_by_pos(const struct ann_net *ann,
					 const int layer_pos,
					 const int neuron_pos)
{
	int i, j;
	struct ann_layer *layer;
	struct ann_neuron *neuron;

	for (i = 0, layer = ann->layer_head; layer && i <= layer_pos;
	     layer = layer->next, ++i)
		if (i == layer_pos)
			for (j = 0, neuron = layer->neuron_head;
			     j <= neuron_pos && neuron;
			     neuron = neuron->next, ++j)
				if (j == neuron_pos)
					return neuron;

	return NULL;
}


/**
 * Returns a list of neurons to the heap.
 *
 * @param head Neuron at head of list.
 */

void ann_destroy_neuron_list(struct ann_neuron *head)
{
	struct ann_neuron *it, *next;

	for (it = head; it; it = next) {
		next = it->next;
		ann_destroy_neuron(it);
	}
}

/**
 * Creates a layer of neurons.
 *
 * @param num_neurons Number of neurons in layer.
 * @param prev Input layer for this layer or NULL if this is an input layer.
 * @param next Layer to which this layer is connected or NULL if this is an
 * output layer.
 * @param rng Random number generator to be used to generate synapse weights
 * from this layer.
 * @param min_weight Minimum weight change for synapses from this layer.
 * @param max_weight Maximum weight change for synapses from this layer.
 *
 * @return synapse pointer upon success else NULL
 */

struct ann_layer *ann_create_layer(struct ann_layer *prev,
				   struct ann_layer *next,
				   gsl_rng *rng,
				   const double min_weight,
				   const double max_weight)
{
	struct ann_layer *layer;

	layer = malloc(sizeof(*layer));
	ANN_CHECK(layer, "Cannot allocate layer.", return NULL);
	layer->label = NULL;
	layer->neuron_ctr = layer->num_neurons = 0;
	layer->prev = prev;
	layer->next = next;
	layer->rng = rng;
	layer->min_weight = min_weight;
	layer->max_weight = max_weight;
	layer->neuron_head = NULL;
	layer->id = 0;
	if (prev)
		layer->prev->next = layer;
	if (next)
		layer->next->prev = layer;

	return layer;
}

struct ann_neuron *ann_add_neuron(struct ann_layer *layer,
				  void (*fire_func) (struct ann_neuron *,
						     void *),
				  void *data)
{
	struct ann_neuron *neuron;

	neuron = ann_create_neuron();
	ANN_CHECK(neuron, "Cannot create neuron.", return NULL);
	layer->num_neurons++;
	neuron->id = layer->neuron_ctr++;
	neuron->fire_func = fire_func;
	neuron->data = data;
	neuron->layer = layer;
	neuron->next = layer->neuron_head;
	layer->neuron_head = neuron;

	return neuron;
}

struct ann_neuron *ann_add_neurons(struct ann_layer *layer, int num_neurons,
				   void (*fire_func) (struct ann_neuron *,
						      void *),
				   void *data)
{
	struct ann_neuron *neuron = NULL;

	for (int i = 0; i < num_neurons; ++i) {
		neuron = ann_add_neuron(layer, fire_func, data);
		ANN_CHECK(neuron, "Could not add neuron.",
			  {
				  for (int j = 0; j < i; ++j)
					  ann_destroy_neuron(neuron);
				  return NULL;
			  });
	}
	return neuron;
}

struct ann_layer *ann_insert_after_layer(struct ann_net *ann,
					 struct ann_layer *layer)
{
	struct ann_layer *new_layer;

	new_layer = ann_create_layer(layer, layer->next, ann->rng,
				     ann->min_weight, ann->max_weight);
	return new_layer;
}

struct ann_layer *ann_prepend_layer(struct ann_net *ann)
{
	ann->layer_head = ann_create_layer(NULL, ann->layer_head,
					   ann->rng, ann->min_weight,
					   ann->max_weight);
	return ann->layer_head;
}

struct ann_layer *ann_add_layer(struct ann_net *ann)
{
	struct ann_layer *layer;

	layer = ann_create_layer(ann->layer_last, NULL, ann->rng,
				 ann->min_weight, ann->max_weight);
	ANN_CHECK(layer, "Could not add layer.", return NULL);
	layer->id = ann->layer_ctr++;
	++ann->num_layers;
	if (ann->layer_last == NULL)
		ann->layer_head = ann->layer_last = layer;
	else
		ann->layer_last = layer;
	return layer;
}


/**
 * Connects every neuron in a layer to its neighbouring layer.
 *
 * @param from The layer from which the synapses will originate.
 * @param to The layer to which the synapses will go.
 * @param weight_min Minimum of the randomly generated weight of the synapse.
 * @param weight_max Maximum of the randomly generated weight of the synapse.
 *
 * @return True upon success else false.
 */

bool ann_connect_layers(struct ann_layer *from,
			struct ann_layer *to)
{
	struct ann_neuron *i, *j;
	double weight;

	for (i = from->neuron_head; i; i = i->next)
		for (j = to->neuron_head; j; j = j->next) {
			weight = ann_get_weight(from->rng, from->min_weight,
						from->max_weight);
			ANN_CHECK(ann_add_synapse(i, j, weight),
				  "Failed to connect two neurons.",
				  return false);
		}
	return true;
}

/**
 * Return a layer, including all its neurons and synapses, to the heap.
 *
 * @param layer The layer to destroy.
 *
 */

void ann_destroy_layer(struct ann_layer *layer)
{
	if (layer->next)
		layer->next->prev = layer->prev;
	if (layer->prev)
		layer->prev->next = layer->next;
	ann_destroy_neuron_list(layer->neuron_head);
	free(layer);
}

struct ann_layer *
ann_add_feed_forward_layer(struct ann_net *ann,
			   void (*fire_func) (struct ann_neuron *, void *),
			   const int num_neurons)
{
	struct ann_layer *layer;

	layer = ann_add_layer(ann);
	ANN_CHECK(layer, "Could not allocate feed forward layer.", return NULL);
	ANN_CHECK(ann_add_neurons(layer, num_neurons, fire_func, NULL),
		  "Could not add layer to feedforward network.", return NULL);

	return layer;
}


/**
 * Allocates a fully connected feedforward neural network on the heap.
 *
 * @param layers Each element in this array contains the number of neurons in
 * the layer, with the first element being the number of neurons in the input
 * layer and the last element being the number of neurons in the output layer.
 * @param num_layers Number of entries in the layers array.
 *
 * @return A neural network allocated from the heap upon success, else NULL.
 *
 */

struct ann_net *ann_create_feed_forward_net(const int layers[],
					    const int num_layers)
{
	struct ann_net *ann;
	struct ann_layer *curr, *bias;

	ann = ann_create_net();
	ANN_CHECK(ann, "Cannot allocate feed forward net.", return NULL);

	// Input layer;
	curr = ann_add_feed_forward_layer(ann, ann_neuron_fire_input,
					  layers[0]);
	ANN_CHECK(curr, "Could not allocate input layer for feed forward net",
		  {
			  ann_destroy_net(ann);
			  return NULL;
		  });
	bias = ann_add_feed_forward_layer(ann, ann_neuron_fire_bias, 1);
	ANN_CHECK(bias, "Could not allocate bias layer",
		  {
			  ann_destroy_net(ann);
			  return NULL;
		  });
	// Hidden layers and output layer
	for (int i = 1; i < num_layers; ++i)  {
		curr = ann_add_feed_forward_layer(ann, ann_neuron_fire_sigmoid,
						  layers[i]);
		ANN_CHECK(curr, "Could not allocate feed forward layer.",
			  {
				  ann_destroy_net(ann);
				  return NULL;
			  });
		ANN_CHECK(ann_connect_layers(curr->prev, curr),
			  "Could not connect feed forward layers.",
			  {
				  ann_destroy_net(ann);
				  return NULL;
			  });
		if (i > 1) // Connect bias layer
			ANN_CHECK(ann_connect_layers(bias, curr),
				  "Could not connect feed forward layers.",
				  {
					  ann_destroy_net(ann);
					  return NULL;
				  });
		else // Connect input layer to first hidden layer
			ANN_CHECK(ann_connect_layers(ann->layer_head, curr),
				  "Could not connect feed forward layers.",
				  {
					  ann_destroy_net(ann);
					  return NULL;
				  });
	}
	return ann;
}

/**
 * Returns a neural network, and all its constituent parts, to the heap.
 *
 * @param ann Neural net to destroy.
 *
 */

void ann_destroy_net(struct ann_net *ann)
{
	struct ann_layer *it, *next;

	if (ann->free_name) free(ann->name);
	if (ann->free_description) free(ann->description);

	gsl_rng_free(ann->rng);
	for (it = ann->layer_head; it != NULL; it = next) {
		next = it->next;
		ann_destroy_layer(it);
	}
	free(ann);
}

/**
 *
 */

bool ann_traverse_neuron(const struct ann_neuron *neuron,
			 bool (* in_synapse_func)(const struct ann_synapse *,
						  void *data),
			 bool (* out_synapse_func)(const struct ann_synapse *,
						   void *data),
			 void *data)
{
	struct ann_synapse_list *l;

	if (in_synapse_func)
		for (l = neuron->inputs; l; l = l->next)
			if (in_synapse_func (l->synapse, data) == false)
				return false;
	if (out_synapse_func)
		for (l = neuron->outputs; l; l = l->next)
			if (out_synapse_func (l->synapse, data) == false)
				return false;
	return true;
}

bool ann_traverse_layer(const struct ann_layer *layer,
			bool (* neuron_func)(const struct ann_neuron *,
					     void *data),
			bool (* in_synapse_func)(const struct ann_synapse *,
						 void *data),
			bool (* out_synapse_func)(const struct ann_synapse *,
						  void *data),
			void *data)
{
	struct ann_neuron *n;

	if (neuron_func || in_synapse_func || out_synapse_func)
		for (n = layer->neuron_head; n; n = n->next) {
			if (neuron_func && neuron_func(n, data) == false)
				return false;
			if (ann_traverse_neuron(n, in_synapse_func,
						out_synapse_func,
						data) == false)
				return false;
	}
	return true;
}

bool ann_traverse_net(const struct ann_net *ann,
		      bool (* layer_func)(const struct ann_layer *,
					  void *data),
		      bool (* neuron_func)(const struct ann_neuron *,
					   void *data),
		      bool (* in_synapse_func)(const struct ann_synapse *,
					       void *data),
		      bool (* out_synapse_func)(const struct ann_synapse *,
						void *data),
		      void *data)
{
	struct ann_layer *l;

	if (layer_func || neuron_func || in_synapse_func || out_synapse_func)
		for (l = ann->layer_head; l; l = l->next) {
			if (layer_func && layer_func(l, data) == false)
				return false;
			if (ann_traverse_layer(l, neuron_func, in_synapse_func,
					       out_synapse_func,
					       data) == false)
				return false;
		}
	return true;
}

bool ann_print_synapse(const struct ann_synapse *synapse, void *data)
{
	printf("Synapse connected to layer %d neuron %d: %.2f",
	       synapse->to->layer->id, synapse->to->id, synapse->weight);
	if (synapse->label)
		printf(" %s", synapse->label);
	printf("\n");
	return true;
}

bool ann_print_neuron(const struct ann_neuron *neuron, void *data)
{
	printf("Neuron %d", neuron->id);
	if (neuron->label)
		printf(" %s", neuron->label);
	printf("\n");
	return true;
}

bool ann_print_layer(const struct ann_layer *layer, void *data)
{
	printf("Layer %d", layer->id);
	if (layer->label)
		printf(": %s", layer->label);
	printf("\n");
	return true;
}


/**
 * Prints values of all the layers, neurons and synapses of a neural network.
 *
 * @param net Neural network to print.
 */

void ann_print_net(const struct ann_net *ann)
{
	printf("Neural network:\t%s\n", ann->name);
	if (strlen(ann->description))
		printf("%s\n", ann->description);
	ann_traverse_net(ann, ann_print_layer, ann_print_neuron, NULL,
			 ann_print_synapse, NULL);
}

bool ann_check_synapse(const struct ann_synapse *synapse, void *data)
{
	struct ann_synapse_list *l;

	if (synapse == NULL) {
		fprintf(stderr, "Synapse should not be NULL.");
		return false;
	}
	l = synapse->to->inputs;
	while (l) {
		if (l->synapse == NULL)  {
			fprintf(stderr, "Synapse in neuron input list "
				"should not be NULL.");
			return false;
		}
		if (l->synapse == synapse)
			break;
		l = l->next;
	}
	if (l->synapse != synapse) {
		fprintf(stderr, "Synapse not found in neuron input list.");
		return false;
	}
	l = synapse->from->outputs;
	while (l) {
		if (l->synapse == NULL)  {
			fprintf(stderr, "Synapse in neuron input list "
				"should not be NULL.");
			return false;
		}
		if (l->synapse == synapse)
			break;
		l = l->next;
	}
	if (l->synapse != synapse) {
		fprintf(stderr, "Synapse not found in neuron output list.");
		return false;
	}
	return true;
}

bool ann_check_layer(const struct ann_layer *layer, void *data)
{
	if (layer->prev == NULL || layer->prev->next == layer) {
		return true;
	} else {
		fprintf(stderr, "Layer %d and %d not linked properly.\n",
			layer->id, layer->prev->id);
		return false;
	}
}

/**
 * Checks the integrity of a neural network.
 *
 * @param ann Neural network to check.
 * @param verbose True if information on neural network should be printed.
 *
 */

bool ann_check_net(const struct ann_net *ann)
{
	if (ann->layer_head && ann->layer_head->prev != NULL) {
		fprintf(stderr, "Neural network head layer is corrupted.\n");
		return false;
	}
	if (ann->layer_last && ann->layer_last->next != NULL) {
		fprintf(stderr, "Neural network last layer is corrupted.\n");
		return false;
	}
	return ann_traverse_net(ann,
				ann_check_layer,
				NULL,
				NULL,
				ann_check_synapse,
				NULL);
}

/**
 * Prints the firing values (outputs) of all the neurons in a layer.
 *
 * @param layer Layer, usually the last one, of the neural network whose neuron
 * outputs must be printed.
 *
 */

void ann_print_layer_outputs(struct ann_layer *layer)
{
	struct ann_neuron *neuron;
	for (neuron = layer->neuron_head; neuron; neuron = neuron->next)
		printf("Neuron:\t%d\tOutput:\t%.2f\n",
		       neuron->id, neuron->value);
}
