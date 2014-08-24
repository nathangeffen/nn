/**
 * @file
 * High-level structs and prototypes for initializing, modifying, running,
 * saving, loading, training and destroying a neural network.
 *
 * The philosophy behind this library is that each neuron can be a standalone
 * object, so lists instead of adjacency matrices have been used.
 *
 * @author Nathan Geffen
 * @copyright The BSD 2-Clause License
 */

#ifndef ANN_H
#define ANN_H

#include <gsl/gsl_rng.h>
#if __STDC_VERSION__ >= 199901L
#include <stdbool.h>
#else
#define bool int
#define false 0
#define true 1
#endif
#include "uthash/uthash.h"

// Handles errors. If exp is true then no error, else print msg and take action.


/* Firing functions */
#define ANN_NULL 0
#define ANN_INPUT 1
#define ANN_BIAS 2
#define ANN_SIGMOID 3

#define ANN_CHECK(exp, msg, action)					\
	do {								\
		if (! (exp) ) {						\
			fprintf(stderr, "%s. FILE %s. Line:%d\n",	\
				msg, __FILE__, __LINE__);		\
			action;						\
		}							\
	} while (0)

#define UNUSED(expr) do { (void)(expr); } while (0)

#define MIN_WEIGHT -10.0; // Default synapse minimum weight change
#define MAX_WEIGHT 10.0; // Default synapse maximum weight change

struct ann_synapse;
struct ann_synapse_list;
struct ann_neuron;
struct ann_neuron_hash;
struct ann_layer;
struct ann_net;

/**
 * Represents synapse that connects two neurons.
 */

struct ann_synapse {
	int id; ///< Unique identifier of synapse.
	char *label; ///< Optional label
	struct ann_neuron *from; ///< Neuron from which synapse comes.
	struct ann_neuron *to; ///< Neuron to which synapse goes.
	double weight; ///< Weight of the synapse.
};

/**
 * Holds a list of synapses. Used, for example, inside a neuron.
 */

struct ann_synapse_list {
	struct ann_synapse_list *next; ///< Next synapse in list.
	struct ann_synapse *synapse; ///< Synapse for this node.
};

/**
 * Represents a neuron, the fundamental constituent of a neural network.
 */

struct ann_neuron {
	int id; ///< With layer id, a unique identifier of neuron.
	char *label; ///< Optional label
	int synapse_ctr; ///< Used to set id of output synapses.
	struct ann_layer *layer;
	struct ann_neuron *next; ///< Neurons are stored in layers as lists.
	double value; ///< Calculated by neuron firing function.
	void *data; ///< Generic data for firing function to use.
	void (*fire_func) (struct ann_neuron *, void *);
	struct ann_synapse_list *inputs; ///< List of all input synapses.
	struct ann_synapse_list *outputs; ///< List of all output synapses.
};

/**
 * Keep track of neuron addresses in a hash table.
 */

struct ann_neuron_hash {
	char key[42];
	struct ann_neuron *neuron;
	UT_hash_handle hh;
};

/**
 * Represents a layer of neurons in a neural network.
 */

struct ann_layer {
	int id; ///< Unique identifier of this layer.
	char *label; ///< Optional label
	int neuron_ctr; ///< Used to assign unique id to each neuron in layer.
	int num_neurons; ///< Tracks number of neurons in layer.
	struct ann_layer *prev; ///< Pointer to previous layer.
	struct ann_layer *next; ///< Pointer to next layer.
	struct ann_neuron *neuron_head; ///< Head of list of neurons.
	gsl_rng *rng; ///< Random number generator.
	double min_weight; ///< Minimum weight change for synapse.
	double max_weight; ///< Maximum weight change for synapse.
};

/**
 * Represents an artificial neural network.
 */

struct ann_net {
	char *name; ///< Name of network. "Untitled" by default.
	char *description; ///< Description of network. Empty string by default.
	bool free_name;
	bool free_description;
	int layer_ctr; ///< Used to assign unique id to each layer in network.
	int num_layers; ///< Tracks number of layers in network.
	struct ann_layer *layer_head; ///< List of layers in the network.
        ///< Special layer with one connected to every non input neuron.
	struct ann_layer *layer_last;
	gsl_rng *rng; ///< Random number generator
	double min_weight; ///< Minimum weight change for synapse
	double max_weight; ///< Maximum weight change for synapse
};

// Prototypes

struct ann_net *ann_create_net(void);
char *ann_set_net_name(struct ann_net *ann, const char *name);
char *ann_set_net_description(struct ann_net *ann, const char *desc);
struct ann_synapse *ann_add_synapse(struct ann_neuron *from,
				    struct ann_neuron *to,
				    const double weight);
void ann_set_synapse_weight(struct ann_synapse *synapse, const double weight);
void ann_remove_synapse_from_neuron(struct ann_synapse_list **synapse_list,
				    const struct ann_synapse *synapse);
void ann_destroy_synapse(struct ann_synapse *synapse);
void ann_destroy_synapse_list(struct ann_synapse_list *synapse_list,
			      bool free_synapse);
struct ann_synapse *ann_get_synapse(const struct ann_neuron *from,
				    const struct ann_neuron *to);
struct ann_synapse *ann_get_synapse_or_null(const struct ann_neuron *from,
					    const struct ann_neuron *to);
double ann_sigmoid(const double);
void ann_neuron_fire_sigmoid(struct ann_neuron *neuron, void *data);
void ann_neuron_fire_input(struct ann_neuron *neuron, void *data);
void ann_neuron_fire_bias(struct ann_neuron *neuron, void *data);
struct ann_layer *ann_process_pattern(struct ann_net *ann,
				      const double inputs[],
				      const int num_inputs);
struct ann_neuron *ann_create_neuron(void);
void ann_destroy_synapse_lists(struct ann_neuron *neuron);
void ann_destroy_neuron(struct ann_neuron *neuron);
struct ann_neuron *ann_add_neuron(struct ann_layer *layer,
				  void (*fire_func)
				  (struct ann_neuron *,
				   void *),
				  void *data);
struct ann_neuron *ann_add_neurons(struct ann_layer *layer, int num_neurons,
				   void (*fire_func)
				   (struct ann_neuron *,
				    void *),
				   void *data);
struct ann_neuron *ann_get_neuron_by_id(const struct ann_net *ann,
					const int layer_id,
					const int neuron_id);
struct ann_neuron *ann_get_neuron_by_pos(const struct ann_net *ann,
					 const int layer_pos,
					 const int neuron_pos);
void ann_destroy_neuron_list(struct ann_neuron *head);
struct ann_layer *ann_create_layer(struct ann_layer *prev,
				   struct ann_layer *next,
				   gsl_rng *rng,
				   const double min_weight,
				   const double max_weight);
struct ann_layer *ann_add_layer(struct ann_net *ann);
struct ann_layer *ann_append_layer(struct ann_net *ann);
struct ann_layer *ann_insert_after_layer(struct ann_net *ann,
					 struct ann_layer *layer);
struct ann_layer *ann_prepend_layer(struct ann_net *ann);
struct ann_layer *new_ann_layer(struct ann_net *ann);
_Bool ann_connect_layers(struct ann_layer *from,
			 struct ann_layer *to);
void ann_destroy_layer(struct ann_layer *layer);
struct ann_net *ann_create_feed_forward_net(const int layers[],
					    const int num_layers);
void ann_destroy_net(struct ann_net *ann);
bool ann_check_net(const struct ann_net *ann);
void ann_print_layer_outputs(struct ann_layer *layer);
void ann_print_net(const struct ann_net *ann);
struct ann_net **ann_load(FILE *f, int *num_nets);
bool ann_save(FILE *f, struct ann_net *ann[], int num_anns);
bool ann_save_nets_bin(const char *filename,
		       struct ann_net *ann[],
		       const int num_anns);
struct ann_net **ann_load_nets_bin(const char *filename, int *num_nets);
#endif
