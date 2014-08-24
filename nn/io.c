/**
 * @file
 * Functions for saving and loading neural networks.
 *
 * @author Nathan Geffen
 * @copyright The BSD 2-Clause License
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "nn.h"
#include "cJSON/cJSON.h"

static bool ann_synapse_save(struct ann_synapse *synapse, cJSON *jarray)
{
	cJSON *jsynapse;
	char msg[200];

	cJSON_AddItemToArray(jarray, jsynapse=cJSON_CreateObject());
	sprintf(msg, "Error processing JSON for synapse: (%d %d)",
		synapse->from->id, synapse->to->id);
	ANN_CHECK(jsynapse, msg, return false);
	cJSON_AddNumberToObject(jsynapse, "layer-from",
				synapse->from->layer->id);
	cJSON_AddNumberToObject(jsynapse, "neuron-from", synapse->from->id);
	cJSON_AddNumberToObject(jsynapse, "weight", synapse->weight);

	return true;
}

static bool ann_neuron_save(struct ann_neuron *neuron, cJSON *jarray)
{
	cJSON *jneuron, *jsynapses;
	char fire_func[200];
	struct ann_synapse_list *it;
	char msg[200];

	cJSON_AddItemToArray(jarray, jneuron=cJSON_CreateObject());
	sprintf(msg, "Error processing JSON for neuron %d", neuron->id);
	ANN_CHECK(jneuron, msg, return false);
	cJSON_AddNumberToObject(jneuron, "neuron-id", neuron->id);
	if (neuron->fire_func == ann_neuron_fire_sigmoid)
		snprintf(fire_func, 200, "sigmoid");
	else if (neuron->fire_func == ann_neuron_fire_input)
		snprintf(fire_func, 200, "input");
	else if (neuron->fire_func == ann_neuron_fire_bias)
		snprintf(fire_func, 200, "bias");
	else
		snprintf(fire_func, 200, "unknown");
	cJSON_AddStringToObject(jneuron, "firing-function", fire_func);
	if (neuron->inputs) {
		cJSON_AddItemToObject(jneuron, "synapses",
				      jsynapses=cJSON_CreateArray());
		ANN_CHECK(jsynapses, "Error processing JSON for synapses",
			  return false);
		for (it = neuron->inputs; it; it = it->next)
			ANN_CHECK(ann_synapse_save(it->synapse, jsynapses),
				  "Error processing JSON for synapses",
				  return false);
	}
	return true;
}

static bool ann_layer_save(struct ann_layer *layer, cJSON *jarray)
{
	cJSON *jlayer, *jneurons;
	struct ann_neuron *neuron;
	char msg[200];

	cJSON_AddItemToArray(jarray, jlayer=cJSON_CreateObject());
	sprintf(msg, "Error processing JSON for layer %d", layer->id);
	ANN_CHECK(jlayer, msg, return false);
	cJSON_AddNumberToObject(jlayer, "layer-id", layer->id);
	if (layer->neuron_head) {
		cJSON_AddItemToObject(jlayer, "neurons",
				      jneurons=cJSON_CreateArray());
		ANN_CHECK(jneurons, "Error processing JSON for neurons",
			  return false);
		for (neuron = layer->neuron_head; neuron;
		     neuron = neuron->next)
			ANN_CHECK(ann_neuron_save(neuron, jneurons),
				  "Error processing JSON for neurons",
				  return false);
	}
	return true;
}

static bool ann_net_save(struct ann_net *ann, cJSON *json)
{
	cJSON *jnet, *jlayers;
	struct ann_layer *layer;

	cJSON_AddItemToArray(json, jnet=cJSON_CreateObject());
	ANN_CHECK(jnet, "Error creating JSON object.", return false);
	cJSON_AddStringToObject(jnet, "ann-name", ann->name);
	if (ann->description && strlen(ann->description) > 0)
		cJSON_AddStringToObject(jnet, "ann-description",
					ann->description);
	if (ann->layer_head) {
		cJSON_AddItemToObject(jnet, "layers",
				      jlayers=cJSON_CreateArray());
		ANN_CHECK(jlayers, "Error creating JSON for layers",
			  return false);
		for (layer = ann->layer_head; layer; layer = layer->next)
			ANN_CHECK(ann_layer_save(layer, jlayers),
				  "Error processing JSON for layers",
				  return false);
	}
	return true;
}

bool ann_save(FILE *f, struct ann_net *ann[], int num_anns)
{
	cJSON *root;
	char *rendered;
	char msg[200];

	if (num_anns > 0) {
		root = cJSON_CreateArray();
		ANN_CHECK(root, "Cannot start JSON processing", return false);
		for (int i = 0; i < num_anns; ++i) {
			sprintf(msg, "Failed to convert net number %d to JSON", i);
			ANN_CHECK(ann_net_save(ann[i], root),
				  msg,
				  {
					  cJSON_Delete(root);
					  return false;
				  });
		}
		rendered = cJSON_Print(root);
		fprintf(f, "%s\n", rendered);
		free(rendered);
		cJSON_Delete(root);
	}
	return true;
}


/*
 * Code for this taken and adapted from Stack Overflow:
http://stackoverflow.com/questions/174531/easiest-way-to-get-files-contents-in-c
 */

static char *ann_read_file_into_string(FILE *f)
{
	char *buffer = 0;
	long length;

	ANN_CHECK(fseek(f, 0, SEEK_END) == 0, "Error reading file",
		  return NULL);
	ANN_CHECK(length = ftell (f), "Error getting file length", return NULL);
	ANN_CHECK(fseek (f, 0, SEEK_SET) == 0, "Error rewinding file",
		  return NULL);
	ANN_CHECK(buffer = malloc (length),
		  "Cannot allocate string to receive file", return NULL);
	ANN_CHECK(fread (buffer, 1, length, f),  "Cannot read file into string",
		  {
			  free(buffer);
			  return NULL;
		  });
	return buffer;
}

static bool ann_load_synapses(cJSON *json,
			      struct ann_net *ann,
			      struct ann_neuron_hash **hash)
{
	cJSON *json_layer, *json_neuron, *json_neurons, *json_synapses,
		*json_synapse, *json_layer_from,
		*json_neuron_from, *json_weight;
	int num_layers, num_neurons, num_synapses, layer_from_id, layer_to_id,
		neuron_from_id, neuron_to_id;
	double weight;
	struct ann_neuron_hash *h, *t;
	struct ann_neuron *neuron_to, *neuron_from;
	char key[42];

	// We can avoid a lot of error checking in this function because it
	// occurs in a second pass and many errors would have already been
	// found.

	num_layers = cJSON_GetArraySize(json);

	for (int i = 0; i < num_layers; ++i) {
		json_layer = cJSON_GetArrayItem(json, i);
		layer_to_id = cJSON_GetObjectItem(json_layer,
						  "layer-id")->valueint;
		json_neurons = cJSON_GetObjectItem(json_layer, "neurons");
		num_neurons = json_neurons ? cJSON_GetArraySize(json_neurons)
			: 0;
		for (int j = 0; j < num_neurons; ++j) {
			json_neuron = cJSON_GetArrayItem(json_neurons, j);
			neuron_to_id = cJSON_GetObjectItem(
				json_neuron, "neuron-id")->valueint;
			sprintf(key, "%d,%d", layer_to_id, neuron_to_id);
			HASH_FIND_STR(*hash, key, h);
			neuron_to = h->neuron;
			json_synapses = cJSON_GetObjectItem(json_neuron,
							    "synapses");
			if (json_synapses == NULL) continue;
			num_synapses = cJSON_GetArraySize(json_synapses);
			for (int k = 0; k < num_synapses; ++k) {
				json_synapse =
					cJSON_GetArrayItem(json_synapses, k);
				json_layer_from =
					cJSON_GetObjectItem(json_synapse,
							    "layer-from");
				json_neuron_from =
					cJSON_GetObjectItem(json_synapse,
							    "neuron-from");
				json_weight = cJSON_GetObjectItem(json_synapse,
							    "weight");
				ANN_CHECK(json_layer_from && json_neuron_from
					  && json_weight,
					  "Missing synapse elements",
					  return false);
				layer_from_id = json_layer_from->valueint;
				neuron_from_id = json_neuron_from->valueint;
				weight = json_weight->valuedouble;
				sprintf(key, "%d,%d", layer_from_id,
					neuron_from_id);
				HASH_FIND_STR(*hash, key, h);
				ANN_CHECK(h, "Neuron for synapse not found",
					  return false);
				neuron_from = h->neuron;
				ANN_CHECK(ann_add_synapse(neuron_from,
							  neuron_to, weight),
					  "Could not connect neurons",
					  return false);
			}
		}
	}
	// Delete the hash table. No longer needed.
	HASH_ITER(hh, *hash, h, t) {
		HASH_DEL(*hash, h);
		free(h);
	}
	return true;
}

static bool ann_insert_neuron_hash(char key[],
				   struct ann_neuron *neuron,
				   struct ann_neuron_hash **hash)
{
	struct ann_neuron_hash *h;
	char msg[200];

	HASH_FIND_STR(*hash, key, h);
	sprintf(msg, "Duplicate id for neuron (%s)", key);
	ANN_CHECK(h == NULL, msg, return false);
	h = malloc(sizeof(*h));
	strcpy(h->key, key);
	HASH_ADD_STR(*hash, key, h);
	h->neuron = neuron;

	return true;
}

static bool ann_load_neuron(cJSON *json_neuron,
			    struct ann_neuron *neuron,
			    struct ann_neuron_hash **hash)
{
	cJSON *key;
	char hash_key[42];

	ANN_CHECK(key = cJSON_GetObjectItem(json_neuron, "neuron-id"),
		  "Neurons must have neuron-id", return false);
	neuron->id = key->valueint;
	key = cJSON_GetObjectItem(json_neuron, "firing-function");
	if (key) {
		if (strcmp(key->valuestring, "sigmoid") == 0)
			neuron->fire_func = ann_neuron_fire_sigmoid;
		else if (strcmp(key->valuestring, "bias") == 0)
			neuron->fire_func = ann_neuron_fire_bias;
		else
			neuron->fire_func = ann_neuron_fire_input;
	} else {
		neuron->fire_func = ann_neuron_fire_input;
	}
	sprintf(hash_key, "%d,%d", neuron->layer->id, neuron->id);

	return ann_insert_neuron_hash(hash_key, neuron, hash);
}

static bool ann_load_layer(cJSON *json_layer,
			   cJSON *json_neurons,
			   int num_neurons,
			   struct ann_net *ann,
			   struct ann_neuron_hash **hash)
{
	struct ann_layer *layer;
	struct ann_neuron *neuron;
	cJSON *key, *json_neuron;


	layer = ann_add_layer(ann);
	ANN_CHECK(layer, "Could not add layer.", return false);
	ANN_CHECK(key = cJSON_GetObjectItem(json_layer, "layer-id"),
		  "No layer-id for layer", return false);
	layer->id = key->valueint;
	ann->layer_ctr = (layer->id > ann->layer_ctr)
		? layer->id + 1 : ann->layer_ctr;
	if (num_neurons) {
		neuron = ann_add_neurons(layer, num_neurons,
					 ann_neuron_fire_input, false);
		ANN_CHECK(neuron, "Could not add neurons", return false);
		for (int i = 0; i < num_neurons; ++i, neuron = neuron->next) {
			json_neuron = cJSON_GetArrayItem(json_neurons, i);
			ann_load_neuron(json_neuron, neuron, hash);
		}
	}
	return true;
}

static struct ann_net *ann_load_net(cJSON *json_ann)
{
	struct ann_net *ann;
	cJSON *key, *json_layer, *json_neurons;
	int num_layers, num_neurons, total_neurons = 0;
	struct ann_neuron_hash *hash = NULL;

	ANN_CHECK(ann = ann_create_net(), "Could not create net", return NULL);
	if ( (key = cJSON_GetObjectItem(json_ann, "ann-name")) &&
	     key->valuestring)
		ann_set_net_name(ann, key->valuestring);
	if ( (key = cJSON_GetObjectItem(json_ann, "ann-description")) &&
	     key->valuestring)
		ann_set_net_description(ann, key->valuestring);
	if ( (key = cJSON_GetObjectItem(json_ann, "layers")) ) {
		num_layers = cJSON_GetArraySize(key);
		// We make two passes through the layers: Once to load
		// the layers and and neurons, and once to process neurons
		// and their synapses. We have to do it in two passes
		// because we don't want to try to connect a neuron to one
		// that doesn't yet exist.
		for (int i = 0; i < num_layers; ++i) {
			json_layer = cJSON_GetArrayItem(key, i);
			json_neurons = cJSON_GetObjectItem(json_layer,
							   "neurons");
			num_neurons = json_neurons
				? cJSON_GetArraySize(json_neurons) : 0;
			ANN_CHECK(ann_load_layer(json_layer, json_neurons,
						 num_neurons, ann, &hash),
				  "Could not load layer.",
				  {ann_destroy_net(ann); return NULL;});
			total_neurons += num_neurons;
		}
		ANN_CHECK(ann_load_synapses(key, ann, &hash),
			  "Could not load synapses",
			  {ann_destroy_net(ann); return NULL;});
	}
	return ann;
}

struct ann_net **ann_load(FILE *f, int *num_nets)
{
	struct ann_net ** anns;
	struct ann_net *ann;
	char *s;
	cJSON *root, *json_ann;
	char msg[200];

	ANN_CHECK(s = ann_read_file_into_string(f),
		  "Can't process file into string", return NULL);
	ANN_CHECK(root = cJSON_Parse(s), "Cannot parse file",
		{free(s); return NULL; });
	*num_nets = cJSON_GetArraySize(root);
	anns = malloc(*num_nets * sizeof(*anns));

	for (int i = 0; i < *num_nets; ++i) {
		sprintf(s, "Could not load net %d", i);
		ANN_CHECK( (json_ann = cJSON_GetArrayItem(root, i)) &&
			   (ann = ann_load_net(json_ann)), msg,
			   {
				   for (int j = 0; j < i; ++j)
					   ann_destroy_net(anns[j]);
				   free(anns); free(root); free(s);
				   *num_nets = 0;
				   return NULL;
			   });
		anns[i] = ann;
	}
	free(s);
	cJSON_Delete(root);
	return anns;
}



bool
ann_save_synapses_bin(FILE *f,
		     const struct ann_synapse_list *synapses)
{
	const struct ann_synapse_list *l;
	const struct ann_synapse *s;

	for (l = synapses; l; l = l->next) {
		s = l->synapse;
		if (fwrite(&s->from->layer->id, sizeof(int), 1, f) == 0)
			return false;
		if (fwrite(&s->from->id, sizeof(int), 1, f) == 0)
			return false;
		if (fwrite(&s->to->layer->id, sizeof(int), 1, f) == 0)
			return false;
		if (fwrite(&s->to->id, sizeof(int), 1, f) == 0)
			return false;
		if (fwrite(&s->weight, sizeof(double), 1, f) == 0)
			return false;
	}
	return true;
}

bool
ann_save_neurons_bin(FILE *f,
		     const struct ann_neuron *neuron)
{
	const struct ann_neuron *n;
	int i;

	for (n = neuron; n; n = n->next) {
		if (fwrite(&n->id, sizeof(int), 1, f) == 0)
			return false;
		if (neuron->fire_func == ann_neuron_fire_input)
			i = ANN_INPUT;
		else if (n->fire_func == ann_neuron_fire_bias)
			i = ANN_BIAS;
		else if (n->fire_func == ann_neuron_fire_sigmoid)
			i = ANN_SIGMOID;
		else
			i = ANN_NULL;
		if (fwrite(&i, sizeof(int), 1, f) == 0)
			return false;
	}
	return true;
}

bool
ann_save_layers_bin(FILE *f,
		    const struct ann_layer *layer)
{
	const struct ann_layer *l;

	for (l = layer; l; l = l->next) {
		if (fwrite(&l->id, sizeof(int), 1, f) == 0)
			return false;
		if (fwrite(&l->num_neurons, sizeof(int), 1, f) == 0)
			return false;
	}
	for (l = layer; l; l = l->next) {
		if (ann_save_neurons_bin(f, l->neuron_head) == false)
			return false;
	}
	return true;
}


bool
ann_save_net_bin(FILE *f,
		 const struct ann_net *ann)
{
	struct ann_layer *l;
	struct ann_neuron *n;
	int i;

	/* First save all information except the synapses. */
	if (ann->name)
		i = strlen(ann->name);
	else
		i = 0;
	if (fwrite(&i, sizeof(int), 1, f) == 0)
		return false;
	if (i)
		if (fwrite(ann->name, sizeof(char), i, f) == 0)
			return false;
	if (ann->description)
		i = strlen(ann->description);
	else
		i = 0;
	if (fwrite(&i, sizeof(int), 1, f) == 0)
		return false;
	if (i)
		if (fwrite(ann->description, sizeof(char), i, f) == 0)
			return false;
	if (fwrite(&ann->num_layers, sizeof(int), 1, f) == 0)
		return false;
	if (ann_save_layers_bin(f, ann->layer_head) == false)
		return false;

	/* Now save the synapses. */
	for (l = ann->layer_head; l; l = l->next)
		for (n = l->neuron_head; n; n = n->next)
			if (ann_save_synapses_bin(f, n->outputs) == false)
				return false;
	return true;
}

bool
ann_save_nets_bin(const char *filename,
		  struct ann_net *ann[],
		  const int num_anns)
{
	FILE *f;
	int i;

	f = fopen(filename, "wb");
	if (f == NULL)
		goto fail;
	if (fwrite(&num_anns, sizeof(num_anns), 1, f) == 0)
		goto fail_close;
	for (i = 0; i < num_anns; ++i)
		if (ann_save_net_bin(f, ann[i]) == false)
			goto fail_close;

/* no error */
	fclose(f);
	return true;

/* error */
fail_close:
	fclose(f);
fail:
	return false;
}

/**
 num_anns
 len_name
 ann_name
 len_description
 ann_description
 num_layers
{(repeated num_layers)
 layer_id
 num_neurons_layer_x
}
{ (repeated for num_layers for num_neurons_layer_x)
neuron-id
firing_function (int)
}
 { (repeated until end of file)
 layer_from
 neuron_from
 layer_to
 neuron_to
 weight
 }
 */

bool
ann_load_synapses_bin(FILE *f,
		      struct ann_net *ann,
		      struct ann_neuron_hash **hash)
{
	struct ann_neuron_hash *h;
	struct ann_neuron *from, *to;
	struct syn_rec {
		int l_f, n_f, l_t, n_t;
		double w;
	} s;
	char key[42];

	while (fread(&s, sizeof(s), 1, f)) {
		sprintf(key, "%d,%d", s.l_f, s.n_f);
		HASH_FIND_STR(*hash, key, h);
		if (h == NULL)
			return false;
		from = h->neuron;
		sprintf(key, "%d,%d", s.l_t, s.n_t);
		HASH_FIND_STR(*hash, key, h);
		if (h == NULL)
			return false;
		to = h->neuron;
		if (ann_add_synapse(from, to, s.w) == NULL)
			return false;
	}
	if (feof(f) == false)
		return false;

	return true;
}

bool
ann_load_layer_bin(FILE *f,
		   struct ann_net *ann)
{
	int i, j;
	struct ann_layer *l;
	struct ann_neuron *n;

	if (fread(&i, sizeof(int), 1, f) == 0)
		return false;
	if (fread(&j, sizeof(int), 1, f) == 0)
		return false;
	l = ann_add_layer(ann);
	if (l == NULL)
		return false;
	l->id = i;
	n = ann_add_neurons(l, j, NULL, NULL);
	if (n == NULL)
		return false;

	return true;

}



bool
ann_load_layers_bin(FILE *f,
		    struct ann_net *ann,
		    struct ann_neuron_hash **hash)
{
	int i, j, num_layers;
	struct ann_layer *l;
	struct ann_neuron *n;
	void (*fire_func) (struct ann_neuron *, void *);
	char key[42];

	if (fread(&num_layers, sizeof(int), 1, f) == 0)
		goto fail;
	for (i = 0; i < num_layers; ++i)
		if (ann_load_layer_bin(f, ann) == false)
			goto fail;
	for (l = ann->layer_head; l; l = l->next)
		for (n = l->neuron_head; n; n = n->next) {
			if (fread(&i, sizeof(int), 1, f) == 0)
				goto fail;
			if (fread(&j, sizeof(int), 1, f) == 0)
				goto fail;
			switch(j) {
			case ANN_INPUT:
				fire_func = ann_neuron_fire_input;
				break;
			case ANN_BIAS:
				fire_func = ann_neuron_fire_bias;
				break;
			case ANN_SIGMOID:
				fire_func = ann_neuron_fire_sigmoid;
				break;
			default:
				fire_func = NULL;
			}
			n->id = i;
			n->fire_func = fire_func;
			sprintf(key, "%d,%d", n->layer->id, n->id);
			if (ann_insert_neuron_hash(key, n, hash) == false)
				goto fail;
		}

/* No error */
	return true;

/* Error */
fail:
	return false;
}

struct ann_net *
ann_load_net_bin(FILE *f)
{
	int i;
	char *s;
	struct ann_net *ann;
	struct ann_neuron_hash *hash = NULL, *h, *t;

	ann = ann_create_net();
	if (ann == NULL)
		goto fail;

	/* ANN name */
	if (fread(&i, sizeof(int), 1, f) == 0)
		goto fail_destroy;
	if (i) {
		s = malloc(sizeof(char) * (i + 1) );
		if (s == NULL)
			goto fail_destroy;
		if (fread(s, sizeof(char), i, f) < i)
			goto fail_str;
		s[i] = '\0';
		ann_set_net_name(ann, s);
		free(s);
	}

	/* ANN description */
	if (fread(&i, sizeof(int), 1, f) == 0)
		goto fail_destroy;
	if (i) {
		s = malloc(sizeof(char) * (i + 1) );
		if (s == NULL)
			goto fail_destroy;
		if (fread(s, sizeof(char), i, f) < i)
			goto fail_str;
		s[i] = '\0';
		ann_set_net_description(ann, s);
		free(s);
	}

	if (ann_load_layers_bin(f, ann, &hash) == false)
		goto fail_destroy;

	if (ann_load_synapses_bin(f, ann, &hash) == false)
		goto fail_destroy;

	// Delete the hash table. No longer needed.
	HASH_ITER(hh, hash, h, t) {
		HASH_DEL(hash, h);
		free(h);
	}

/* No errors */
	return ann;

/* Error */
fail_str:
	free(s);
fail_destroy:
	ann_destroy_net(ann);
fail:
	return NULL;
}

struct ann_net **
ann_load_nets_bin(const char *filename,
		  int *num_nets)
{
	struct ann_net **anns, *ann;
	FILE *f;
	int i;

	f = fopen(filename, "rb");
	if (f == NULL)
		goto fail;
	if (fread(num_nets, sizeof(int), 1, f) == 0)
		goto fail_close;
	anns = malloc(sizeof(*anns) * *num_nets);
	if (anns == NULL)
		goto fail_close;
	for (i = 0; i < *num_nets; ++i) {
		ann = ann_load_net_bin(f);
		if (ann == NULL) {
			*num_nets = i;
			goto fail_ann;
		}
		anns[i] = ann;
	}

/* no error */
	fclose(f);
	return anns;

/* error */
fail_ann:
	for (i = 0; i < *num_nets; ++i)
		ann_destroy_net(anns[i]);

fail_close:
	fclose(f);

fail:
	return NULL;
}
